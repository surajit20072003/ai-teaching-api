"""
routers/documents.py — Document upload, list, status, and delete endpoints.
"""
import uuid, logging, asyncio
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func, text

from db.models import get_db, Document, DocumentChunk, DocumentQuestion, TeachingCache, AsyncSessionLocal
from core.document_processor import process_document
from core.local_storage import (
    write_doc_meta, delete_document_files, delete_cache_files,
    append_log, check_storage_available,
)
from core.b2_client import upload_to_b2

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}
MAX_FILE_SIZE_MB   = 50


# ── POST /documents/upload ─────────────────────────────────────────────────────

@router.post("/upload", summary="Upload a document (admin)")
async def upload_document(
    background_tasks: BackgroundTasks,
    subject_id: str  = Form(...),
    chapter_id: Optional[str] = Form(None),
    topic_id:   Optional[str] = Form(None),
    title:      str  = Form(...),
    language:   str  = Form("hi-IN"),
    questions:  str  = Form(""),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported type '{ext}'. Use: {', '.join(ALLOWED_EXTENSIONS)}")

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f}MB). Max {MAX_FILE_SIZE_MB}MB.")

    storage = check_storage_available()
    if storage["status"] == "error":
        raise HTTPException(503, f"Local storage unavailable: {storage.get('error')}")

    doc_id        = str(uuid.uuid4())
    safe_filename = file.filename.replace(" ", "_")

    # Process: extract + chunk + embed + save locally
    try:
        result = await process_document(subject_id, doc_id, safe_filename, file_bytes)
    except ValueError as e:
        raise HTTPException(422, str(e))

    # Upload raw file to B2 (non-fatal if fails)
    b2_url = None
    try:
        b2_url = await upload_to_b2(
            file_bytes,
            f"documents/{subject_id}/{doc_id}/{safe_filename}",
            file.content_type or "application/octet-stream",
        )
    except Exception as e:
        logger.warning(f"[upload] B2 failed (non-fatal): {e}")

    # Insert Document row
    doc = Document(
        id=uuid.UUID(doc_id), subject_id=subject_id, 
        chapter_id=chapter_id, topic_id=topic_id,
        title=title, filename=safe_filename,
        local_raw_path=result["local_raw_path"],
        local_processed_path=result["local_processed_path"],
        b2_url=b2_url, total_chunks=result["total_chunks"],
        status="processing", language=language,
    )
    db.add(doc)
    await db.flush()

    # Insert chunks
    for chunk in result["chunks"]:
        db.add(DocumentChunk(
            document_id=uuid.UUID(doc_id), subject_id=subject_id,
            chunk_index=chunk["chunk_index"],
            section_title=chunk.get("section_title"),
            chunk_text=chunk["chunk_text"],
            chunk_embedding=chunk.get("embedding"),
        ))

    # Insert admin questions
    from core.cache import hash_question as _hash_question
    question_list = [q.strip() for q in questions.split("\n") if q.strip()]
    for q_text in question_list:
        # Save to DocumentQuestion
        db.add(DocumentQuestion(
            document_id=uuid.UUID(doc_id), subject_id=subject_id,
            question_text=q_text, is_pregen_done=False,
        ))
        # Queue in teaching_qa_cache using the SAME hash function as main.py
        # (previously used MD5 which never matched SHA256 lookups → always cache miss)
        q_hash = _hash_question(q_text)
        new_q_id = str(uuid.uuid4())
        await db.execute(
            text("""
                INSERT INTO teaching_qa_cache 
                (id, subject_id, chapter_id, topic_id, question_hash, question_text, variation_number, pregen_status)
                VALUES (:id, :subject_id, :chapter_id, :topic_id, :question_hash, :question, 1, 'pending')
                ON CONFLICT (question_hash, subject_id, variation_number) DO UPDATE SET pregen_status = 'pending'
            """),
            {"id": new_q_id, "subject_id": subject_id, "chapter_id": chapter_id, "topic_id": topic_id, "question_hash": q_hash, "question": q_text}
        )

    doc.pregen_total = len(question_list)
    await db.commit()

    # Save meta.json locally
    await write_doc_meta(subject_id, doc_id, {
        "id": doc_id, "subject_id": subject_id, "title": title,
        "filename": safe_filename, "total_chunks": result["total_chunks"],
        "total_chars": result["total_chars"], "pregen_total": len(question_list),
        "b2_url": b2_url, "language": language,
    })
    await append_log("uploads", f"doc={doc_id} subject={subject_id} title={title!r} chunks={result['total_chunks']}")

    # Launch pre-generation background task
    background_tasks.add_task(_launch_pregen, doc_id, subject_id)

    return {
        "success": True, "document_id": doc_id, "title": title,
        "subject_id": subject_id, "total_chunks": result["total_chunks"],
        "total_chars": result["total_chars"], "questions_queued": len(question_list),
        "b2_url": b2_url, "status": "processing",
        "message": f"Uploaded. Pre-generating {len(question_list)} Q&A pairs in background.",
    }


async def _launch_pregen(doc_id: str, subject_id: str):
    """
    Background task after document upload:
    1. Fetch all chunk texts for this document
    2. Call predict_questions() to generate 20 AI-predicted questions (8.3)
    3. Mark document status = 'ready' (pre-gen batch runs separately via /pregen/start)
    """
    try:
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select, update
            from db.models import DocumentChunk, Document

            # Get chunk texts for prediction
            rows = (await db.execute(
                select(DocumentChunk.chunk_text)
                .where(DocumentChunk.document_id == uuid.UUID(doc_id))
                .order_by(DocumentChunk.chunk_index)
            )).scalars().all()
            chunk_texts = list(rows)

            if chunk_texts:
                # 8.3: predict 20 student questions from document content
                # User requested to disable predict_questions for now
                # from core.pregen import predict_questions
                # n = await predict_questions(doc_id, subject_id, chunk_texts, AsyncSessionLocal)
                # logger.info(f"[pregen] Predicted {n} questions for doc={doc_id}")
                n = 0

                # Update pregen_total to include predicted questions
                await db.execute(
                    update(Document)
                    .where(Document.id == uuid.UUID(doc_id))
                    .values(pregen_total=Document.pregen_total + n, status="ready")
                )
            else:
                await db.execute(
                    update(Document)
                    .where(Document.id == uuid.UUID(doc_id))
                    .values(status="ready")
                )
            await db.commit()
            logger.info(f"[pregen] doc={doc_id} marked ready. Use POST /pregen/start to begin batch.")
    except Exception as e:
        logger.error(f"[pregen] Background task failed for doc={doc_id}: {e}")
        try:
            async with AsyncSessionLocal() as db:
                from sqlalchemy import update
                from db.models import Document
                await db.execute(
                    update(Document)
                    .where(Document.id == uuid.UUID(doc_id))
                    .values(status="ready")  # mark ready even on predict failure
                )
                await db.commit()
        except Exception:
            pass



# ── GET /documents ─────────────────────────────────────────────────────────────

@router.get("", summary="List documents for a subject")
async def list_documents(
    subject_id: str,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    if not subject_id:
        raise HTTPException(400, "subject_id is required")
    stmt = select(Document).where(Document.subject_id == subject_id)
    if status:
        stmt = stmt.where(Document.status == status)
    stmt = stmt.order_by(Document.created_at.desc())
    docs = (await db.execute(stmt)).scalars().all()

    return {
        "subject_id": subject_id, "total": len(docs),
        "documents": [
            {
                "id": str(d.id), "title": d.title, "filename": d.filename,
                "status": d.status, "total_chunks": d.total_chunks,
                "pregen_done": d.pregen_done, "pregen_total": d.pregen_total,
                "pregen_pct": round(d.pregen_done / d.pregen_total * 100) if d.pregen_total else 0,
                "language": d.language, "b2_url": d.b2_url,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in docs
        ],
    }


# ── GET /documents/{id}/status ─────────────────────────────────────────────────

@router.get("/{doc_id}/status", summary="Pre-generation status")
async def get_document_status(doc_id: str, db: AsyncSession = Depends(get_db)):
    doc = (await db.execute(
        select(Document).where(Document.id == uuid.UUID(doc_id))
    )).scalar_one_or_none()
    if not doc:
        raise HTTPException(404, f"Document {doc_id} not found")

    done_count = (await db.execute(
        select(func.count()).where(
            DocumentQuestion.document_id == uuid.UUID(doc_id),
            DocumentQuestion.is_pregen_done == True,
        )
    )).scalar() or 0

    return {
        "document_id": doc_id, "title": doc.title, "subject_id": doc.subject_id,
        "status": doc.status, "total_chunks": doc.total_chunks,
        "pregen_done": done_count, "pregen_total": doc.pregen_total,
        "pregen_pct": round(done_count / doc.pregen_total * 100) if doc.pregen_total else 100,
        "is_ready": doc.status == "ready", "language": doc.language,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }


# ── DELETE /documents/{id} ─────────────────────────────────────────────────────

@router.delete("/{doc_id}", summary="Delete document and all its data")
async def delete_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    doc = (await db.execute(
        select(Document).where(Document.id == uuid.UUID(doc_id))
    )).scalar_one_or_none()
    if not doc:
        raise HTTPException(404, f"Document {doc_id} not found")

    subject_id = doc.subject_id
    delete_document_files(subject_id, doc_id)

    cache_entries = (await db.execute(
        select(TeachingCache).where(TeachingCache.document_id == uuid.UUID(doc_id))
    )).scalars().all()
    for entry in cache_entries:
        delete_cache_files(subject_id, str(entry.id))

    # B2 cleanup (best effort)
    if doc.b2_url:
        try:
            import boto3, os as _os
            from botocore.config import Config
            client = boto3.client(
                "s3",
                endpoint_url=_os.getenv("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com"),
                aws_access_key_id=_os.getenv("B2_KEY_ID", ""),
                aws_secret_access_key=_os.getenv("B2_APP_KEY", ""),
                config=Config(signature_version="s3v4"),
            )
            client.delete_object(
                Bucket=_os.getenv("B2_BUCKET", "simplelecture-media"),
                Key=f"documents/{subject_id}/{doc_id}/{doc.filename}",
            )
        except Exception as e:
            logger.warning(f"[delete] B2 cleanup failed (non-fatal): {e}")

    await db.execute(delete(Document).where(Document.id == uuid.UUID(doc_id)))
    await db.commit()
    await append_log("uploads", f"DELETED doc={doc_id} subject={subject_id} title={doc.title!r}")

    return {
        "success": True, "document_id": doc_id,
        "message": f"Document '{doc.title}' and all data deleted.",
        "cache_entries_cleaned": len(cache_entries),
    }
