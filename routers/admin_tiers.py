"""
routers/admin_tiers.py
──────────────────────
Admin endpoints to label documents as 'free' or 'pro'.
Only accessible with the admin API key.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from db.models import get_db

router = APIRouter(prefix="/admin", tags=["admin"])

ADMIN_KEY = "pramod2003@@"


def _require_admin(request: Request):
    key = request.headers.get("X-API-Key", "")
    if key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Admin access required")


@router.patch("/documents/{doc_id}/access-tier")
async def set_document_tier(
    doc_id: str,
    body: dict,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Set a document (and all its chunks) as 'free' or 'pro'.
    Body: { "tier": "free" }  or  { "tier": "pro" }
    """
    _require_admin(request)
    tier = (body.get("tier") or "").strip().lower()
    if tier not in ("free", "pro"):
        raise HTTPException(status_code=400, detail="tier must be 'free' or 'pro'")

    result = await db.execute(
        text("UPDATE documents SET access_tier = :tier WHERE id::text = :id RETURNING id, title"),
        {"tier": tier, "id": doc_id}
    )
    doc = result.mappings().first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Propagate to chunks (denormalized copy)
    await db.execute(
        text("UPDATE document_chunks SET access_tier = :tier WHERE document_id::text = :id"),
        {"tier": tier, "id": doc_id}
    )
    await db.commit()
    return {"doc_id": doc_id, "title": doc["title"], "access_tier": tier, "ok": True}


@router.patch("/chapters/{chapter_id}/access-tier")
async def set_chapter_tier(
    chapter_id: str,
    body: dict,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Set ALL documents in a chapter as 'free' or 'pro'.
    Body: { "tier": "free" }
    """
    _require_admin(request)
    tier = (body.get("tier") or "").strip().lower()
    if tier not in ("free", "pro"):
        raise HTTPException(status_code=400, detail="tier must be 'free' or 'pro'")

    r = await db.execute(
        text("UPDATE documents SET access_tier = :tier WHERE chapter_id = :cid RETURNING id"),
        {"tier": tier, "cid": chapter_id}
    )
    doc_ids = [str(row[0]) for row in r.fetchall()]
    if not doc_ids:
        raise HTTPException(status_code=404, detail="No documents found for this chapter_id")

    await db.execute(
        text("UPDATE document_chunks SET access_tier = :tier WHERE document_id IN (SELECT id FROM documents WHERE chapter_id = :cid)"),
        {"tier": tier, "cid": chapter_id}
    )
    await db.commit()
    return {"chapter_id": chapter_id, "docs_updated": len(doc_ids), "access_tier": tier, "ok": True}


@router.get("/subjects/{subject_id}/documents")
async def list_documents_with_tiers(
    subject_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """List all documents for a subject showing their current access tier."""
    _require_admin(request)
    rows = await db.execute(
        text("""
            SELECT id::text, title, access_tier, status, chapter_id::text,
                   (SELECT COUNT(*) FROM document_chunks WHERE document_id = documents.id) AS chunk_count
            FROM documents
            WHERE subject_id = :subj
            ORDER BY access_tier, title
        """),
        {"subj": subject_id}
    )
    docs = [dict(r) for r in rows.mappings().all()]
    free_count = sum(1 for d in docs if d["access_tier"] == "free")
    pro_count  = sum(1 for d in docs if d["access_tier"] == "pro")
    return {
        "subject_id": subject_id,
        "total": len(docs),
        "free_docs": free_count,
        "pro_docs": pro_count,
        "documents": docs,
    }
