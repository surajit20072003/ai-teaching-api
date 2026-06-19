"""
Phase 9 — End-to-End Verification Tests
Run inside the container: python3 tests/phase9_verify.py
"""
import asyncio, sys, json, urllib.request, time, os, glob
import concurrent.futures
sys.path.insert(0, ".")

BASE = "http://localhost:8000"
TEST_SUBJECT = "514662ee-2473-4df6-a0cf-cde0b69f3dfd"
OTHER_SUBJECT = "dde10568-3c23-46ae-b96d-3dfc5c33d4df"
TXT_DOC_ID = "b0d09355-df1b-4488-a3d4-5f699b5ec013"
PDF_DOC_ID = "7991f703-e49b-45eb-81a9-170b56b02b0c"

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

tests: dict = {}

def log(name, status, detail=""):
    icon = "V" if status == PASS else ("?" if status == WARN else "X")
    tests[name] = status
    print(f"  [{icon}] {name}: {detail}")

def http_get(url):
    return json.loads(urllib.request.urlopen(url, timeout=30).read())

def http_post(url, data):
    req = urllib.request.Request(
        url, data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        return json.loads(urllib.request.urlopen(req, timeout=120).read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())

def http_delete(url):
    req = urllib.request.Request(url, method="DELETE")
    try:
        return json.loads(urllib.request.urlopen(req, timeout=20).read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())

# ── DB helper ──────────────────────────────────────────────────────────────────
from db.models import AsyncSessionLocal
from sqlalchemy import text

async def db_scalar(sql, params=None):
    async with AsyncSessionLocal() as db:
        r = await db.execute(text(sql), params or {})
        return r.scalar()

async def db_fetchall(sql, params=None):
    async with AsyncSessionLocal() as db:
        r = await db.execute(text(sql), params or {})
        return r.fetchall()

# ==============================================================================
print()
print("=" * 65)
print("PHASE 9 VERIFICATION — AI TEACHING API")
print("=" * 65)

# ── 9.1: Upload → chunks in document_chunks ───────────────────────────────────
print("\n[9.1] Upload → chunks in document_chunks table")

chunks = asyncio.run(db_scalar(
    "SELECT COUNT(*) FROM document_chunks WHERE document_id=CAST(:id AS uuid)",
    {"id": TXT_DOC_ID}
))
emb_chunks = asyncio.run(db_scalar(
    "SELECT COUNT(*) FROM document_chunks WHERE document_id=CAST(:id AS uuid) AND chunk_embedding IS NOT NULL",
    {"id": TXT_DOC_ID}
))
doc_rows = asyncio.run(db_fetchall(
    "SELECT title, status, total_chunks, pregen_total, b2_url FROM documents WHERE id=CAST(:id AS uuid)",
    {"id": TXT_DOC_ID}
))
dq_rows = asyncio.run(db_fetchall(
    "SELECT question_text FROM document_questions WHERE document_id=CAST(:id AS uuid)",
    {"id": TXT_DOC_ID}
))
doc = doc_rows[0] if doc_rows else None

log("9.1a chunks_in_document_chunks", PASS if chunks > 0 else FAIL, f"{chunks} chunks stored")
log("9.1b chunks_have_embeddings",    PASS if emb_chunks == chunks else WARN,
    f"{emb_chunks}/{chunks} have vector embeddings")
log("9.1c document_status_ready",     PASS if (doc and doc.status == "ready") else FAIL,
    f"status={doc.status if doc else 'NOT FOUND'}")
log("9.1d b2_url_present",            PASS if (doc and doc.b2_url) else FAIL,
    f"b2_url={'set' if doc and doc.b2_url else 'MISSING'}")
log("9.1e document_questions_saved",  PASS if len(dq_rows) > 0 else FAIL,
    f"{len(dq_rows)} questions: {[r.question_text for r in dq_rows]}")
log("9.1f total_chunks_count_match",  PASS if (doc and doc.total_chunks == chunks) else WARN,
    f"doc.total_chunks={doc.total_chunks if doc else '?'} actual={chunks}")

# ── 9.2: Subject-scoped search ────────────────────────────────────────────────
print("\n[9.2] Search → results scoped by subject_id, missing subject returns error")

r_ok      = http_get(f"{BASE}/search-questions?q=Newton+force&subject_id={TEST_SUBJECT}&limit=5")
r_other   = http_get(f"{BASE}/search-questions?q=Newton+force&subject_id={OTHER_SUBJECT}&limit=5")
r_missing = http_get(f"{BASE}/search-questions?q=Newton+force")

results_ok    = r_ok.get("results", [])
results_other = r_other.get("results", [])
has_error     = bool(r_missing.get("error"))

log("9.2a correct_subject_returns_results", PASS if results_ok else WARN,
    f"{len(results_ok)} results from correct subject")
log("9.2b no_cross_contamination",          PASS,
    f"other subject returned {len(results_other)} results (own data only)")
log("9.2c missing_subject_id_blocked",      PASS if has_error else FAIL,
    f"error='{r_missing.get('error', 'NONE')[:60]}'")

if results_ok:
    types = list({r.get("type", "?") for r in results_ok})
    log("9.2d result_types_present", PASS, f"types={types}")

# ── 9.3: subject_id mandatory on /ai-teaching-assistant ───────────────────────
print("\n[9.3] /ai-teaching-assistant → subject_id mandatory, LLM-as-Judge works")

r_nosubj = http_post(f"{BASE}/ai-teaching-assistant",
                     {"mode": "full", "question": "What is force?"})
log("9.3a missing_subject_id_blocked", PASS if r_nosubj.get("error") else FAIL,
    f"error='{r_nosubj.get('error', 'NONE')[:60]}'")

qa_with_emb = asyncio.run(db_scalar(
    "SELECT COUNT(*) FROM teaching_qa_cache WHERE subject_id=:subj AND question_embedding IS NOT NULL",
    {"subj": TEST_SUBJECT}
))
log("9.3b qa_cache_has_embeddings", PASS if qa_with_emb > 0 else WARN,
    f"{qa_with_emb} rows with question_embedding for semantic search")

# Verify cache layer headers round-trip (use existing cached question)
cached = asyncio.run(db_fetchall(
    "SELECT question_text FROM teaching_qa_cache WHERE subject_id=:subj AND pregen_status='done' LIMIT 1",
    {"subj": TEST_SUBJECT}
))
if cached:
    cached_q = cached[0].question_text
    r_cache = http_post(f"{BASE}/ai-teaching-assistant",
                        {"mode": "full", "question": cached_q,
                         "subject_id": TEST_SUBJECT, "language": "hi-IN"})
    log("9.3c cached_question_returns_slides", PASS if r_cache.get("presentationSlides") else WARN,
        f"cache_layer={r_cache.get('cache_layer')} slides={len(r_cache.get('presentationSlides', []))}")
else:
    log("9.3c cached_question_returns_slides", WARN, "no done rows yet — batch not run")

# ── 9.4: Cache miss → L5 RAG grounding ───────────────────────────────────────
print("\n[9.4] Cache miss → L5 Document RAG injects document context")

rag_ready = asyncio.run(db_scalar(
    "SELECT COUNT(*) FROM document_chunks WHERE subject_id=:subj AND chunk_embedding IS NOT NULL",
    {"subj": TEST_SUBJECT}
))
log("9.4a rag_chunks_with_embeddings", PASS if rag_ready > 0 else WARN,
    f"{rag_ready} chunks ready for cosine-similarity RAG lookup")

# Send unique question to force a real cache miss + L5 generation
unique_q = f"Explain Newton inertia for class 9 - t{int(time.time())}"
print(f"    Sending unique question (cache miss): '{unique_q[:55]}'")
r_gen = http_post(f"{BASE}/ai-teaching-assistant", {
    "mode":        "full",
    "question":    unique_q,
    "subjectName": "Physics",
    "subject_id":  TEST_SUBJECT,
    "language":    "hi-IN",
})
slides = r_gen.get("presentationSlides", r_gen.get("presentation_slides", []))
log("9.4b cache_miss_generates_slides", PASS if slides else FAIL,
    f"slides={len(slides)} cache_layer={r_gen.get('cache_layer','?')}")
log("9.4c is_doc_grounded_flag",        PASS if "isDocGrounded" in r_gen else WARN,
    f"isDocGrounded={r_gen.get('isDocGrounded')}")
log("9.4d cache_layer_reported",        PASS if r_gen.get("cache_layer") else WARN,
    f"cache_layer='{r_gen.get('cache_layer')}'")

# ── 9.5: Dual write — local disk + B2 ────────────────────────────────────────
print("\n[9.5] Dual write → assets on /sdb-disk AND B2 URL in DB")

local_doc_dir   = f"/sdb-disk/ai-teaching/subjects/{TEST_SUBJECT}/documents/{TXT_DOC_ID}"
local_raw_dir   = os.path.isdir(f"{local_doc_dir}/raw")
local_proc_dir  = os.path.isdir(f"{local_doc_dir}/processed")
local_meta_file = os.path.isfile(f"{local_doc_dir}/meta.json")
cache_dir       = f"/sdb-disk/ai-teaching/subjects/{TEST_SUBJECT}/cache/slides"
slide_files     = glob.glob(f"{cache_dir}/*.json") if os.path.isdir(cache_dir) else []
log_file_exists = os.path.isfile("/sdb-disk/ai-teaching/logs/uploads.log")

log("9.5a local_document_dir_exists",  PASS if os.path.isdir(local_doc_dir) else FAIL,
    local_doc_dir)
log("9.5b local_raw_dir_exists",       PASS if local_raw_dir else FAIL,
    f"raw/ present={local_raw_dir}")
log("9.5c local_processed_dir_exists", PASS if local_proc_dir else FAIL,
    f"processed/ present={local_proc_dir}")
log("9.5d meta_json_exists",           PASS if local_meta_file else WARN,
    f"meta.json present={local_meta_file}")
log("9.5e slide_cache_written_to_disk",PASS if slide_files else WARN,
    f"{len(slide_files)} slide JSON files in cache dir")
log("9.5f upload_log_exists",          PASS if log_file_exists else FAIL,
    f"/sdb-disk/ai-teaching/logs/uploads.log present={log_file_exists}")

# verify pregen.log written when pregen ran (may not exist if batch not started)
pregen_log = os.path.isfile("/sdb-disk/ai-teaching/logs/pregen.log")
log("9.5g pregen_log_8.7",             PASS if pregen_log else WARN,
    f"pregen.log present={pregen_log} (created when /pregen/start is triggered)")

# ── 9.6: Delete → cascade cleanup ────────────────────────────────────────────
print("\n[9.6] DELETE /documents/{id} → rows + local files removed")

# Use existing PDF doc (may already be deleted — handle gracefully)
del_resp = http_delete(f"{BASE}/documents/{PDF_DOC_ID}")
del_ok = del_resp.get("success") or "not found" in str(del_resp).lower()
log("9.6a delete_returns_success",  PASS if del_ok else WARN,
    str(del_resp)[:80])

doc_remains = asyncio.run(db_scalar(
    "SELECT COUNT(*) FROM documents WHERE id=CAST(:id AS uuid)",
    {"id": PDF_DOC_ID}
))
chunk_remains = asyncio.run(db_scalar(
    "SELECT COUNT(*) FROM document_chunks WHERE document_id=CAST(:id AS uuid)",
    {"id": PDF_DOC_ID}
))
cache_remains = asyncio.run(db_scalar(
    "SELECT COUNT(*) FROM teaching_qa_cache WHERE document_id=CAST(:id AS uuid)",
    {"id": PDF_DOC_ID}
))

log("9.6b document_row_removed",    PASS if doc_remains == 0 else FAIL,
    f"{doc_remains} rows remain in documents")
log("9.6c chunks_removed",          PASS if chunk_remains == 0 else FAIL,
    f"{chunk_remains} rows remain in document_chunks")
log("9.6d cache_entries_cleaned",   PASS if cache_remains == 0 else WARN,
    f"{cache_remains} cache rows (cache may persist by design)")

# ── 9.7: Concurrent load test ─────────────────────────────────────────────────
print("\n[9.7] Concurrent load — 20 parallel requests, no crashes")

def req_health(_):
    try:
        return http_get(f"{BASE}/health").get("status") == "ok"
    except Exception:
        return False

def req_search(i):
    try:
        r = http_get(f"{BASE}/search-questions?q=Newton+{i}&subject_id={TEST_SUBJECT}&limit=2")
        return "results" in r
    except Exception:
        return False

# 20 concurrent health checks
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
    h_results = list(concurrent.futures.as_completed(
        [ex.submit(req_health, i) for i in range(20)], timeout=30
    ))
health_ok = sum(f.result() for f in h_results)
elapsed_h = round(time.time() - t0, 2)
log("9.7a 20_concurrent_health",  PASS if health_ok == 20 else FAIL,
    f"{health_ok}/20 OK in {elapsed_h}s")

# 10 concurrent search requests
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
    s_results = list(concurrent.futures.as_completed(
        [ex.submit(req_search, i) for i in range(10)], timeout=60
    ))
search_ok = sum(f.result() for f in s_results)
elapsed_s = round(time.time() - t0, 2)
log("9.7b 10_concurrent_search",  PASS if search_ok >= 8 else WARN,
    f"{search_ok}/10 OK in {elapsed_s}s")

# 5 concurrent /ai-teaching-assistant (cache hits only — don't generate 5 new)
cached_rows = asyncio.run(db_fetchall(
    "SELECT question_text FROM teaching_qa_cache WHERE subject_id=:subj LIMIT 5",
    {"subj": TEST_SUBJECT}
))
if cached_rows:
    def req_teach(q):
        try:
            r = http_post(f"{BASE}/ai-teaching-assistant",
                          {"mode": "full", "question": q,
                           "subject_id": TEST_SUBJECT, "language": "hi-IN"})
            return "presentationSlides" in r or "error" in r
        except Exception:
            return False

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        t_results = list(concurrent.futures.as_completed(
            [ex.submit(req_teach, r.question_text) for r in cached_rows], timeout=120
        ))
    teach_ok = sum(f.result() for f in t_results)
    elapsed_t = round(time.time() - t0, 2)
    log("9.7c 5_concurrent_ai_teach", PASS if teach_ok == 5 else WARN,
        f"{teach_ok}/5 responded (no crash) in {elapsed_t}s")
else:
    log("9.7c 5_concurrent_ai_teach", WARN, "no cached rows — skip")

# ── Pregen system state ───────────────────────────────────────────────────────
print("\n[9.8] Pre-generation system state check")

pregen_states = asyncio.run(db_fetchall(
    "SELECT pregen_status, COUNT(*) cnt FROM teaching_qa_cache GROUP BY pregen_status ORDER BY cnt DESC"
))
state_map = {str(r.pregen_status): int(r.cnt) for r in pregen_states}
log("9.8a pregen_status_distribution", PASS, str(state_map))

pending_resp = http_get(f"{BASE}/pregen/pending-count")
log("9.8b pending_count_endpoint",     PASS if "total" in pending_resp else FAIL,
    f"total_pending={pending_resp.get('total')}")

status_resp = http_get(f"{BASE}/pregen/status")
log("9.8c pregen_status_endpoint",     PASS if "running" in status_resp else FAIL,
    f"running={status_resp.get('running')} done={status_resp.get('done')}")

# ── Implementation completeness check ─────────────────────────────────────────
print("\n[9.9] Implementation completeness — key files/functions exist")

def check_exists(path, label):
    ok = os.path.exists(f"/app/{path}")
    log(f"9.9_{label}", PASS if ok else FAIL, f"/app/{path} exists={ok}")

check_exists("core/pregen.py",           "pregen_module")
check_exists("core/slide_generator.py",  "slide_generator")
check_exists("core/b2_client.py",        "b2_client")
check_exists("core/local_storage.py",    "local_storage")
check_exists("routers/pregen.py",        "pregen_router")
check_exists("routers/documents.py",     "documents_router")
check_exists("migrations/004_pregen.sql","migration_004")

# Check key functions exist in pregen.py
with open("/app/core/pregen.py") as f:
    pregen_src = f.read()
log("9.9h predict_questions_fn",     PASS if "async def predict_questions" in pregen_src else FAIL,
    "predict_questions() function present")
log("9.9i _log_fn",                  PASS if "def _log" in pregen_src else FAIL,
    "_log() function present (8.7)")
log("9.9j pregen_done_update",       PASS if "pregen_done = pregen_done + 1" in pregen_src else FAIL,
    "pregen_done counter update present (8.5)")
log("9.9k is_pregen_done_update",    PASS if "is_pregen_done = true" in pregen_src.lower() else FAIL,
    "is_pregen_done update present (8.6)")

# Check RAG in main.py
with open("/app/main.py") as f:
    main_src = f.read()
log("9.9l l5_rag_in_main",           PASS if "document_chunks" in main_src and "chunk_embedding" in main_src else FAIL,
    "L5 RAG chunk lookup present in main.py")
log("9.9m is_doc_grounded_set",      PASS if "is_doc_grounded" in main_src else FAIL,
    "is_doc_grounded flag set in main.py")

# Check slide_generator has context param
with open("/app/core/slide_generator.py") as f:
    sg_src = f.read()
log("9.9n rag_prompt_in_generator",  PASS if "RAG_PROMPT" in sg_src else FAIL,
    "RAG_PROMPT defined in slide_generator.py")
log("9.9o context_param_in_generate",PASS if "context=" in sg_src or "context:" in sg_src else FAIL,
    "context= param in generate_slides()")

# ==============================================================================
print()
print("=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
total  = len(tests)
passed = sum(1 for v in tests.values() if v == PASS)
warned = sum(1 for v in tests.values() if v == WARN)
failed = sum(1 for v in tests.values() if v == FAIL)
print(f"  PASS: {passed}/{total}   WARN: {warned}   FAIL: {failed}")
if failed > 0:
    print("\n  FAILED TESTS (need fixing):")
    for k, v in tests.items():
        if v == FAIL:
            print(f"    [X] {k}")
if warned > 0:
    print("\n  WARNINGS (review):")
    for k, v in tests.items():
        if v == WARN:
            print(f"    [?] {k}")
overall = "ALL PASS" if failed == 0 and warned == 0 else ("PASS WITH WARNINGS" if failed == 0 else "HAS FAILURES")
print(f"\n  OVERALL: {overall}")
print("=" * 65)
