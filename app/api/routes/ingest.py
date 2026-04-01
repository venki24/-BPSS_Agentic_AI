"""
POST /api/ingest        – trigger full ingestion (async via Celery)
POST /api/ingest/sync   – trigger full ingestion synchronously (for dev/testing)
POST /api/ingest/file   – upload + ingest a single file
GET  /api/ingest/task/{task_id} – poll Celery task status
"""
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.ingestion.pipeline import run_ingestion
from app.observability import logger

router = APIRouter(prefix="/api/ingest", tags=["ingestion"])


class IngestResponse(BaseModel):
    task_id: str = None
    message: str
    summary: dict = None


# ── Async ingestion via Celery ────────────────────────────────────────────────

@router.post("", response_model=IngestResponse)
async def ingest_async():
    """
    Trigger full document ingestion as a background Celery task.
    Returns a task_id you can poll at GET /api/ingest/task/{task_id}.
    """
    try:
        from app.tasks.ingest_tasks import ingest_all_task
        task = ingest_all_task.delay()
        logger.info("ingest_task_queued", task_id=task.id)
        return IngestResponse(task_id=task.id, message="Ingestion queued")
    except Exception as exc:
        logger.warning("celery_unavailable_falling_back", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Celery unavailable: {exc}")


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Poll the status of an async ingestion task."""
    try:
        from app.tasks.celery_app import celery_app
        result = celery_app.AsyncResult(task_id)
        return {
            "task_id": task_id,
            "state": result.state,
            "result": result.result if result.ready() else None,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Synchronous ingestion (dev / testing) ─────────────────────────────────────

@router.post("/sync", response_model=IngestResponse)
async def ingest_sync():
    """
    Run full ingestion synchronously (blocks until done).
    Use only for dev/testing with small datasets.
    """
    logger.info("ingest_sync_start")
    try:
        summary = run_ingestion()
        return IngestResponse(message="Ingestion complete", summary=summary)
    except Exception as exc:
        logger.error("ingest_sync_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# ── Single-file upload + ingest ───────────────────────────────────────────────

@router.post("/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Upload a single PDF, DOCX, CSV, or XLSX and ingest it immediately."""
    allowed = {".pdf", ".docx", ".csv", ".xlsx"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        from app.tasks.ingest_tasks import ingest_file_task
        task = ingest_file_task.delay(tmp_path)
        return IngestResponse(task_id=task.id, message=f"File {file.filename} queued for ingestion")
    except Exception as exc:
        # Fallback: ingest synchronously
        from app.ingestion.pipeline import get_embeddings
        from app.vector_store.qdrant_store import add_documents
        from pathlib import Path as P

        p = P(tmp_path)
        if suffix == ".pdf":
            from app.ingestion.pdf_loader import load_pdf
            docs = load_pdf(p)
        elif suffix == ".docx":
            from app.ingestion.docx_loader import load_docx
            docs = load_docx(p)
        else:
            from app.ingestion.csv_loader import load_csv_to_docs
            docs = load_csv_to_docs(p)

        add_documents(docs, get_embeddings())
        return IngestResponse(message=f"File {file.filename} ingested ({len(docs)} chunks)")
