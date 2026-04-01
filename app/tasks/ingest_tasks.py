"""
Celery tasks for background document ingestion.

Usage (fire-and-forget):
    from app.tasks.ingest_tasks import ingest_all_task
    task = ingest_all_task.delay()
    print(task.id)          # poll status
    print(task.get())       # wait for result
"""
from app.tasks.celery_app import celery_app
from app.observability import logger


@celery_app.task(bind=True, name="ingest.all")
def ingest_all_task(self, data_dir: str = None) -> dict:
    """Ingest all documents from DATA_DIR into Qdrant."""
    self.update_state(state="STARTED", meta={"status": "loading documents"})
    logger.info("celery_ingest_start", task_id=self.request.id)

    from app.ingestion.pipeline import run_ingestion

    summary = run_ingestion(data_dir=data_dir)
    logger.info("celery_ingest_done", **summary)
    return summary


@celery_app.task(bind=True, name="ingest.file")
def ingest_file_task(self, file_path: str) -> dict:
    """Ingest a single file (pdf/docx/csv/xlsx) into Qdrant."""
    from pathlib import Path
    from app.ingestion.pipeline import get_embeddings
    from app.vector_store.qdrant_store import add_documents

    path = Path(file_path)
    ext = path.suffix.lower()
    logger.info("celery_ingest_file", file=path.name)

    if ext == ".pdf":
        from app.ingestion.pdf_loader import load_pdf
        docs = load_pdf(path)
    elif ext == ".docx":
        from app.ingestion.docx_loader import load_docx
        docs = load_docx(path)
    elif ext in (".csv", ".xlsx"):
        from app.ingestion.csv_loader import load_csv_to_docs
        docs = load_csv_to_docs(path)
    else:
        return {"error": f"Unsupported file type: {ext}"}

    embeddings = get_embeddings()
    add_documents(docs, embeddings)
    return {"file": path.name, "chunks": len(docs)}
