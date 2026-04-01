from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import settings
from app.ingestion.csv_loader import load_all_csvs
from app.ingestion.docx_loader import load_all_docx
from app.ingestion.pdf_loader import load_all_pdfs
from app.ingestion.embeddings import get_embeddings
from app.observability import logger
from app.vector_store.qdrant_store import add_documents, collection_stats, ensure_collection, get_qdrant_client


def run_ingestion(data_dir: str = None, batch_size: int = 64) -> dict:
    data_dir = data_dir or settings.DATA_DIR
    logger.info("ingestion_start", data_dir=data_dir)

    embeddings = get_embeddings()
    ensure_collection(get_qdrant_client())

    pdf_docs   = load_all_pdfs(data_dir)
    docx_docs  = load_all_docx(data_dir)
    csv_docs   = load_all_csvs(data_dir)

    all_docs: List[Document] = pdf_docs + docx_docs + csv_docs
    logger.info("total_docs", count=len(all_docs))

    for i in range(0, len(all_docs), batch_size):
        add_documents(all_docs[i: i + batch_size], embeddings)

    stats = collection_stats()
    summary = {
        "pdf_chunks":   len(pdf_docs),
        "docx_chunks":  len(docx_docs),
        "csv_rows":     len(csv_docs),
        "total":        len(all_docs),
        "qdrant_points": stats.get("points_count"),
    }
    logger.info("ingestion_done", **summary)
    return summary
