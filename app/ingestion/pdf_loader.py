"""
PDF ingestion: load → chunk → tag metadata → return Documents.
"""
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.observability import logger

_CAND_RE = re.compile(r"CAND-\d{3}", re.IGNORECASE)


def _extract_candidate_id(text: str) -> str:
    m = _CAND_RE.search(text)
    return m.group(0).upper() if m else ""


def load_pdf(file_path: str | Path) -> List[Document]:
    path = Path(file_path)
    logger.info("loading_pdf", file=path.name)

    loader = PyPDFLoader(str(path))
    raw_pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(raw_pages)

    for i, chunk in enumerate(chunks):
        cid = _extract_candidate_id(chunk.page_content)
        chunk.metadata.update(
            {
                "source_type": "pdf",
                "source_file": path.name,
                "candidate_id": cid,
                "chunk_index": i,
            }
        )

    logger.info("pdf_chunked", file=path.name, chunks=len(chunks))
    return chunks


def load_all_pdfs(data_dir: str = None) -> List[Document]:
    data_dir = Path(data_dir or settings.DATA_DIR)
    docs: List[Document] = []
    for pdf in data_dir.rglob("*.pdf"):
        docs.extend(load_pdf(pdf))
    logger.info("all_pdfs_loaded", total_chunks=len(docs))
    return docs
