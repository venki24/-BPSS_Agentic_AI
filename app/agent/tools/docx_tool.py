from langchain_core.tools import tool
from app.ingestion.embeddings import get_embeddings
from app.vector_store.qdrant_store import similarity_search
from app.observability import logger

def _format(docs):
    if not docs:
        return "Nothing found in DOCX sources."
    out = []
    for d in docs:
        src = d.metadata.get("source_file", "unknown")
        cid = d.metadata.get("candidate_id", "")
        header = f"Source: {src}" + (f" | {cid}" if cid else "")
        out.append(f"{header}\n{d.page_content.strip()}")
    return "\n\n---\n\n".join(out)


@tool
def extract_docx(query: str) -> str:
    """
    Search DOCX files: candidate packs (CAND-101 to CAND-106),
    analyst working notes, and email approvals/escalations.
    Use this for candidate-specific details, analyst decisions,
    approval emails, and exception narratives.
    """
    logger.info("tool_extract_docx", query=query[:150])
    return _format(similarity_search(query, get_embeddings(), source_type="docx"))
