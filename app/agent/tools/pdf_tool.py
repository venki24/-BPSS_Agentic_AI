from langchain_core.tools import tool
from app.ingestion.embeddings import get_embeddings
from app.vector_store.qdrant_store import similarity_search
from app.observability import logger

def _format(docs):
    if not docs:
        return "Nothing found in PDF sources."
    out = []
    for d in docs:
        src = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page", "?")
        cid = d.metadata.get("candidate_id", "")
        header = f"Source: {src} | Page: {page}" + (f" | {cid}" if cid else "")
        out.append(f"{header}\n{d.page_content.strip()}")
    return "\n\n---\n\n".join(out)


@tool
def search_pdfs(query: str) -> str:
    """
    Search PDF documents: BPSS screening policy, operations SOP,
    adjudication register, and permitted RTW evidence matrix.
    Use this for policy rules, compliance requirements, adjudication
    decisions, and right-to-work document standards.
    """
    logger.info("tool_search_pdfs", query=query[:150])
    return _format(similarity_search(query, get_embeddings(), source_type="pdf"))
