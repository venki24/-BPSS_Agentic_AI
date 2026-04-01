"""
Qdrant vector store wrapper.
Single collection `bpss_documents` with metadata:
  source_type : "pdf" | "docx" | "csv"
  source_file : basename of the originating file
  candidate_id: CAND-XXX if extractable, else null
"""
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from app.config import settings
from app.observability import logger

_client: Optional[QdrantClient] = None
_vector_store: Optional[QdrantVectorStore] = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    return _client


def ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if settings.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=settings.QDRANT_VECTOR_SIZE, distance=Distance.COSINE
            ),
        )
        logger.info("qdrant_collection_created", name=settings.QDRANT_COLLECTION)
    else:
        logger.info("qdrant_collection_ready", name=settings.QDRANT_COLLECTION)


def get_vector_store(embeddings: Embeddings) -> QdrantVectorStore:
    global _vector_store
    if _vector_store is None:
        client = get_qdrant_client()
        ensure_collection(client)
        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.QDRANT_COLLECTION,
            embedding=embeddings,
        )
    return _vector_store


def add_documents(docs: List[Document], embeddings: Embeddings) -> None:
    vs = get_vector_store(embeddings)
    vs.add_documents(docs)
    logger.info("docs_added_to_qdrant", count=len(docs))


def similarity_search(
    query: str,
    embeddings: Embeddings,
    k: int = None,
    source_type: Optional[str] = None,
) -> List[Document]:
    """Semantic search, optionally filtered by source_type."""
    k = k or settings.TOP_K_RETRIEVAL
    vs = get_vector_store(embeddings)
    search_kwargs: dict = {"k": k}

    if source_type:
        search_kwargs["filter"] = Filter(
            must=[
                FieldCondition(
                    key="metadata.source_type",
                    match=MatchValue(value=source_type),
                )
            ]
        )

    retriever = vs.as_retriever(search_kwargs=search_kwargs)
    return retriever.invoke(query)


def collection_stats() -> dict:
    client = get_qdrant_client()
    info = client.get_collection(settings.QDRANT_COLLECTION)
    return {
        "points_count": info.points_count,
        "collection": settings.QDRANT_COLLECTION,
        "status": str(info.status),
    }
