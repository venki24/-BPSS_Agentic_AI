"""
POST /api/query  – run the agent against a user question
GET  /api/status – collection health check
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.agent import run_query
from app.observability import logger
from app.vector_store.qdrant_store import collection_stats

router = APIRouter(prefix="/api", tags=["query"])


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural-language question")
    chat_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Optional prior turns: [{role: user|assistant, content: ...}]",
    )


class TraceStep(BaseModel):
    type: str
    tool: Optional[str] = None
    input: Optional[str] = None
    output_preview: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    trace: Dict[str, Any]


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """
    Run the BPSS compliance agent.
    The agent decides which tools to call (PDF / DOCX / CSV) based on the question,
    retrieves evidence, and synthesises a grounded answer with source citations.
    """
    logger.info("api_query", question=req.question[:200])

    # Convert chat_history dicts to LangChain message objects if provided
    history = []
    if req.chat_history:
        from langchain_core.messages import HumanMessage, AIMessage
        for turn in req.chat_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                history.append(HumanMessage(content=content))
            else:
                history.append(AIMessage(content=content))

    try:
        result = run_query(question=req.question, chat_history=history)
    except Exception as exc:
        logger.error("api_query_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(**result)


@router.get("/status")
async def status_endpoint():
    """Health check + Qdrant collection stats."""
    try:
        stats = collection_stats()
        return {"status": "ok", "qdrant": stats}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
