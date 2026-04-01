"""
BPSS Compliance Agent — FastAPI Application Entry Point

Endpoints:
  POST /api/query          → run agent on a question
  GET  /api/status         → health check + Qdrant stats
  POST /api/ingest/sync    → ingest all data synchronously
  POST /api/ingest         → ingest asynchronously (Celery)
  POST /api/ingest/file    → upload + ingest a single file
  GET  /api/ingest/task/{id} → poll Celery task status
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.config import settings
from app.observability import setup_logging, logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup
    setup_logging(settings.LOG_LEVEL)
    logger.info(
        "app_startup",
        name=settings.APP_NAME,
        qdrant=f"{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
        embedding_provider=settings.EMBEDDING_PROVIDER,
    )

    # Pre-warm embeddings model (avoids cold-start on first query)
    try:
        from app.ingestion.pipeline import get_embeddings
        get_embeddings()
        logger.info("embeddings_warmed")
    except Exception as exc:
        logger.warning("embeddings_warm_failed", error=str(exc))

    yield  # app is running

    # ── Shutdown 
    logger.info("app_shutdown")


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Agentic AI system for BPSS compliance analysis. "
        "Accepts natural-language questions and retrieves evidence from "
        "PDFs, DOCX, and CSV/XLSX data sources."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/", tags=["root"])
async def root():
    return {
        "service": settings.APP_NAME,
        "docs": "/docs",
        "status": "/api/status",
        "query": "POST /api/query",
    }
