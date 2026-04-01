"""
Celery application for background ingestion tasks.
Broker + result backend: Redis (configured in .env).
"""
from celery import Celery
from app.config import settings

celery_app = Celery(
    "bpss_agent",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.ingest_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_expires=3600,
)
