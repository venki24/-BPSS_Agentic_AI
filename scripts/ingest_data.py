#!/usr/bin/env python
"""
One-shot ingestion script — run this ONCE before starting the API.

Usage:
    python scripts/ingest_data.py
    python scripts/ingest_data.py --data-dir /custom/path
    python scripts/ingest_data.py --reset   # drop + recreate collection first
"""
import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.observability import setup_logging, logger
from app.vector_store.qdrant_store import get_qdrant_client


def reset_collection():
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.QDRANT_COLLECTION in existing:
        client.delete_collection(settings.QDRANT_COLLECTION)
        logger.info("collection_deleted", name=settings.QDRANT_COLLECTION)


def main():
    parser = argparse.ArgumentParser(description="Ingest BPSS dataset into Qdrant")
    parser.add_argument("--data-dir", default=settings.DATA_DIR)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the Qdrant collection before ingesting",
    )
    args = parser.parse_args()

    setup_logging(settings.LOG_LEVEL)
    logger.info("ingest_script_start", data_dir=args.data_dir)

    if args.reset:
        logger.info("resetting_collection")
        reset_collection()

    from app.ingestion.pipeline import run_ingestion

    summary = run_ingestion(data_dir=args.data_dir)

    print("\n" + "=" * 50)
    print("Ingestion complete!")
    print(f"  PDF chunks  : {summary['pdf_chunks']}")
    print(f"  DOCX chunks : {summary['docx_chunks']}")
    print(f"  CSV rows    : {summary['csv_rows']}")
    print(f"  Total docs  : {summary['total']}")
    print(f"  Qdrant pts  : {summary['qdrant_points']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
