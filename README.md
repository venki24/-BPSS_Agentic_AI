# BPSS Compliance Agent

Backend-only agentic AI system that answers questions over a mixed-format BPSS
screening dataset — PDFs, Word docs, and spreadsheets.

Built with FastAPI, LangGraph, Qdrant, and Azure OpenAI.

---

## Setup

```bash
# 1. clone / navigate to project
cd compliance_task

# 2. create virtual env
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# 3. install dependencies
pip install -r requirements.txt
```

Qdrant and Redis need to be running. If you have Docker:
```bash
docker-compose up -d
```
If they're already running from another project, skip this step.

---

## One-time data ingestion

Run this once before starting the API. It loads all PDFs, DOCX, and CSVs into Qdrant.

```bash
python scripts/ingest_data.py
```

To reset and re-ingest from scratch:
```bash
python scripts/ingest_data.py --reset
```

---

## Start the API

```bash
uvicorn app.main:app --port 8001
```

Swagger UI: http://localhost:8001/docs


## API endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/query` | Ask the agent a question |
| GET | `/api/status` | Health check + Qdrant stats |
| POST | `/api/ingest/sync` | Re-ingest all documents (blocking) |
| POST | `/api/ingest` | Re-ingest async via Celery |
| GET | `/api/ingest/task/{id}` | Check async task status |

---

## Configuration

All config lives in `.env`. Key settings:

```env
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=your-deployment-name

# Embeddings: "huggingface" (default, free) or "azure"
EMBEDDING_PROVIDER=huggingface

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

---

