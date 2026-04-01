"""
CSV / XLSX ingestion.

Two modes:
  1. Vector mode  – each row → text → Document stored in Qdrant for semantic search
  2. DataFrame mode – in-memory pandas DataFrames for structured queries
"""
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from langchain_core.documents import Document

from app.config import settings
from app.observability import logger

_CAND_RE = re.compile(r"CAND-\d{3}", re.IGNORECASE)

# Global in-memory store  { stem: DataFrame }
_DATAFRAMES: Dict[str, pd.DataFrame] = {}


def _row_to_text(row: pd.Series, filename: str) -> str:
    pairs = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
    return f"[{filename}] " + " | ".join(pairs)


def load_csv_to_docs(file_path: str | Path) -> List[Document]:
    path = Path(file_path)
    logger.info("loading_csv", file=path.name)

    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)

    df.fillna("", inplace=True)
    _DATAFRAMES[path.stem] = df  # cache for structured queries

    docs: List[Document] = []
    for idx, row in df.iterrows():
        text = _row_to_text(row, path.name)
        cid = _CAND_RE.search(text)
        doc = Document(
            page_content=text,
            metadata={
                "source_type": "csv",
                "source_file": path.name,
                "candidate_id": cid.group(0).upper() if cid else "",
                "row_index": int(idx),
            },
        )
        docs.append(doc)

    logger.info("csv_rows_as_docs", file=path.name, rows=len(docs))
    return docs


def load_all_csvs(data_dir: str = None) -> List[Document]:
    data_dir = Path(data_dir or settings.DATA_DIR)
    docs: List[Document] = []
    for f in data_dir.rglob("*.csv"):
        docs.extend(load_csv_to_docs(f))
    for f in data_dir.rglob("*.xlsx"):
        docs.extend(load_csv_to_docs(f))
    logger.info("all_csvs_loaded", total_rows=len(docs))
    return docs


# ── Structured query helpers ─────────────────────────────────────────────────

def get_all_dataframes() -> Dict[str, pd.DataFrame]:
    """Return the in-memory cache; will be populated after load_all_csvs()."""
    return _DATAFRAMES


def query_dataframes_by_candidate(candidate_id: str) -> str:
    """Return all rows matching a candidate_id across all loaded DataFrames."""
    results = []
    for name, df in _DATAFRAMES.items():
        matches = df[df.apply(lambda r: candidate_id.upper() in " ".join(r.astype(str)).upper(), axis=1)]
        if not matches.empty:
            results.append(f"=== {name} ===\n{matches.to_string(index=False)}")
    return "\n\n".join(results) if results else f"No data found for {candidate_id}"


def format_dataframe_summary() -> str:
    """Return column names + row count for each loaded DataFrame."""
    lines = []
    for name, df in _DATAFRAMES.items():
        lines.append(f"File: {name}.csv  Rows: {len(df)}  Columns: {list(df.columns)}")
    return "\n".join(lines)
