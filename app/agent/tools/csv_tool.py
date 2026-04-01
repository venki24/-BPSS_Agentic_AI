import re
from langchain_core.tools import tool
from app.ingestion.csv_loader import (
    get_all_dataframes,
    query_dataframes_by_candidate,
    format_dataframe_summary,
)
from app.ingestion.embeddings import get_embeddings
from app.vector_store.qdrant_store import similarity_search
from app.observability import logger

CAND_RE = re.compile(r"CAND-\d{3}", re.IGNORECASE)


def _pandas_query(query: str) -> str:
    """
    Tries to answer structured questions directly via pandas.
    Handles: candidate lookups, risk filters, incomplete check detection.
    Falls back gracefully if no match.
    """
    dfs = get_all_dataframes()
    if not dfs:
        return ""

    q = query.lower()
    results = []

    # direct candidate lookup e.g. "CAND-104"
    m = CAND_RE.search(query)
    if m:
        return query_dataframes_by_candidate(m.group(0))

    # risk level filter
    for level in ["high", "medium", "low"]:
        if level in q and "risk" in q:
            for name, df in dfs.items():
                if "risk_level" in df.columns:
                    rows = df[df["risk_level"].str.lower() == level]
                    if not rows.empty:
                        results.append(f"=== {name} (risk={level}) ===\n{rows.to_string(index=False)}")

    # incomplete checks
    if any(w in q for w in ["incomplete", "missing", "not complete", "failed"]):
        for name, df in dfs.items():
            flag_cols = [c for c in df.columns if "complete" in c.lower() or "ready" in c.lower()]
            if flag_cols:
                mask = df[flag_cols].apply(lambda col: col.str.lower().isin(["no", "false", "n/a", ""])).any(axis=1)
                rows = df[mask]
                if not rows.empty:
                    results.append(f"=== {name} (incomplete flags) ===\n{rows.to_string(index=False)}")

    # ready to join but checks not done
    if "ready" in q and ("incomplete" in q or "not" in q):
        for name, df in dfs.items():
            if "ready_to_join" in df.columns:
                check_cols = [c for c in df.columns if "complete" in c.lower()]
                if check_cols:
                    mask = (df["ready_to_join"].str.lower() == "yes") & (
                        df[check_cols].apply(lambda col: col.str.lower() == "no").any(axis=1)
                    )
                    rows = df[mask]
                    if not rows.empty:
                        results.append(f"=== {name} (ready but incomplete) ===\n{rows.to_string(index=False)}")

    return "\n\n".join(results)


def _format_docs(docs) -> str:
    return "\n\n".join(
        f"Source: {d.metadata.get('source_file', '?')} | Row: {d.metadata.get('row_index', '?')}\n{d.page_content.strip()}"
        for d in docs
    )


@tool
def query_csv(query: str) -> str:
    """
    Query structured data: BPSS case tracker, employment history,
    and document inventory (CSV and XLSX files).
    Use this for status flags, check completion, risk levels,
    employment gaps, document counts, and candidate list queries.
    Recognises candidate IDs like CAND-103 directly.
    """
    logger.info("tool_query_csv", query=query[:150])

    structured = _pandas_query(query)
    docs = similarity_search(query, get_embeddings(), source_type="csv")
    semantic = _format_docs(docs)

    if not structured and not semantic:
        return f"No matches found.\nAvailable data:\n{format_dataframe_summary()}"

    parts = []
    if structured:
        parts.append(f"[Structured]\n{structured}")
    if semantic:
        parts.append(f"[Semantic]\n{semantic}")

    return "\n\n---\n\n".join(parts)
