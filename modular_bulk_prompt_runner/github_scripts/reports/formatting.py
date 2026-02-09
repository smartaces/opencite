"""Display formatters and summary computation for report output."""
from __future__ import annotations

import pandas as pd

from .url_utils import clean_url, derive_run_id, derive_prompt_run_id, extract_domain


def format_timestamp_short(value: object) -> str:
    """UTC datetime to ``%Y-%m-%d %H:%M`` string."""
    if value in (None, "", "NaT"):
        return ""
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return ""
    return ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M")


def format_numeric(value: object, decimals: int = 2) -> str:
    """Fixed-decimal string, empty for NaN."""
    if pd.isna(value):
        return ""
    return f"{float(value):.{decimals}f}"


def format_domain_link(domain: str) -> str:
    """Clickable HTML link to a domain."""
    if not isinstance(domain, str) or not domain:
        return ""
    return f'<a href="https://{domain}" target="_blank">{domain}</a>'


def format_url_link(url: str) -> str:
    """Clickable HTML link labelled 'Open'."""
    if not isinstance(url, str) or not url:
        return ""
    return f'<a href="{url}" target="_blank">Open</a>'


def truncate_prompt_text(value: object, limit: int = 140) -> str:
    """Shorten prompt text for display."""
    if not isinstance(value, str):
        return "" if value is None else str(value)
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


def compute_summary(df: pd.DataFrame, ai_role: str = "AI System") -> dict:
    """Unified summary metrics dict from raw detail data."""
    advisor_rows = df[df["role"] == ai_role].copy() if "role" in df.columns else df.copy()
    if advisor_rows.empty:
        return {
            "total_prompts": 0,
            "prompt_with_cites": 0,
            "total_outputs": 0,
            "outputs_with_cites": 0,
            "total_citations": 0,
        }

    advisor_rows["run_id"] = advisor_rows.apply(derive_run_id, axis=1)
    advisor_rows["prompt_run_id"] = advisor_rows.apply(derive_prompt_run_id, axis=1)
    advisor_rows["query_or_topic"] = advisor_rows["query_or_topic"].fillna("Unknown prompt")

    total_prompts = advisor_rows["query_or_topic"].nunique()
    total_outputs = advisor_rows["run_id"].nunique()

    cited = advisor_rows.dropna(subset=["citation_url"]).copy()
    if cited.empty:
        return {
            "total_prompts": total_prompts,
            "prompt_with_cites": 0,
            "total_outputs": total_outputs,
            "outputs_with_cites": 0,
            "total_citations": 0,
        }

    return {
        "total_prompts": total_prompts,
        "prompt_with_cites": cited["query_or_topic"].nunique(),
        "total_outputs": total_outputs,
        "outputs_with_cites": cited["run_id"].nunique(),
        "total_citations": int(cited.shape[0]),
    }
