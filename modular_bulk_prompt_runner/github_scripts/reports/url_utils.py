"""URL cleaning, domain extraction, and ID derivation utilities."""
from __future__ import annotations

from pathlib import PurePosixPath
from urllib.parse import unquote, urlparse

import pandas as pd


def clean_url(url: str) -> str:
    """Strip whitespace and remove UTM tracking parameters."""
    if not isinstance(url, str):
        return ""
    sanitized = url.strip()
    if not sanitized:
        return ""
    if "?utm_" in sanitized:
        sanitized = sanitized.split("?utm_")[0]
    return sanitized


def extract_domain(url: str) -> str:
    """Extract domain from URL, stripping www. prefix."""
    parsed = urlparse(url)
    netloc = parsed.netloc.replace("www.", "")
    return netloc or parsed.path or url


def derive_title_from_url(url: str) -> str:
    """Generate a human-readable title from a URL path slug."""
    parsed = urlparse(url)
    slug = unquote(PurePosixPath(parsed.path).name or parsed.netloc or url)
    normalized = slug.replace("-", " ").replace("_", " ").strip()
    if not normalized:
        return extract_domain(url)
    return normalized.title()


def derive_run_id(row: pd.Series) -> str:
    """Create composite key: execution_id|turn_or_run."""
    exec_id = row.get("execution_id") or "exec"
    turn = row.get("turn_or_run")
    if pd.isna(turn) or turn == "":
        turn = row.get("unit_id") or "unit"
    return f"{exec_id}|{turn}"


def derive_prompt_run_id(row: pd.Series) -> str:
    """Use unit_id preferentially, fallback to run_id composite."""
    unit = row.get("unit_id")
    if pd.notna(unit) and unit != "":
        return str(unit)
    return row.get("run_id") or derive_run_id(row)
