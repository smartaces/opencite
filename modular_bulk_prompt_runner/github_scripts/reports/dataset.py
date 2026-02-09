"""Manifest-based master dataset builder with single-pass enrichment."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .url_utils import clean_url, derive_run_id, derive_prompt_run_id, extract_domain

AI_ROLE_LABEL = "AI System"
LEGACY_ROLE_ALIASES = {"advisor"}


# ------------------------------------------------------------------
# Manifest helpers
# ------------------------------------------------------------------

def _load_manifest(cache_dir: Path) -> Dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    return {"processed_files": {}}


def _save_manifest(manifest: Dict[str, Any], cache_dir: Path) -> None:
    manifest_path = cache_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)


def _read_detail_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["detail_file"] = str(path)
    return df


# ------------------------------------------------------------------
# Master DataFrame loading
# ------------------------------------------------------------------

def refresh_master_df(
    csv_dir: Path,
    cache_dir: Path,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Manifest-based incremental loading of detail CSVs.

    Parameters
    ----------
    csv_dir : Path
        Directory containing ``*_detail_*.csv`` files.
    cache_dir : Path
        Directory for ``manifest.json`` and ``master_detail.csv``.
    force_rebuild : bool
        If True, discard cache and rebuild from all files.
    """
    csv_dir = Path(csv_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    master_path = cache_dir / "master_detail.csv"

    detail_files = sorted(csv_dir.glob("*_detail_*.csv"))
    manifest = _load_manifest(cache_dir)
    processed = manifest.get("processed_files", {})

    if force_rebuild or not master_path.is_file():
        dfs = [_read_detail_file(f) for f in detail_files]
        master_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        master_df.drop_duplicates(inplace=True)
        master_df.to_csv(master_path, index=False)
        manifest["processed_files"] = {
            str(f): {
                "rows": int(master_df[master_df["detail_file"] == str(f)].shape[0]),
                "ingested_at": datetime.utcnow().isoformat(),
            }
            for f in detail_files
        }
        _save_manifest(manifest, cache_dir)
        return master_df

    new_files = [f for f in detail_files if str(f) not in processed]
    if not new_files:
        return pd.read_csv(master_path)

    master_df = pd.read_csv(master_path)
    new_dfs = [_read_detail_file(f) for f in new_files]
    new_df = pd.concat(new_dfs, ignore_index=True)
    combined = pd.concat([master_df, new_df], ignore_index=True).drop_duplicates()
    combined.to_csv(master_path, index=False)

    for f in new_files:
        rows = int(new_df[new_df["detail_file"] == str(f)].shape[0])
        processed[str(f)] = {"rows": rows, "ingested_at": datetime.utcnow().isoformat()}

    manifest["processed_files"] = processed
    _save_manifest(manifest, cache_dir)
    return combined


# ------------------------------------------------------------------
# Role normalization
# ------------------------------------------------------------------

def normalize_role_labels(df: pd.DataFrame, ai_role: str = AI_ROLE_LABEL) -> pd.DataFrame:
    """Map legacy role names (e.g. 'advisor') to canonical label."""
    if "role" not in df.columns or df.empty:
        return df
    normalized = df.copy()
    rename_map = {alias: ai_role for alias in LEGACY_ROLE_ALIASES}
    normalized["role"] = normalized["role"].replace(rename_map)
    return normalized


# ------------------------------------------------------------------
# Single-pass enrichment
# ------------------------------------------------------------------

def enrich_master_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns once so report builders don't repeat this work.

    Derived columns: ``run_id``, ``prompt_run_id``, ``clean_url``,
    ``domain``, ``citation_rank`` (numeric), ``row_timestamp`` (UTC datetime).
    """
    if df.empty:
        return df

    enriched = df.copy()

    # Composite keys
    enriched["run_id"] = enriched.apply(derive_run_id, axis=1)
    enriched["prompt_run_id"] = enriched.apply(derive_prompt_run_id, axis=1)

    # URL cleaning
    if "citation_url" in enriched.columns:
        enriched["clean_url"] = enriched["citation_url"].apply(clean_url)
        enriched["domain"] = enriched["clean_url"].apply(
            lambda u: extract_domain(u) if u else ""
        )
    else:
        enriched["clean_url"] = ""
        enriched["domain"] = ""

    # Numeric rank
    if "citation_rank" in enriched.columns:
        enriched["citation_rank"] = pd.to_numeric(
            enriched["citation_rank"], errors="coerce"
        )

    # Timestamp parsing
    if "row_timestamp" in enriched.columns:
        enriched["row_timestamp"] = pd.to_datetime(
            enriched["row_timestamp"], errors="coerce", utc=True
        )

    # Fill missing prompts
    if "query_or_topic" in enriched.columns:
        enriched["query_or_topic"] = enriched["query_or_topic"].fillna("Unknown prompt")

    return enriched


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------

def export_dataframe(df: pd.DataFrame, name: str, output_dir: Path) -> Path:
    """Write DataFrame to timestamped CSV and return the path."""
    if df.empty:
        raise ValueError("No data to export.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{name}_{timestamp}.csv"
    df.to_csv(path, index=False)
    return path
