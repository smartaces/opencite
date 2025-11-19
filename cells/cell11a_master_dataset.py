import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Callable

import pandas as pd
import ipywidgets as widgets

AI_ROLE_LABEL = "AI System"
LEGACY_ROLE_ALIASES = {"advisor"}


def _load_paths():
    if 'PATHS' in globals():
        return {k: Path(v) for k, v in PATHS.items()}

    config_path = os.environ.get("WORKSPACE_CONFIG")
    if not config_path or not Path(config_path).is_file():
        raise RuntimeError("Workspace not configured. Run the workspace setup cell first.")

    with open(config_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return {k: Path(v) for k, v in data["paths"].items()}


PATH_MAP = _load_paths()
CSV_DIR = PATH_MAP["csv_output"]
CACHE_DIR = CSV_DIR / "report_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MASTER_PATH = CACHE_DIR / "master_detail.csv"
MANIFEST_PATH = CACHE_DIR / "manifest.json"


def _load_manifest() -> Dict[str, Any]:
    if MANIFEST_PATH.is_file():
        with MANIFEST_PATH.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    return {"processed_files": {}}


def _save_manifest(manifest: Dict[str, Any]) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)


def _read_detail_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["detail_file"] = str(path)
    return df


def refresh_master_df(force_rebuild: bool = False) -> pd.DataFrame:
    detail_files = sorted(CSV_DIR.glob("*_detail_*.csv"))
    manifest = _load_manifest()
    processed = manifest.get("processed_files", {})

    if force_rebuild or not MASTER_PATH.is_file():
        dfs = [_read_detail_file(f) for f in detail_files]
        master_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        master_df.drop_duplicates(inplace=True)
        master_df.to_csv(MASTER_PATH, index=False)
        manifest["processed_files"] = {
            str(f): {
                "rows": int(master_df[master_df["detail_file"] == str(f)].shape[0]),
                "ingested_at": datetime.utcnow().isoformat(),
            }
            for f in detail_files
        }
        _save_manifest(manifest)
        return master_df

    new_files = [f for f in detail_files if str(f) not in processed]
    if not new_files:
        return pd.read_csv(MASTER_PATH)

    master_df = pd.read_csv(MASTER_PATH)
    new_dfs = [_read_detail_file(f) for f in new_files]
    new_df = pd.concat(new_dfs, ignore_index=True)
    combined = pd.concat([master_df, new_df], ignore_index=True).drop_duplicates()
    combined.to_csv(MASTER_PATH, index=False)

    for f in new_files:
        rows = int(new_df[new_df["detail_file"] == str(f)].shape[0])
        processed[str(f)] = {"rows": rows, "ingested_at": datetime.utcnow().isoformat()}

    manifest["processed_files"] = processed
    _save_manifest(manifest)
    return combined


def refresh(force_rebuild: bool = False) -> pd.DataFrame:
    """Alias so older cells that call refresh() continue to work."""
    return refresh_master_df(force_rebuild=force_rebuild)


def load_master_df() -> pd.DataFrame:
    if MASTER_PATH.is_file():
        return pd.read_csv(MASTER_PATH)
    return refresh_master_df(force_rebuild=False)


def dropdown_from_series(name: str, series: pd.Series, allow_blank: bool = False) -> widgets.Dropdown:
    options = ["All"]
    if allow_blank:
        options.append("Blank")
    values = sorted({v for v in series.dropna().unique()} if not series.empty else [])
    options.extend(values)
    return widgets.Dropdown(description=name, options=options, value="All")


def create_filter_panel(df: pd.DataFrame, *, include_prompt_dropdown: bool = True, include_run_filters: bool = True) -> Dict[str, widgets.Widget]:
    widgets_dict: Dict[str, widgets.Widget] = {
        "scenario": dropdown_from_series("Scenario:", df["scenario"] if "scenario" in df else pd.Series(dtype=str)),
        "role": dropdown_from_series("Role:", df["role"] if "role" in df else pd.Series(dtype=str)),
        "persona": dropdown_from_series("Persona:", df["persona_profile"] if "persona_profile" in df else pd.Series(dtype=str), allow_blank=True),
        "model": dropdown_from_series("Model:", df["model"] if "model" in df else pd.Series(dtype=str)),
        "execution": dropdown_from_series("Execution ID:", df["execution_id"] if "execution_id" in df else pd.Series(dtype=str)),
        "country": dropdown_from_series("Country:", df["location_country"] if "location_country" in df else pd.Series(dtype=str), allow_blank=True),
        "query_dropdown": dropdown_from_series("Prompt:", df["query_or_topic"] if "query_or_topic" in df else pd.Series(dtype=str)),
        "query_text": widgets.Text(description="Query search:", placeholder="contains…"),
        "message_text": widgets.Text(description="Message search:", placeholder="contains…"),
        "citations_only": widgets.Checkbox(description="Citations only", value=False),
        "rows": widgets.IntSlider(description="Rows", value=25, min=5, max=200, step=5),
    }

    if include_run_filters:
        widgets_dict["unit"] = dropdown_from_series("Unit ID:", df["unit_id"] if "unit_id" in df else pd.Series(dtype=str))
        widgets_dict["turn"] = dropdown_from_series("Turn/Run #:", df["turn_or_run"] if "turn_or_run" in df else pd.Series(dtype=str))
    else:
        widgets_dict["unit"] = widgets.Dropdown(description="Unit ID:", options=["All"], value="All")
        widgets_dict["turn"] = widgets.Dropdown(description="Turn/Run #:", options=["All"], value="All")

    if not include_prompt_dropdown:
        widgets_dict["query_dropdown"].layout.display = "none"

    return widgets_dict


def apply_filters(df: pd.DataFrame, filters: Dict[str, widgets.Widget]) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df.copy()

    if "scenario" in filtered.columns and filters["scenario"].value != "All":
        filtered = filtered[filtered["scenario"] == filters["scenario"].value]
    if "role" in filtered.columns and filters["role"].value != "All":
        filtered = filtered[filtered["role"] == filters["role"].value]
    if "model" in filtered.columns and filters["model"].value != "All":
        filtered = filtered[filtered["model"] == filters["model"].value]
    if "execution_id" in filtered.columns and filters["execution"].value != "All":
        filtered = filtered[filtered["execution_id"] == filters["execution"].value]

    # Persona and country filters include blank option
    if "persona_profile" in filtered.columns:
        if filters["persona"].value == "Blank":
            filtered = filtered[filtered["persona_profile"].isna()]
        elif filters["persona"].value != "All":
            filtered = filtered[filtered["persona_profile"] == filters["persona"].value]

    if "location_country" in filtered.columns:
        if filters["country"].value == "Blank":
            filtered = filtered[filtered["location_country"].isna()]
        elif filters["country"].value != "All":
            filtered = filtered[filtered["location_country"] == filters["country"].value]

    if filters["citations_only"].value and "citation_url" in filtered.columns:
        filtered = filtered[filtered["citation_url"].notna()]

    if "unit_id" in filtered.columns and filters["unit"].value != "All":
        filtered = filtered[filtered["unit_id"] == filters["unit"].value]
    if "turn_or_run" in filtered.columns and filters["turn"].value != "All":
        value = filters["turn"].value
        filtered = filtered[filtered["turn_or_run"].astype(str) == str(value)]

    if "query_or_topic" in filtered.columns:
        dropdown_value = filters["query_dropdown"].value
        if dropdown_value != "All":
            filtered = filtered[filtered["query_or_topic"] == dropdown_value]

    query_text = filters["query_text"].value.strip().lower()
    if query_text and "query_or_topic" in filtered.columns:
        filtered = filtered[
            filtered["query_or_topic"].str.lower().str.contains(query_text, na=False)
        ]

    message_text = filters["message_text"].value.strip().lower()
    if message_text:
        filtered = filtered[
            (filtered["message_text"].str.lower().str.contains(message_text, na=False) if "message_text" in filtered.columns else False)
            | (filtered["citation_title"].str.lower().str.contains(message_text, na=False) if "citation_title" in filtered.columns else False)
            | (filtered["citation_url"].str.lower().str.contains(message_text, na=False) if "citation_url" in filtered.columns else False)
        ]

    return filtered


def export_dataframe(df: pd.DataFrame, name: str) -> Path:
    if df.empty:
        raise ValueError("No data to export.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = CSV_DIR / f"{name}_{timestamp}.csv"
    df.to_csv(path, index=False)
    return path


def normalize_role_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with legacy role names mapped to the canonical label."""
    if "role" not in df.columns or df.empty:
        return df
    normalized = df.copy()
    rename_map = {alias: AI_ROLE_LABEL for alias in LEGACY_ROLE_ALIASES}
    normalized["role"] = normalized["role"].replace(rename_map)
    return normalized


print("✅ Master dataset helpers ready. Use refresh_master_df()/refresh(), create_filter_panel(), apply_filters(), and export_dataframe().")
