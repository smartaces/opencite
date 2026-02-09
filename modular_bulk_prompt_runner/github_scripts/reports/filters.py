"""Two-tier filter system for analytics reports.

Tier 1 — Data filters: change the analysis universe, require recomputation.
Tier 2 — View filters: narrow/sort displayed results, instant operations.
"""
from __future__ import annotations

from typing import Dict

import ipywidgets as widgets
import pandas as pd

from .url_utils import clean_url, extract_domain

# Columns that may contain searchable output text
POTENTIAL_OUTPUT_COLUMNS = [
    "message_text",
    "assistant_response",
    "model_response",
    "response_text",
    "citation_title",
    "citation_text",
    "query_or_topic",
]


# ------------------------------------------------------------------
# Widget helpers
# ------------------------------------------------------------------

def _dropdown_from_series(
    name: str, series: pd.Series, allow_blank: bool = False
) -> widgets.Dropdown:
    options: list[tuple[str, object]] = [("All", "All")]
    if allow_blank:
        options.append(("Blank", "Blank"))
    values = sorted({v for v in series.dropna().unique()} if not series.empty else [])
    options.extend([(str(v), v) for v in values])
    widget = widgets.Dropdown(description=name, options=options, value="All")
    widget._default = "All"  # type: ignore[attr-defined]
    return widget


def _safe_series(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column] if column in df.columns else pd.Series(dtype=str)


# ------------------------------------------------------------------
# Tier 1 — Data filters
# ------------------------------------------------------------------

def create_data_filter_panel(
    df: pd.DataFrame, include_provider: bool = True
) -> Dict[str, widgets.Widget]:
    """Create Tier 1 filter widgets from a master DataFrame.

    Changing any of these requires re-running the aggregation engine.
    """
    widgets_dict: Dict[str, widgets.Widget] = {
        "scenario": _dropdown_from_series("Scenario:", _safe_series(df, "scenario")),
        "run_label": _dropdown_from_series("Run label:", _safe_series(df, "run_label"), allow_blank=True),
        "role": _dropdown_from_series("Role:", _safe_series(df, "role")),
        "persona": _dropdown_from_series("Persona:", _safe_series(df, "persona_profile"), allow_blank=True),
        "model": _dropdown_from_series("Model:", _safe_series(df, "model")),
        "execution": _dropdown_from_series("Execution ID:", _safe_series(df, "execution_id")),
        "country": _dropdown_from_series("Country:", _safe_series(df, "location_country"), allow_blank=True),
        "query_dropdown": _dropdown_from_series("Prompt:", _safe_series(df, "query_or_topic")),
        "query_text": widgets.Text(description="Prompt search:", placeholder="contains\u2026"),
        "message_text": widgets.Text(description="Message search:", placeholder="contains\u2026"),
        "citations_only": widgets.Checkbox(description="Citations only", value=False),
        "unit": _dropdown_from_series("Unit ID:", _safe_series(df, "unit_id")),
        "turn": _dropdown_from_series("Turn/Run #:", _safe_series(df, "turn_or_run")),
        "rows": widgets.IntSlider(description="Rows", value=25, min=5, max=200, step=5),
    }

    if include_provider:
        widgets_dict["provider"] = _dropdown_from_series(
            "Provider:", _safe_series(df, "provider")
        )

    # Store defaults for reset
    widgets_dict["citations_only"]._default = False  # type: ignore[attr-defined]
    widgets_dict["rows"]._default = 25  # type: ignore[attr-defined]

    return widgets_dict


def apply_data_filters(
    df: pd.DataFrame, filters: Dict[str, widgets.Widget]
) -> pd.DataFrame:
    """Apply all Tier 1 filters and return filtered DataFrame."""
    if df.empty:
        return df

    filtered = df.copy()

    def _apply_dropdown(column: str, key: str, allow_blank: bool = False) -> None:
        nonlocal filtered
        if key not in filters or column not in filtered.columns:
            return
        value = filters[key].value
        if value in ("All", None):
            return
        if allow_blank and value == "Blank":
            filtered = filtered[filtered[column].isna()]
        else:
            filtered = filtered[filtered[column] == value]

    _apply_dropdown("scenario", "scenario")
    _apply_dropdown("run_label", "run_label", allow_blank=True)
    _apply_dropdown("role", "role")
    _apply_dropdown("model", "model")
    _apply_dropdown("execution_id", "execution")
    _apply_dropdown("persona_profile", "persona", allow_blank=True)
    _apply_dropdown("location_country", "country", allow_blank=True)
    _apply_dropdown("unit_id", "unit")
    _apply_dropdown("turn_or_run", "turn")

    if "provider" in filters:
        _apply_dropdown("provider", "provider")

    if "query_or_topic" in filtered.columns:
        _apply_dropdown("query_or_topic", "query_dropdown")

    if filters["citations_only"].value and "citation_url" in filtered.columns:
        filtered = filtered[filtered["citation_url"].notna()]

    query_text = filters["query_text"].value.strip().lower()
    if query_text and "query_or_topic" in filtered.columns:
        filtered = filtered[
            filtered["query_or_topic"].str.lower().str.contains(query_text, na=False)
        ]

    message_text = filters["message_text"].value.strip().lower()
    if message_text:
        mask = pd.Series(False, index=filtered.index)
        for col in ("message_text", "citation_title", "citation_url"):
            if col in filtered.columns:
                mask = mask | filtered[col].astype(str).str.lower().str.contains(
                    message_text, na=False
                )
        filtered = filtered[mask]

    return filtered


# ------------------------------------------------------------------
# Tier 2 — View filters
# ------------------------------------------------------------------

def create_view_filter_panel(
    report_columns: list[str],
    default_display_columns: list[str],
    sort_options: list[tuple[str, str]],
    default_sort: str = "Overall Impact Score",
) -> Dict[str, widgets.Widget]:
    """Create Tier 2 filter widgets for an already-computed report.

    These only filter/sort the display — no recomputation needed.
    """
    widgets_dict: Dict[str, widgets.Widget] = {
        "domain_dropdown": widgets.Dropdown(
            description="Domain:", options=["All"], value="All"
        ),
        "domain_search": widgets.Text(
            description="Domain contains:", placeholder="contains\u2026"
        ),
        "page_search": widgets.Text(
            description="Page contains:", placeholder="url or slug\u2026"
        ),
        "output_text_filter": widgets.Text(
            description="Output text:", placeholder="prompt/output contains\u2026"
        ),
        "sort_column": widgets.Dropdown(
            description="Sort by:", options=sort_options, value=default_sort
        ),
        "sort_order": widgets.ToggleButtons(
            description="Order:",
            options=[("Desc", "desc"), ("Asc", "asc")],
            value="desc",
        ),
    }

    # Column picker
    column_checkboxes = {
        col: widgets.Checkbox(description=col, value=(col in default_display_columns))
        for col in report_columns
    }
    widgets_dict["column_checkboxes"] = column_checkboxes  # type: ignore[assignment]

    return widgets_dict


def apply_view_filters(
    report_df: pd.DataFrame,
    view_filters: Dict[str, widgets.Widget],
    rows_limit: int = 25,
) -> pd.DataFrame:
    """Filter, sort, and slice an already-computed report DataFrame."""
    if report_df.empty:
        return report_df

    working = report_df.copy()

    # Domain dropdown
    domain_val = view_filters["domain_dropdown"].value
    if domain_val != "All" and "Domain" in working.columns:
        working = working[working["Domain"] == domain_val]

    # Domain text search
    domain_search = view_filters["domain_search"].value.strip().lower()
    if domain_search and "Domain" in working.columns:
        working = working[
            working["Domain"].astype(str).str.lower().str.contains(domain_search, na=False)
        ]

    # Page text search
    page_search = view_filters["page_search"].value.strip().lower()
    if page_search:
        for col in ("Full URL", "Page Title"):
            if col in working.columns:
                working = working[
                    working[col].astype(str).str.lower().str.contains(page_search, na=False)
                ]
                break

    # Output text filter
    output_text = view_filters["output_text_filter"].value.strip().lower()
    if output_text:
        text_cols = [c for c in POTENTIAL_OUTPUT_COLUMNS if c in working.columns]
        if text_cols:
            mask = pd.Series(False, index=working.index)
            for col in text_cols:
                mask = mask | working[col].astype(str).str.lower().str.contains(
                    output_text, na=False
                )
            working = working[mask]

    # Sort
    sort_col = view_filters["sort_column"].value
    if sort_col in working.columns:
        ascending = view_filters["sort_order"].value == "asc"
        working = working.sort_values(sort_col, ascending=ascending)

    # Slice
    return working.head(rows_limit)


def update_domain_options(
    view_filters: Dict[str, widgets.Widget],
    enriched_df: pd.DataFrame,
) -> None:
    """Refresh the domain dropdown from current enriched data."""
    if enriched_df.empty or "domain" not in enriched_df.columns:
        options = ["All"]
    else:
        domains = sorted(
            {d for d in enriched_df["domain"].dropna().unique() if d}
        )
        options = ["All"] + domains

    dd = view_filters["domain_dropdown"]
    current = dd.value if dd.value in options else "All"
    dd.options = options
    dd.value = current


# ------------------------------------------------------------------
# Reset helpers
# ------------------------------------------------------------------

def reset_filter_widgets(filters: Dict[str, widgets.Widget]) -> None:
    """Reset all filter widgets to their defaults."""
    for widget in filters.values():
        if isinstance(widget, dict):
            # Column checkboxes dict
            continue
        if isinstance(widget, widgets.Dropdown):
            widget.value = getattr(widget, "_default", "All")
        elif isinstance(widget, widgets.Text):
            widget.value = ""
        elif isinstance(widget, widgets.Checkbox):
            widget.value = getattr(widget, "_default", False)
        elif isinstance(widget, widgets.IntSlider):
            widget.value = getattr(widget, "_default", widget.value)
        elif isinstance(widget, widgets.ToggleButtons):
            widget.value = widget.options[0][1] if widget.options else widget.value


def create_clear_filters_button(
    data_filters: Dict[str, widgets.Widget],
    view_filters: Dict[str, widgets.Widget] | None = None,
    description: str = "Clear filters",
) -> widgets.Button:
    """Button that resets both Tier 1 and Tier 2 filter widgets."""
    button = widgets.Button(
        description=description, icon="times", button_style="warning"
    )

    def _on_click(_):
        reset_filter_widgets(data_filters)
        if view_filters:
            reset_filter_widgets(view_filters)

    button.on_click(_on_click)
    return button
