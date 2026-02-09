# cell_08_reports.py
"""
Reports - Domain, Page, and Prompt-level citation intelligence.

Single cell with 4 tabs:
  0. Dataset  - Refresh / rebuild master data, export master CSV
  1. Domain   - 27-column domain intelligence report
  2. Page     - 26-column page intelligence report
  3. Prompt   - 29-column prompt insights (Domain+Page / Domain-only views)

Each report tab has independent two-tier filtering:
  Tier 1 (data filters) trigger recomputation.
  Tier 2 (view filters) instantly filter/sort the display.
"""

import json
import os
import sys
from pathlib import Path

import ipywidgets as widgets
import pandas as pd
from IPython.display import HTML, display


# ============================================================================
# WORKSPACE PATHS
# ============================================================================

def _ensure_paths():
    if "PATHS" in globals() and globals()["PATHS"]:
        return {k: Path(v) for k, v in globals()["PATHS"].items()}
    config_path = Path(os.environ.get("WORKSPACE_CONFIG", ""))
    if not config_path.is_file():
        raise RuntimeError("Workspace not configured. Run the workspace setup cell first.")
    with open(config_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return {k: Path(v) for k, v in data["paths"].items()}


PATHS = _ensure_paths()
CSV_OUTPUT_DIR = Path(PATHS["csv_output"])
REPORT_CACHE_DIR = CSV_OUTPUT_DIR / "report_cache"
REPORT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# IMPORTS FROM reports/ PACKAGE
# ============================================================================

from reports import (
    refresh_master_df,
    enrich_master_df,
    normalize_role_labels,
    export_dataframe,
    create_data_filter_panel,
    apply_data_filters,
    create_view_filter_panel,
    apply_view_filters,
    create_clear_filters_button,
    update_domain_options,
    compute_summary,
    format_domain_link,
    format_url_link,
    format_numeric,
    format_timestamp_short,
    truncate_prompt_text,
    build_domain_report,
    build_page_report,
    build_prompt_insights,
)
from reports.domain_report import (
    REPORT_COLUMNS as DOMAIN_COLUMNS,
    DEFAULT_DISPLAY_COLUMNS as DOMAIN_DEFAULT_COLS,
    SORT_OPTIONS as DOMAIN_SORT_OPTIONS,
)
from reports.page_report import (
    REPORT_COLUMNS as PAGE_COLUMNS,
    DEFAULT_DISPLAY_COLUMNS as PAGE_DEFAULT_COLS,
    SORT_OPTIONS as PAGE_SORT_OPTIONS,
)
from reports.prompt_report import (
    REPORT_COLUMNS as PROMPT_COLUMNS,
    DEFAULT_DISPLAY_COLUMNS as PROMPT_DEFAULT_COLS,
    SORT_OPTIONS as PROMPT_SORT_OPTIONS,
)


# ============================================================================
# COLAB DOWNLOAD HELPER
# ============================================================================

def _maybe_trigger_download(path: Path) -> None:
    if "google.colab" in sys.modules:
        try:
            from google.colab import files  # type: ignore
            files.download(str(path))
        except Exception:
            pass


# ============================================================================
# TABLE STYLING
# ============================================================================

TABLE_STYLE = """
<style>
.report-table thead th {
    position: sticky;
    top: 0;
    background: #f6f6f6;
    z-index: 1;
}
.report-table tbody td {
    vertical-align: top;
}
</style>
"""

PERCENT_COLUMNS = {
    "Domain Citation Share %",
    "% of Total Outputs Citing Page",
    "% of Unique Prompts Citing Page",
    "% of Total Prompt Runs Citing Page",
    "% of Citations in Top 3",
    "Rank Quality Score",
    "Recent Citation Velocity",
    "Predictability Score",
    "Topical Authority Score",
    "% of Total Prompt Runs",
    "% of Outputs with Citations",
    "% of Prompt Runs Citing Domain",
    "% of Prompt Runs Citing Page",
    "% of Outputs Citing Domain",
    "% of Outputs Citing Page",
    "Recent Domain Velocity",
}

NUMERIC_COLUMNS = {
    "Avg Citations per Output",
    "Prompt Repetition Rate",
    "Overall Impact Score",
    "Overall Average Rank",
    "Avg Rank on Repeated Prompts",
    "Avg Rank on Unique Prompts",
    "Avg Domain Rank",
    "Avg Page Rank",
    "Avg Citations per Output (Domain)",
    "Avg Citations per Output (Page)",
}

TIMESTAMP_COLUMNS = {
    "First Seen Timestamp",
    "Last Seen Timestamp",
    "Prompt Last Seen",
    "Page Last Seen Timestamp",
}


def _format_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply percentage, numeric, timestamp, and link formatting for HTML display."""
    display_df = df.copy()

    if "Domain" in display_df.columns:
        display_df["Domain"] = display_df["Domain"].apply(format_domain_link)
    if "Full URL" in display_df.columns:
        display_df["Full URL"] = display_df["Full URL"].apply(format_url_link)
    if "Prompt Text" in display_df.columns:
        display_df["Prompt Text"] = display_df["Prompt Text"].apply(truncate_prompt_text)

    for col in PERCENT_COLUMNS:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(
                lambda v: f"{format_numeric(v, 2)}%" if pd.notna(v) and v != "---" else (v if v == "---" else "")
            )

    for col in NUMERIC_COLUMNS:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(
                lambda v: format_numeric(v, 2) if pd.notna(v) and v != "---" else (v if v == "---" else "")
            )

    for col in TIMESTAMP_COLUMNS:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(
                lambda v: format_timestamp_short(v) if isinstance(v, str) and v != "---" else (v if v == "---" else "")
            )

    return display_df


# ============================================================================
# SHARED STATE
# ============================================================================

AI_ROLE = "AI System"
_state = {
    "master_df": pd.DataFrame(),
    "enriched_df": pd.DataFrame(),
}


def _load_and_enrich(force_rebuild: bool = False) -> None:
    raw = refresh_master_df(CSV_OUTPUT_DIR, REPORT_CACHE_DIR, force_rebuild=force_rebuild)
    raw = normalize_role_labels(raw, ai_role=AI_ROLE)
    _state["master_df"] = raw
    _state["enriched_df"] = enrich_master_df(raw)


# Initial load
_load_and_enrich(force_rebuild=False)


# ============================================================================
# TAB 0 — DATASET
# ============================================================================

dataset_status = widgets.HTML()
dataset_output = widgets.Output()
dataset_message = widgets.Output()

dataset_refresh_btn = widgets.Button(description="Refresh data", icon="refresh", button_style="primary")
dataset_rebuild_btn = widgets.Button(description="Force rebuild", icon="repeat", button_style="danger")
dataset_export_btn = widgets.Button(description="Export master CSV", icon="download", button_style="success")


def _update_dataset_status():
    raw = _state["master_df"]
    enriched = _state["enriched_df"]
    detail_files = sorted(CSV_OUTPUT_DIR.glob("*_detail_*.csv"))
    dataset_status.value = (
        f"<b>Detail files:</b> {len(detail_files)} &nbsp;&bull;&nbsp; "
        f"<b>Total rows:</b> {len(raw):,} &nbsp;&bull;&nbsp; "
        f"<b>Enriched rows:</b> {len(enriched):,}"
    )


def _on_dataset_refresh(_):
    _load_and_enrich(force_rebuild=False)
    _update_dataset_status()
    with dataset_message:
        dataset_message.clear_output()
        print("Data refreshed.")


def _on_dataset_rebuild(_):
    _load_and_enrich(force_rebuild=True)
    _update_dataset_status()
    with dataset_message:
        dataset_message.clear_output()
        print("Full rebuild complete.")


def _on_dataset_export(_):
    raw = _state["master_df"]
    if raw.empty:
        with dataset_message:
            dataset_message.clear_output()
            print("No data to export.")
        return
    path = export_dataframe(raw, "master_detail", CSV_OUTPUT_DIR)
    _maybe_trigger_download(path)
    with dataset_message:
        dataset_message.clear_output()
        print(f"Exported: {path.name}")


dataset_refresh_btn.on_click(_on_dataset_refresh)
dataset_rebuild_btn.on_click(_on_dataset_rebuild)
dataset_export_btn.on_click(_on_dataset_export)

dataset_tab = widgets.VBox([
    widgets.HTML("<h3>Dataset Management</h3>"),
    widgets.HBox([dataset_refresh_btn, dataset_rebuild_btn, dataset_export_btn]),
    dataset_status,
    dataset_message,
    dataset_output,
])

_update_dataset_status()


# ============================================================================
# REPORT TAB FACTORY
# ============================================================================

def _build_report_tab(
    title: str,
    report_columns: list[str],
    default_display_columns: list[str],
    sort_options: list[tuple[str, str]],
    default_sort: str,
    build_fn,
    report_name: str,
    has_view_toggle: bool = False,
):
    """Create an independent report tab with Tier 1 + Tier 2 filtering."""

    # State for this tab
    tab_state = {
        "full_report": pd.DataFrame(columns=report_columns),
        "filtered_report": pd.DataFrame(columns=report_columns),
        "display_report": pd.DataFrame(columns=report_columns),
    }
    # For prompt report: second DataFrame
    if has_view_toggle:
        tab_state["full_domain_report"] = pd.DataFrame(columns=report_columns)
        tab_state["full_page_report"] = pd.DataFrame(columns=report_columns)

    # Tier 1 data filters
    data_filters = create_data_filter_panel(_state["enriched_df"], include_provider=True)
    if "role" in data_filters and isinstance(data_filters["role"], widgets.Widget):
        role_opts = getattr(data_filters["role"], "options", [])
        opt_values = [o[1] if isinstance(o, tuple) else o for o in role_opts]
        if AI_ROLE in opt_values:
            data_filters["role"].value = AI_ROLE
            data_filters["role"]._default = AI_ROLE  # type: ignore[attr-defined]

    # Tier 2 view filters
    view_filters = create_view_filter_panel(report_columns, default_display_columns, sort_options, default_sort)

    clear_btn = create_clear_filters_button(data_filters, view_filters)

    # Action buttons
    refresh_btn = widgets.Button(description="Refresh data", icon="refresh", button_style="primary")
    rebuild_btn = widgets.Button(description="Force rebuild", icon="repeat", button_style="danger")
    export_all_btn = widgets.Button(description="Export All", icon="table", button_style="warning")
    export_view_btn = widgets.Button(description="Export View", icon="eye", button_style="success")
    export_highlights_btn = widgets.Button(description="Export highlights", icon="star")
    export_highlights_btn.style.button_color = "#e0e0e0"

    # Optional view toggle (prompt report)
    view_toggle = None
    if has_view_toggle:
        view_toggle = widgets.ToggleButtons(
            description="View:",
            options=[("Domain + Page", "with_pages"), ("Domain only", "domain_only")],
            value="with_pages",
        )

    # Output areas
    summary_output = widgets.Output()
    table_output = widgets.Output(layout=widgets.Layout(max_height="520px", overflow="auto"))
    message_output = widgets.Output()
    download_html = widgets.HTML()

    # Column picker
    column_checkboxes = view_filters.get("column_checkboxes", {})
    if column_checkboxes:
        cb_grid = widgets.GridBox(
            list(column_checkboxes.values()),
            layout=widgets.Layout(grid_template_columns="repeat(2, 50%)", grid_gap="4px 12px"),
        )
        cb_box = widgets.VBox([widgets.HTML("<b>Select columns to display:</b>"), cb_grid])
        column_picker = widgets.Accordion(children=[cb_box])
        column_picker.set_title(0, "+ Column picker")
        column_picker.selected_index = None

        def _sync_picker_title(change):
            if change["name"] == "selected_index":
                column_picker.set_title(0, "+ Column picker" if change["new"] is None else "- Column picker")
        column_picker.observe(_sync_picker_title, names="selected_index")
    else:
        column_picker = widgets.HTML("")

    # ------------------------------------------------------------------
    # Core update logic
    # ------------------------------------------------------------------

    def _recompute(_=None):
        """Tier 1 change: re-filter enriched data, rebuild report."""
        enriched = _state["enriched_df"]
        filtered_data = apply_data_filters(enriched, data_filters)
        update_domain_options(view_filters, filtered_data)

        if has_view_toggle:
            domain_df, page_df = build_fn(filtered_data, ai_role=AI_ROLE)
            tab_state["full_domain_report"] = domain_df
            tab_state["full_page_report"] = page_df
            active = domain_df if view_toggle.value == "domain_only" else page_df
            tab_state["full_report"] = active
        else:
            report_df = build_fn(filtered_data, ai_role=AI_ROLE)
            tab_state["full_report"] = report_df

        _update_view()

    def _update_view(_=None):
        """Tier 2 change: filter/sort/display the already-computed report."""
        report_df = tab_state["full_report"]

        if has_view_toggle and view_toggle is not None:
            if view_toggle.value == "domain_only":
                report_df = tab_state["full_domain_report"]
            else:
                report_df = tab_state["full_page_report"]

        rows_limit = data_filters["rows"].value
        view_df = apply_view_filters(report_df, view_filters, rows_limit=rows_limit)
        tab_state["filtered_report"] = report_df
        tab_state["display_report"] = view_df

        # Summary
        with summary_output:
            summary_output.clear_output()
            display(HTML("<h4>Summary metrics</h4>"))
            if report_df.empty:
                display(HTML("<p>No citations available for the current selection.</p>"))
            elif not has_view_toggle:
                summary = compute_summary(_state["enriched_df"], ai_role=AI_ROLE)
                avg_per_output = summary["total_citations"] / max(summary["total_outputs"], 1)
                avg_per_prompt = summary["total_citations"] / max(summary["total_prompts"], 1)

                extra = {}
                if "Domain" in report_df.columns and "Unique Pages Cited" not in report_df.columns:
                    # Page report
                    extra["Unique Pages"] = f"{report_df.shape[0]:,}"
                    extra["Unique Domains"] = f"{report_df['Domain'].nunique():,}"
                elif "Unique Pages Cited" in report_df.columns:
                    # Domain report
                    extra["Unique Domains"] = f"{report_df['Domain'].nunique():,}"
                    extra["Unique Pages"] = f"{int(report_df['Unique Pages Cited'].sum()):,}"

                row = {
                    "Total Prompts": f"{summary['total_prompts']:,}",
                    "Prompts w/ Citations": f"{summary['prompt_with_cites']:,}",
                    "Total Outputs": f"{summary['total_outputs']:,}",
                    "Outputs w/ Citations": f"{summary['outputs_with_cites']:,}",
                    "Citations": f"{summary['total_citations']:,}",
                    "Avg Cites/Output": f"{avg_per_output:.2f}",
                    "Avg Cites/Prompt": f"{avg_per_prompt:.2f}",
                    **extra,
                }
                display(HTML(pd.DataFrame([row]).to_html(index=False, escape=False)))
            else:
                # Prompt report summary
                domain_df = tab_state.get("full_domain_report", pd.DataFrame())
                page_df = tab_state.get("full_page_report", pd.DataFrame())
                prompts = domain_df["Prompt Text"].nunique() if not domain_df.empty else 0
                domains = domain_df["Domain"].nunique() if not domain_df.empty else 0
                pages = page_df["Full URL"].nunique() if not page_df.empty else 0
                total_cites = int(page_df["Total Page Citations"].sum()) if not page_df.empty and "Total Page Citations" in page_df else 0
                row = {
                    "Prompts w/ Citations": f"{prompts:,}",
                    "Domains": f"{domains:,}",
                    "Pages": f"{pages:,}",
                    "Total Page Citations": f"{total_cites:,}",
                }
                display(HTML(pd.DataFrame([row]).to_html(index=False, escape=False)))

        # Table
        with table_output:
            table_output.clear_output()
            if view_df.empty:
                return

            display_df = _format_display_df(view_df)

            # Domain-only view: blank out page columns
            if has_view_toggle and view_toggle is not None and view_toggle.value == "domain_only":
                for col in [
                    "Page Title", "Full URL", "Total Page Citations", "Avg Page Rank",
                    "% of Prompt Runs Citing Page", "% of Outputs Citing Page",
                    "Avg Citations per Output (Page)", "First Seen Timestamp",
                    "Page Last Seen Timestamp", "Days Since Last Seen",
                ]:
                    if col in display_df.columns:
                        display_df[col] = "---"

            # Column selection
            selected = [col for col, cb in column_checkboxes.items() if cb.value] if column_checkboxes else report_columns
            if not selected:
                selected = default_display_columns
            valid = [c for c in selected if c in display_df.columns]

            table_html = display_df[valid].to_html(escape=False, index=False, classes="report-table")
            display(HTML(TABLE_STYLE + table_html))

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _on_refresh(_):
        _load_and_enrich(force_rebuild=False)
        _update_dataset_status()
        _recompute()

    def _on_rebuild(_):
        _load_and_enrich(force_rebuild=True)
        _update_dataset_status()
        _recompute()

    def _on_export_all(_):
        full = tab_state["full_report"]
        if full.empty:
            with message_output:
                message_output.clear_output()
                print("Nothing to export yet.")
            return
        path = export_dataframe(full, f"{report_name}_all", CSV_OUTPUT_DIR)
        _maybe_trigger_download(path)
        with message_output:
            message_output.clear_output()
            print(f"Exported full dataset: {path.name}")
        download_html.value = f'<a href="file://{path.resolve()}" target="_blank">{path.name}</a>'

    def _on_export_view(_):
        view = tab_state["display_report"]
        if view.empty:
            with message_output:
                message_output.clear_output()
                print("No filtered rows to export yet.")
            return
        path = export_dataframe(view, f"{report_name}_view", CSV_OUTPUT_DIR)
        _maybe_trigger_download(path)
        with message_output:
            message_output.clear_output()
            print(f"Exported current view: {path.name}")
        download_html.value = f'<a href="file://{path.resolve()}" target="_blank">{path.name}</a>'

    def _on_export_highlights(_):
        full = tab_state["filtered_report"]
        if full.empty:
            with message_output:
                message_output.clear_output()
                print("Nothing to export yet.")
            return
        # Pick highlight columns that exist
        highlight_cols = ["Overall Impact Score", "Recent Citation Velocity", "Topical Authority Score",
                          "Predictability Score", "Recent Domain Velocity"]
        parts = []
        for col in highlight_cols:
            if col in full.columns and full[col].notna().any():
                parts.append(full.nlargest(5, col).assign(Highlight=f"{col}_top5"))
        if not parts:
            with message_output:
                message_output.clear_output()
                print("No highlight columns available.")
            return
        highlights = pd.concat(parts, ignore_index=True)
        # Deduplicate by primary key + highlight
        dedup_cols = ["Domain", "Highlight"]
        if "Full URL" in highlights.columns:
            dedup_cols.append("Full URL")
        if "Prompt Text" in highlights.columns:
            dedup_cols.append("Prompt Text")
        valid_dedup = [c for c in dedup_cols if c in highlights.columns]
        highlights = highlights.drop_duplicates(subset=valid_dedup)
        path = export_dataframe(highlights, f"{report_name}_highlights", CSV_OUTPUT_DIR)
        _maybe_trigger_download(path)
        with message_output:
            message_output.clear_output()
            print(f"Highlights exported: {path.name}")
        download_html.value = f'<a href="file://{path.resolve()}" target="_blank">{path.name}</a>'

    # ------------------------------------------------------------------
    # Wire events
    # ------------------------------------------------------------------

    refresh_btn.on_click(_on_refresh)
    rebuild_btn.on_click(_on_rebuild)
    export_all_btn.on_click(_on_export_all)
    export_view_btn.on_click(_on_export_view)
    export_highlights_btn.on_click(_on_export_highlights)

    # Tier 1 changes → recompute
    for key, w in data_filters.items():
        if isinstance(w, widgets.Widget):
            w.observe(_recompute, names="value")

    # Tier 2 changes → update view only
    for key in ("domain_dropdown", "domain_search", "page_search", "output_text_filter", "sort_column", "sort_order"):
        if key in view_filters and isinstance(view_filters[key], widgets.Widget):
            view_filters[key].observe(_update_view, names="value")

    if column_checkboxes:
        for cb in column_checkboxes.values():
            cb.observe(_update_view, names="value")

    if view_toggle is not None:
        view_toggle.observe(_update_view, names="value")

    # Style text inputs
    for key in ("domain_dropdown", "domain_search", "page_search", "output_text_filter"):
        if key in view_filters:
            view_filters[key].style = {"description_width": "150px"}
            view_filters[key].layout = widgets.Layout(width="300px")
    for key in ("query_text", "message_text"):
        if key in data_filters:
            data_filters[key].style = {"description_width": "150px"}
            data_filters[key].layout = widgets.Layout(width="300px")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    button_row = [refresh_btn, rebuild_btn, export_all_btn, export_view_btn, export_highlights_btn, clear_btn]

    filter_rows = [
        widgets.HBox(button_row, layout=widgets.Layout(margin="0 0 10px 0")),
        widgets.HBox(
            [data_filters["scenario"], data_filters["run_label"], data_filters["role"], data_filters["persona"]],
            layout=widgets.Layout(margin="0 0 6px 0"),
        ),
    ]

    # Provider filter row
    model_row = [data_filters["model"], data_filters["execution"], data_filters["country"]]
    if "provider" in data_filters:
        model_row.insert(0, data_filters["provider"])
    filter_rows.append(widgets.HBox(model_row, layout=widgets.Layout(margin="0 0 6px 0")))

    filter_rows.append(
        widgets.HBox(
            [data_filters["query_dropdown"], data_filters["query_text"], data_filters["message_text"]],
            layout=widgets.Layout(margin="0 0 6px 0"),
        )
    )
    filter_rows.append(
        widgets.HBox(
            [view_filters["domain_dropdown"], view_filters["domain_search"], view_filters["page_search"]],
            layout=widgets.Layout(margin="0 0 6px 0"),
        )
    )

    misc_row = [view_filters["output_text_filter"], data_filters["unit"], data_filters["turn"],
                data_filters["citations_only"], data_filters["rows"]]
    filter_rows.append(widgets.HBox(misc_row, layout=widgets.Layout(margin="0 0 6px 0")))

    sort_row = [view_filters["sort_column"], view_filters["sort_order"]]
    if view_toggle is not None:
        sort_row.append(view_toggle)
    filter_rows.append(widgets.HBox(sort_row, layout=widgets.Layout(margin="0 0 6px 0")))

    controls = widgets.VBox(filter_rows, layout=widgets.Layout(width="100%"))
    status_box = widgets.VBox([message_output, download_html], layout=widgets.Layout(width="100%"))

    tab_widget = widgets.VBox([
        widgets.HTML(f"<h3>{title}</h3>"),
        controls,
        column_picker,
        summary_output,
        widgets.HTML(f"<h4>{title} metrics</h4>"),
        table_output,
        status_box,
    ], layout=widgets.Layout(width="100%"))

    # Initial computation
    _recompute()

    return tab_widget


# ============================================================================
# BUILD ALL TABS
# ============================================================================

domain_tab = _build_report_tab(
    title="Domain Citations Report",
    report_columns=DOMAIN_COLUMNS,
    default_display_columns=DOMAIN_DEFAULT_COLS,
    sort_options=DOMAIN_SORT_OPTIONS,
    default_sort="Overall Impact Score",
    build_fn=build_domain_report,
    report_name="domain_report",
)

page_tab = _build_report_tab(
    title="Page-Level Citations Report",
    report_columns=PAGE_COLUMNS,
    default_display_columns=PAGE_DEFAULT_COLS,
    sort_options=PAGE_SORT_OPTIONS,
    default_sort="Overall Impact Score",
    build_fn=build_page_report,
    report_name="page_report",
)

prompt_tab = _build_report_tab(
    title="Prompt-Level Insights Report",
    report_columns=PROMPT_COLUMNS,
    default_display_columns=PROMPT_DEFAULT_COLS,
    sort_options=PROMPT_SORT_OPTIONS,
    default_sort="Predictability Score",
    build_fn=build_prompt_insights,
    report_name="prompt_insights",
    has_view_toggle=True,
)


# ============================================================================
# ASSEMBLE TAB WIDGET
# ============================================================================

tab = widgets.Tab(children=[dataset_tab, domain_tab, page_tab, prompt_tab])
tab.set_title(0, "Dataset")
tab.set_title(1, "Domain Report")
tab.set_title(2, "Page Report")
tab.set_title(3, "Prompt Report")

display(tab)
