# @title Page-Level Citations Report
from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse
import sys

import ipywidgets as widgets
import pandas as pd
from IPython.display import HTML, display

try:
    refresh_master_df
    create_filter_panel
    apply_filters
    export_dataframe
    normalize_role_labels
except NameError as exc:  # pragma: no cover - notebook guard
    raise RuntimeError("Run Cell 11a first to load master dataset helpers.") from exc


AI_ROLE = globals().get("AI_ROLE_LABEL", "AI System")


REPORT_COLUMNS = [
    "Domain",
    "Page Title",
    "Full URL",
    "Total Page Citations",
    "Domain Citation Share %",
    "Total Outputs Citing Page",
    "% of Total Outputs Citing Page",
    "Avg Citations per Output",
    "Total Prompt Runs Citing Page",
    "Unique Prompts Citing Page",
    "% of Unique Prompts Citing Page",
    "% of Total Prompt Runs Citing Page",
    "Prompt Repetition Rate",
    "Source Character",
    "Overall Average Rank",
    "Avg Rank on Repeated Prompts",
    "Avg Rank on Unique Prompts",
    "% of Citations in Top 3",
    "Rank Quality Score",
    "First Seen Timestamp",
    "Last Seen Timestamp",
    "Days Since Last Seen",
    "Recent Citation Velocity",
    "Predictability Score",
    "Topical Authority Score",
    "Overall Impact Score",
    "Run Labels",
    "Execution IDs",
    "Run Dates",
]

DEFAULT_DISPLAY_COLUMNS = [
    "Domain",
    "Page Title",
    "Full URL",
    "Total Page Citations",
    "Total Outputs Citing Page",
    "Avg Citations per Output",
    "Total Prompt Runs Citing Page",
    "Unique Prompts Citing Page",
    "% of Unique Prompts Citing Page",
    "% of Total Prompt Runs Citing Page",
    "Avg Rank on Repeated Prompts",
    "Avg Rank on Unique Prompts",
    "% of Citations in Top 3",
    "Rank Quality Score",
    "Topical Authority Score",
    "Overall Impact Score",
]


SORT_OPTIONS = [
    ("Domain (A-Z)", "Domain"),
    ("Page Title (A-Z)", "Page Title"),
    *[(col, col) for col in REPORT_COLUMNS if col not in {"Domain", "Page Title", "Full URL"}],
]

LAST_REPORT = pd.DataFrame(columns=REPORT_COLUMNS)
LAST_VIEW = pd.DataFrame(columns=REPORT_COLUMNS)
FULL_REPORT = pd.DataFrame(columns=REPORT_COLUMNS)


def maybe_trigger_download(path: Path) -> None:
    if "google.colab" in sys.modules:
        try:
            from google.colab import files  # type: ignore

            files.download(str(path))
        except Exception:
            pass


def clean_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    sanitized = url.strip()
    if not sanitized:
        return ""
    if "?utm_" in sanitized:
        sanitized = sanitized.split("?utm_")[0]
    return sanitized


def extract_domain(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.netloc.replace("www.", "")
    return netloc or parsed.path or url


def derive_title_from_url(url: str) -> str:
    parsed = urlparse(url)
    slug = unquote(Path(parsed.path).name or parsed.netloc or url)
    normalized = slug.replace("-", " ").replace("_", " ").strip()
    if not normalized:
        return extract_domain(url)
    return normalized.title()


def derive_run_id(row: pd.Series) -> str:
    exec_id = row.get("execution_id") or "exec"
    turn = row.get("turn_or_run")
    if pd.isna(turn) or turn == "":
        turn = row.get("unit_id") or "unit"
    return f"{exec_id}|{turn}"


def derive_prompt_run_id(row: pd.Series) -> str:
    unit = row.get("unit_id")
    if pd.notna(unit) and unit != "":
        return str(unit)
    return row.get("run_id") or derive_run_id(row)


def label_source_character(rate: float) -> str:
    if rate >= 2.0:
        return "Niche Specialist"
    if rate >= 1.2:
        return "Focused Authority"
    return "General Authority"


def compute_rank_quality_score(ranks: pd.Series) -> float:
    if ranks.empty:
        return 0.0
    capped = ranks.clip(lower=1, upper=50)
    points = capped.apply(lambda r: max(0, 11 - min(int(r), 10)))
    return float(points.sum() / (10 * len(points)) * 100)


def compute_topical_authority_score(unique_prompt_pct: float, avg_rank_unique: float | None) -> float:
    if pd.isna(avg_rank_unique) or avg_rank_unique is None:
        rank_component = 50.0
    else:
        rank_component = max(0.0, (10 - min(avg_rank_unique, 10)) / 9 * 100)
    return (unique_prompt_pct + rank_component) / 2


def format_timestamp_short(value: object) -> str:
    if value in (None, "", "NaT"):
        return ""
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return ""
    return ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M")


def format_numeric(value: object, decimals: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{decimals}f}"


def _build_domain_options(df: pd.DataFrame) -> list[str]:
    if df.empty or "citation_url" not in df.columns:
        return ["All"]
    domains = {
        extract_domain(clean_url(url))
        for url in df["citation_url"].dropna()
        if isinstance(url, str) and clean_url(url)
    }
    return ["All"] + sorted(domains)


POTENTIAL_OUTPUT_COLUMNS = [
    "message_text",
    "assistant_response",
    "model_response",
    "response_text",
    "citation_title",
    "citation_text",
    "query_or_topic",
]


def build_citation_intelligence(df: pd.DataFrame) -> pd.DataFrame:
    advisor_rows = df[df["role"] == AI_ROLE].copy()
    citations = advisor_rows.dropna(subset=["citation_url"]).copy()
    if citations.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    citations["run_id"] = citations.apply(derive_run_id, axis=1)
    citations["prompt_run_id"] = citations.apply(derive_prompt_run_id, axis=1)
    citations["clean_url"] = citations["citation_url"].apply(clean_url)
    citations = citations[citations["clean_url"] != ""]
    if citations.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    citations["domain"] = citations["clean_url"].apply(extract_domain)
    citations = citations[citations["domain"] != ""]
    citations["citation_rank"] = pd.to_numeric(citations["citation_rank"], errors="coerce")
    citations["row_timestamp"] = pd.to_datetime(citations["row_timestamp"], errors="coerce", utc=True)
    citations["query_or_topic"] = citations["query_or_topic"].fillna("Unknown prompt")

    total_outputs = max(citations["run_id"].nunique(), 1)
    total_prompt_runs = max(citations["prompt_run_id"].nunique(), 1)
    total_unique_prompts = max(citations["query_or_topic"].nunique(), 1)

    prompt_run_counts = citations.groupby("query_or_topic")["prompt_run_id"].nunique().to_dict()
    domain_totals = citations.groupby("domain")["clean_url"].count().to_dict()

    now_ts = pd.Timestamp.utcnow()
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    window_start = now_ts - pd.Timedelta(days=7)

    records: list[dict[str, object]] = []
    for url, group in citations.groupby("clean_url"):
        domain = group["domain"].iloc[0]
        if "citation_title" in group.columns and group["citation_title"].notna().any():
            page_title = group["citation_title"].dropna().iloc[0]
        else:
            page_title = derive_title_from_url(url)
        total_citations = int(group.shape[0])
        domain_total = max(domain_totals.get(domain, total_citations), 1)
        domain_share_pct = (total_citations / domain_total) * 100

        outputs_with_domain = group["run_id"].nunique()
        outputs_pct = (outputs_with_domain / total_outputs) * 100

        prompt_runs_with_domain = group["prompt_run_id"].nunique()
        prompt_runs_pct = (prompt_runs_with_domain / total_prompt_runs) * 100

        unique_prompts = group["query_or_topic"].nunique()
        unique_prompts_pct = (unique_prompts / total_unique_prompts) * 100

        prompt_repetition_rate = prompt_runs_with_domain / max(unique_prompts, 1)
        source_character = label_source_character(prompt_repetition_rate)

        ranks = group["citation_rank"].dropna()
        overall_avg_rank = ranks.mean() if not ranks.empty else None
        top3_pct = float((ranks <= 3).mean() * 100) if not ranks.empty else 0.0

        repeated_mask = group["query_or_topic"].map(lambda q: prompt_run_counts.get(q, 0) > 1)
        repeated_ranks = group.loc[repeated_mask, "citation_rank"].dropna()
        avg_rank_repeated = repeated_ranks.mean() if not repeated_ranks.empty else None

        unique_mask = group["query_or_topic"].map(lambda q: prompt_run_counts.get(q, 0) <= 1)
        unique_ranks = group.loc[unique_mask, "citation_rank"].dropna()
        avg_rank_unique = unique_ranks.mean() if not unique_ranks.empty else None

        rank_quality_score = compute_rank_quality_score(ranks)

        timestamps = group["row_timestamp"].dropna()
        first_seen = timestamps.min()
        last_seen = timestamps.max()
        days_since_last_seen = (now_ts - last_seen).days if pd.notna(last_seen) else None
        recent_count = int((timestamps >= window_start).sum())
        recent_velocity = (recent_count / total_citations) * 100 if total_citations else 0.0

        predictability_score = (outputs_pct + top3_pct) / 2
        topical_authority_score = compute_topical_authority_score(unique_prompts_pct, avg_rank_unique)
        overall_impact_score = (
            (predictability_score * 0.4)
            + (topical_authority_score * 0.3)
            + (recent_velocity * 0.2)
            + (domain_share_pct * 0.1)
        )
        run_labels = sorted(
            {
                str(label).strip()
                for label in group.get("run_label", pd.Series(dtype=str)).dropna()
                if str(label).strip()
            }
        )
        execution_ids = sorted(
            {
                str(exe).strip()
                for exe in group.get("execution_id", pd.Series(dtype=str)).dropna()
                if str(exe).strip()
            }
        )
        run_dates_set: set[str] = set()
        for ts in group.get("row_timestamp", pd.Series(dtype="datetime64[ns]")).dropna():
            try:
                if getattr(ts, "tzinfo", None) is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                run_dates_set.add(ts.strftime("%Y-%m-%d"))
            except Exception:
                continue
        run_dates = sorted(run_dates_set)

        records.append(
            {
                "Domain": domain,
                "Page Title": page_title,
                "Full URL": url,
                "Total Page Citations": total_citations,
                "Domain Citation Share %": domain_share_pct,
                "Total Outputs Citing Page": outputs_with_domain,
                "% of Total Outputs Citing Page": outputs_pct,
                "Avg Citations per Output": total_citations / max(outputs_with_domain, 1),
                "Total Prompt Runs Citing Page": prompt_runs_with_domain,
                "Unique Prompts Citing Page": unique_prompts,
                "% of Unique Prompts Citing Page": unique_prompts_pct,
                "% of Total Prompt Runs Citing Page": prompt_runs_pct,
                "Prompt Repetition Rate": prompt_repetition_rate,
                "Source Character": source_character,
                "Overall Average Rank": overall_avg_rank,
                "Avg Rank on Repeated Prompts": avg_rank_repeated,
                "Avg Rank on Unique Prompts": avg_rank_unique,
                "% of Citations in Top 3": top3_pct,
                "Rank Quality Score": rank_quality_score,
                "First Seen Timestamp": first_seen.isoformat() if pd.notna(first_seen) else None,
                "Last Seen Timestamp": last_seen.isoformat() if pd.notna(last_seen) else None,
                "Days Since Last Seen": days_since_last_seen,
                "Recent Citation Velocity": recent_velocity,
                "Predictability Score": predictability_score,
                "Topical Authority Score": topical_authority_score,
                "Overall Impact Score": overall_impact_score,
                "Run Labels": ", ".join(run_labels),
                "Execution IDs": ", ".join(execution_ids),
                "Run Dates": ", ".join(run_dates),
            }
        )

    return pd.DataFrame(records, columns=REPORT_COLUMNS)


def format_domain_link(domain: str) -> str:
    if not isinstance(domain, str):
        return ""
    return f'<a href="https://{domain}" target="_blank">{domain}</a>'


def format_url_link(url: str) -> str:
    if not isinstance(url, str):
        return ""
    return f'<a href="{url}" target="_blank">Open</a>'


master_df = normalize_role_labels(refresh_master_df(force_rebuild=False))
FULL_REPORT = build_citation_intelligence(master_df)
filters = create_filter_panel(master_df)
clear_filters_button = create_clear_filters_button(filters)
if "role" in filters and isinstance(filters["role"], widgets.Widget):
    role_options = getattr(filters["role"], "options", [])
    option_values = [opt[1] if isinstance(opt, tuple) else opt for opt in role_options]
    if AI_ROLE in option_values:
        filters["role"].value = AI_ROLE
        filters["role"]._default = AI_ROLE  # type: ignore[attr-defined]

domain_dropdown = widgets.Dropdown(description="Domain:", options=_build_domain_options(master_df), value="All")
domain_search = widgets.Text(description="Domain contains:", placeholder="contains…")
page_search = widgets.Text(description="Page contains:", placeholder="url or slug…")
output_text_filter = widgets.Text(description="Output text:", placeholder="prompt/output contains…")

column_checkboxes = {
    col: widgets.Checkbox(description=col, value=col in DEFAULT_DISPLAY_COLUMNS)
    for col in REPORT_COLUMNS
}
column_picker_grid = widgets.GridBox(
    list(column_checkboxes.values()),
    layout=widgets.Layout(grid_template_columns="repeat(2, 50%)", grid_gap="4px 12px"),
)
column_picker_box = widgets.VBox([widgets.HTML("<b>Select columns to display:</b>"), column_picker_grid])
column_picker = widgets.Accordion(children=[column_picker_box])
column_picker.set_title(0, "+ Column picker")
column_picker.selected_index = None

sort_column = widgets.Dropdown(description="Sort by:", options=SORT_OPTIONS, value="Overall Impact Score")
sort_order = widgets.ToggleButtons(
    description="Order:",
    options=[("Desc", "desc"), ("Asc", "asc")],
    value="desc",
)
heading_html = widgets.HTML("<h3>Page-Level Citations Report</h3>")
refresh_button = widgets.Button(description="Refresh data", icon="refresh", button_style="primary")
rebuild_button = widgets.Button(description="Force rebuild", icon="repeat", button_style="danger")
export_table_button = widgets.Button(description="Export All", icon="table", button_style="warning")
export_view_button = widgets.Button(description="Export View", icon="eye", button_style="success")
export_highlights_button = widgets.Button(description="Export highlights", icon="star")
export_highlights_button.style.button_color = "#e0e0e0"

summary_output = widgets.Output()
table_header = widgets.HTML("<h4>Page performance metrics</h4>")
table_output = widgets.Output(layout=widgets.Layout(max_height="520px", overflow="auto"))
message_output = widgets.Output()
download_link_html = widgets.HTML()


def sort_report(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ascending = sort_order.value == "asc"
    return df.sort_values(sort_column.value, ascending=ascending)


def compute_summary(df: pd.DataFrame) -> dict[str, int | float]:
    advisor_rows = df[df["role"] == AI_ROLE].copy()
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


def apply_custom_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    working = df.copy()
    if "citation_url" in working.columns:
        working["__domain"] = working["citation_url"].apply(lambda url: extract_domain(clean_url(url)) if isinstance(url, str) else "")
    else:
        working["__domain"] = ""

    if domain_dropdown.value != "All":
        working = working[working["__domain"] == domain_dropdown.value]

    domain_contains = domain_search.value.strip().lower()
    if domain_contains:
        working = working[working["__domain"].str.contains(domain_contains, case=False, na=False)]

    page_contains = page_search.value.strip().lower()
    if page_contains and "citation_url" in working.columns:
        working = working[working["citation_url"].astype(str).str.lower().str.contains(page_contains, na=False)]

    output_contains = output_text_filter.value.strip().lower()
    if output_contains:
        mask = pd.Series(False, index=working.index)
        for col in POTENTIAL_OUTPUT_COLUMNS:
            if col in working.columns:
                mask = mask | working[col].astype(str).str.lower().str.contains(output_contains, na=False)
        if "citation_url" in working.columns:
            mask = mask | working["citation_url"].astype(str).str.lower().str.contains(output_contains, na=False)
        working = working[mask]

    return working.drop(columns=["__domain"], errors="ignore")


def refresh_domain_options_widget() -> None:
    options = _build_domain_options(master_df)
    current = domain_dropdown.value if domain_dropdown.value in options else "All"
    domain_dropdown.options = options
    domain_dropdown.value = current


def update_display(_=None):
    global LAST_REPORT, LAST_VIEW

    filtered = apply_filters(master_df, filters)
    filtered = apply_custom_filters(filtered)
    report_df = build_citation_intelligence(filtered)
    summary = compute_summary(filtered)

    with summary_output:
        summary_output.clear_output()
        display(HTML("<h4>Summary metrics</h4>"))
        if filtered.empty:
            display(HTML("<p>No rows match the selected filters.</p>"))
            display(HTML("<br>"))
        elif report_df.empty:
            display(HTML("<p>No citations available for the current selection.</p>"))
            display(HTML("<br>"))
        else:
            avg_cites_per_output = summary["total_citations"] / max(summary["total_outputs"], 1)
            avg_cites_per_prompt = summary["total_citations"] / max(summary["total_prompts"], 1)
            summary_df = pd.DataFrame(
                [
                    {
                        "Total Prompts": f"{summary['total_prompts']:,}",
                        "Prompts w/ Citations": f"{summary['prompt_with_cites']:,}",
                        "Total Outputs": f"{summary['total_outputs']:,}",
                        "Outputs w/ Citations": f"{summary['outputs_with_cites']:,}",
                        "Citations": f"{summary['total_citations']:,}",
                        "Avg Cites/Output": f"{avg_cites_per_output:.2f}",
                        "Avg Cites/Prompt": f"{avg_cites_per_prompt:.2f}",
                        "Unique Pages": f"{report_df.shape[0]:,}",
                        "Unique Domains": f"{report_df['Domain'].nunique():,}",
                    }
                ]
            )
            summary_html = summary_df.to_html(index=False, escape=False)
            display(HTML(summary_html))
            display(HTML("<br>"))

    with table_output:
        table_output.clear_output()
        if report_df.empty:
            LAST_REPORT = pd.DataFrame(columns=REPORT_COLUMNS)
            LAST_VIEW = pd.DataFrame(columns=REPORT_COLUMNS)
            table_header.value = "<h4>Page performance metrics</h4>"
            return

        sorted_df = sort_report(report_df)
        LAST_REPORT = sorted_df
        rows = filters["rows"].value
        display_df = sorted_df.head(rows).copy()
        if display_df.empty:
            print("No rows to display.")
            return

        LAST_VIEW = display_df.copy()
        display_df["Domain"] = display_df["Domain"].apply(format_domain_link)
        if "Full URL" in display_df.columns:
            display_df["Full URL"] = display_df["Full URL"].apply(format_url_link)

        percent_columns = [
            "Domain Citation Share %",
            "% of Total Outputs Citing Page",
            "% of Unique Prompts Citing Page",
            "% of Total Prompt Runs Citing Page",
            "% of Citations in Top 3",
            "Rank Quality Score",
            "Recent Citation Velocity",
            "Predictability Score",
            "Topical Authority Score",
        ]

        for col in percent_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda v: f"{format_numeric(v, 2)}%" if pd.notna(v) else "")

        numeric_columns = [
            "Avg Citations per Output",
            "Prompt Repetition Rate",
            "Overall Impact Score",
            "Overall Average Rank",
            "Avg Rank on Repeated Prompts",
            "Avg Rank on Unique Prompts",
        ]
        for col in numeric_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda v: format_numeric(v, 2))

        selected_columns = [col for col, cb in column_checkboxes.items() if cb.value]
        if not selected_columns:
            selected_columns = DEFAULT_DISPLAY_COLUMNS

        for ts_col in ("First Seen Timestamp", "Last Seen Timestamp"):
            if ts_col in display_df.columns:
                display_df[ts_col] = display_df[ts_col].map(format_timestamp_short)

        table_header.value = "<h4>Page performance metrics</h4>"
        subset = [c for c in selected_columns if c in display_df.columns]
        table_html = display_df[subset].to_html(escape=False, index=False, classes="domain-report-table")
        styled_html = """
        <style>
        .domain-report-table thead th {
            position: sticky;
            top: 0;
            background: #f6f6f6;
            z-index: 1;
        }
        .domain-report-table tbody td {
            vertical-align: top;
        }
        </style>
        """ + table_html
        display(HTML(styled_html))


def _refresh_full_report():
    global FULL_REPORT
    FULL_REPORT = build_citation_intelligence(master_df)


def handle_refresh(_):
    global master_df
    master_df = normalize_role_labels(refresh_master_df(force_rebuild=False))
    _refresh_full_report()
    refresh_domain_options_widget()
    update_display()


def handle_rebuild(_):
    global master_df
    master_df = normalize_role_labels(refresh_master_df(force_rebuild=True))
    _refresh_full_report()
    refresh_domain_options_widget()
    update_display()


def _render_download_link(path: Path) -> str:
    return f'<a href="file://{path.resolve()}" target="_blank">{path.name}</a>'


def handle_export_table(_):
    if FULL_REPORT.empty:
        with message_output:
            message_output.clear_output()
            print("⚠️ Nothing to export yet.")
        download_link_html.value = ""
        return
    export_df = sort_report(FULL_REPORT)
    path = export_dataframe(export_df, "page_citation_report_all")
    maybe_trigger_download(path)
    with message_output:
        message_output.clear_output()
        print("📄 Exported full dataset.")
    download_link_html.value = _render_download_link(path)


def handle_export_view(_):
    if LAST_REPORT.empty:
        with message_output:
            message_output.clear_output()
            print("⚠️ No filtered rows to export yet.")
        download_link_html.value = ""
        return
    path = export_dataframe(LAST_REPORT, "page_citation_report_view")
    maybe_trigger_download(path)
    with message_output:
        message_output.clear_output()
        print("📄 Exported current view.")
    download_link_html.value = _render_download_link(path)


def handle_export_highlights(_):
    if LAST_REPORT.empty:
        with message_output:
            message_output.clear_output()
            print("⚠️ Nothing to export yet.")
        download_link_html.value = ""
        return
    df = LAST_REPORT
    highlights = pd.concat(
        [
            df.nlargest(5, "Overall Impact Score").assign(Highlight="impact_top5"),
            df.nlargest(5, "Recent Citation Velocity").assign(Highlight="velocity_top5"),
            df.nlargest(5, "Topical Authority Score").assign(Highlight="authority_top5"),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["Full URL", "Highlight"])
    path = export_dataframe(highlights, "citation_highlights")
    maybe_trigger_download(path)
    with message_output:
        message_output.clear_output()
        print("📄 Highlight tables exported.")
    download_link_html.value = _render_download_link(path)


refresh_button.on_click(handle_refresh)
rebuild_button.on_click(handle_rebuild)
export_table_button.on_click(handle_export_table)
export_view_button.on_click(handle_export_view)
export_highlights_button.on_click(handle_export_highlights)

for widget in list(filters.values()) + [sort_column, sort_order]:
    widget.observe(update_display, names="value")

for checkbox in column_checkboxes.values():
    checkbox.observe(update_display, names="value")

def _sync_column_picker_title(change):
    if change["name"] == "selected_index":
        column_picker.set_title(0, "+ Column picker" if change["new"] is None else "− Column picker")

column_picker.observe(_sync_column_picker_title, names="selected_index")

domain_dropdown.observe(update_display, names="value")
domain_search.observe(update_display, names="value")
page_search.observe(update_display, names="value")
output_text_filter.observe(update_display, names="value")

refresh_domain_options_widget()

filters["query_text"].description = "Prompt search:"
filters["message_text"].description = "Message search:"

for control in (domain_dropdown, domain_search, page_search, output_text_filter, filters["query_text"], filters["message_text"]):
    control.style = {"description_width": "150px"}
    control.layout = widgets.Layout(width="300px")

controls = widgets.VBox(
    [
        widgets.HBox(
            [refresh_button, rebuild_button, export_table_button, export_view_button, export_highlights_button, clear_filters_button],
            layout=widgets.Layout(margin="0 0 10px 0"),
        ),
        widgets.HBox(
            [filters["scenario"], filters["run_label"], filters["role"], filters["persona"]],
            layout=widgets.Layout(margin="0 0 6px 0"),
        ),
        widgets.HBox(
            [filters["model"], filters["execution"], filters["country"]],
            layout=widgets.Layout(margin="0 0 6px 0"),
        ),
        widgets.HBox(
            [filters["query_dropdown"], filters["query_text"], filters["message_text"]],
            layout=widgets.Layout(margin="0 0 6px 0"),
        ),
        widgets.HBox(
            [domain_dropdown, domain_search, page_search],
            layout=widgets.Layout(margin="0 0 6px 0"),
        ),
        widgets.HBox(
            [output_text_filter, filters["unit"], filters["turn"], filters["citations_only"], filters["rows"]],
            layout=widgets.Layout(margin="0 0 6px 0"),
        ),
        widgets.HBox(
            [sort_column, sort_order],
            layout=widgets.Layout(margin="0 0 6px 0"),
        ),
    ],
    layout=widgets.Layout(width="100%"),
)

status_box = widgets.VBox([message_output, download_link_html], layout=widgets.Layout(width="100%"))
app_layout = widgets.VBox(
    [heading_html, controls, column_picker, summary_output, table_header, table_output, status_box],
    layout=widgets.Layout(width="100%"),
)

display(app_layout)
update_display()
