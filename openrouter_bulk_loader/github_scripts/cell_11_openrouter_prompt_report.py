# @title Prompt-Level Insights Report (Cell 11d)
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
    "Prompt Text",
    "Scenario",
    "Persona",
    "Model",
    "Total Prompt Runs",
    "Unique Prompt Runs",
    "% of Total Prompt Runs",
    "Total Outputs",
    "% of Outputs with Citations",
    "Prompt Last Seen",
    "Domain",
    "Page Title",
    "Unique Pages Cited",
    "Full URL",
    "Total Domain Citations",
    "Total Page Citations",
    "Avg Domain Rank",
    "Avg Page Rank",
    "% of Prompt Runs Citing Domain",
    "% of Prompt Runs Citing Page",
    "% of Outputs Citing Domain",
    "% of Outputs Citing Page",
    "Avg Citations per Output (Domain)",
    "Avg Citations per Output (Page)",
    "Recent Domain Velocity",
    "First Seen Timestamp",
    "Predictability Score",
    "Page Last Seen Timestamp",
    "Topical Authority Score",
    "Days Since Last Seen",
    "Run Labels",
    "Execution IDs",
    "Run Dates",
]

SORT_OPTIONS = [
    ("Prompt Text (A-Z)", "Prompt Text"),
    ("Domain (A-Z)", "Domain"),
    ("Average Domain Rank", "Avg Domain Rank"),
    ("Average Page Rank", "Avg Page Rank"),
    ("Domain Citations", "Total Domain Citations"),
    ("Page Citations", "Total Page Citations"),
    ("Predictability Score", "Predictability Score"),
    ("Topical Authority Score", "Topical Authority Score"),
    ("Recent Domain Velocity", "Recent Domain Velocity"),
    ("% Prompt Runs (Domain)", "% of Prompt Runs Citing Domain"),
    ("% Prompt Runs (Page)", "% of Prompt Runs Citing Page"),
]

LAST_REPORT = pd.DataFrame(columns=REPORT_COLUMNS)
LAST_VIEW = pd.DataFrame(columns=REPORT_COLUMNS)
FULL_DOMAIN_REPORT = pd.DataFrame(columns=REPORT_COLUMNS)
FULL_PAGE_REPORT = pd.DataFrame(columns=REPORT_COLUMNS)


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


def compute_topical_authority_score(prompt_pct: float, avg_rank: float | None) -> float:
    if avg_rank is None or pd.isna(avg_rank):
        rank_component = 50.0
    else:
        rank_component = max(0.0, (10 - min(avg_rank, 10)) / 9 * 100)
    return (prompt_pct + rank_component) / 2


def first_non_null(series: pd.Series) -> object:
    if series is None or series.empty:
        return None
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[0]


def truncate_prompt_text(value: object, limit: int = 140) -> str:
    if not isinstance(value, str):
        return "" if value is None else str(value)
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def format_domain_link(domain: str) -> str:
    if not isinstance(domain, str) or not domain:
        return ""
    return f'<a href="https://{domain}" target="_blank">{domain}</a>'


def format_page_title_link(title: str, url: str) -> str:
    if not isinstance(url, str) or not url:
        return title or ""
    safe_title = title or derive_title_from_url(url)
    return f'<a href="{url}" target="_blank">{safe_title}</a>'


def format_url_link(url: str) -> str:
    if not isinstance(url, str) or not url:
        return ""
    return f'<a href="{url}" target="_blank">{url}</a>'


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


def build_prompt_insights(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        empty = pd.DataFrame(columns=REPORT_COLUMNS)
        return empty, empty

    working = df[df["role"] == AI_ROLE].copy()
    if working.empty:
        empty = pd.DataFrame(columns=REPORT_COLUMNS)
        return empty, empty
    working["query_or_topic"] = working["query_or_topic"].fillna("Unknown prompt")
    working["row_timestamp"] = pd.to_datetime(working["row_timestamp"], errors="coerce", utc=True)
    working["run_id"] = working.apply(derive_run_id, axis=1)
    working["prompt_run_id"] = working.apply(derive_prompt_run_id, axis=1)

    citations = working.dropna(subset=["citation_url"]).copy()
    if citations.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    citations["clean_url"] = citations["citation_url"].apply(clean_url)
    citations = citations[citations["clean_url"] != ""]
    if citations.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    citations["domain"] = citations["clean_url"].apply(extract_domain)
    citations = citations[citations["domain"] != ""]
    if citations.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    citations["citation_rank"] = pd.to_numeric(citations["citation_rank"], errors="coerce")
    citations["page_title"] = citations.apply(
        lambda row: row["citation_title"] if isinstance(row["citation_title"], str) and row["citation_title"].strip()
        else derive_title_from_url(row["clean_url"]),
        axis=1,
    )

    now_ts = pd.Timestamp.utcnow()
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    window_start = now_ts - pd.Timedelta(days=7)
    total_prompt_runs_all = max(working["prompt_run_id"].nunique(), 1)

    domain_records: list[dict[str, object]] = []
    page_records: list[dict[str, object]] = []
    prompt_groups = working.groupby("query_or_topic")

    for prompt, prompt_rows in prompt_groups:
        prompt_total_runs = prompt_rows["prompt_run_id"].nunique()
        prompt_total_outputs = prompt_rows["run_id"].nunique()
        prompt_citations = citations[citations["query_or_topic"] == prompt]
        if prompt_citations.empty:
            continue

        prompt_citing_runs = prompt_citations["prompt_run_id"].nunique()
        outputs_with_cites = prompt_citations["run_id"].nunique()
        prompt_last_seen = prompt_citations["row_timestamp"].max()

        scenario = first_non_null(prompt_rows["scenario"]) if "scenario" in prompt_rows.columns else None
        persona = first_non_null(prompt_rows["persona_profile"]) if "persona_profile" in prompt_rows.columns else None
        model = first_non_null(prompt_rows["model"]) if "model" in prompt_rows.columns else None
        run_labels = sorted(
            {
                str(label).strip()
                for label in prompt_rows.get("run_label", pd.Series(dtype=str)).dropna()
                if str(label).strip()
            }
        )
        execution_ids = sorted(
            {
                str(exe).strip()
                for exe in prompt_rows.get("execution_id", pd.Series(dtype=str)).dropna()
                if str(exe).strip()
            }
        )
        run_dates_set: set[str] = set()
        for ts in prompt_rows.get("row_timestamp", pd.Series(dtype="datetime64[ns]")).dropna():
            try:
                if getattr(ts, "tzinfo", None) is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                run_dates_set.add(ts.strftime("%Y-%m-%d"))
            except Exception:
                continue
        run_dates = sorted(run_dates_set)

        prompt_row_base = {
            "Prompt Text": prompt,
            "Scenario": scenario,
            "Persona": persona,
            "Model": model,
            "Total Prompt Runs": prompt_total_runs,
            "Unique Prompt Runs": prompt_citing_runs,
            "% of Total Prompt Runs": (prompt_total_runs / total_prompt_runs_all) * 100,
            "Total Outputs": prompt_total_outputs,
            "% of Outputs with Citations": (outputs_with_cites / max(prompt_total_outputs, 1)) * 100,
            "Prompt Last Seen": prompt_last_seen.isoformat() if pd.notna(prompt_last_seen) else None,
            "Run Labels": ", ".join(run_labels),
            "Execution IDs": ", ".join(execution_ids),
            "Run Dates": ", ".join(run_dates),
        }

        for domain, domain_rows in prompt_citations.groupby("domain"):
            domain_unique_pages = domain_rows["clean_url"].nunique()
            domain_total_cites = int(domain_rows.shape[0])
            domain_avg_rank = domain_rows["citation_rank"].mean() if not domain_rows["citation_rank"].dropna().empty else None
            domain_prompt_runs = domain_rows["prompt_run_id"].nunique()
            domain_outputs = domain_rows["run_id"].nunique()

            pct_prompt_runs_domain = (domain_prompt_runs / max(prompt_total_runs, 1)) * 100
            pct_outputs_domain = (domain_outputs / max(prompt_total_outputs, 1)) * 100
            avg_cites_per_output_domain = domain_total_cites / max(domain_outputs, 1)
            domain_top3_pct = (
                float((domain_rows["citation_rank"] <= 3).mean() * 100)
                if domain_rows["citation_rank"].notna().any()
                else 0.0
            )
            timestamps = domain_rows["row_timestamp"].dropna()
            recent_count = int((timestamps >= window_start).sum())
            recent_velocity = (recent_count / domain_total_cites) * 100 if domain_total_cites else 0.0
            predictability_score = (pct_outputs_domain + domain_top3_pct) / 2
            topical_authority_score = compute_topical_authority_score(pct_prompt_runs_domain, domain_avg_rank)
            domain_first_seen = timestamps.min()
            domain_last_seen = timestamps.max()
            domain_days_since_last = (now_ts - domain_last_seen).days if pd.notna(domain_last_seen) else None

            domain_base = {
                **prompt_row_base,
                "Domain": domain,
                "Unique Pages Cited": domain_unique_pages,
                "Total Domain Citations": domain_total_cites,
                "Avg Domain Rank": domain_avg_rank,
                "% of Prompt Runs Citing Domain": pct_prompt_runs_domain,
                "% of Outputs Citing Domain": pct_outputs_domain,
                "Avg Citations per Output (Domain)": avg_cites_per_output_domain,
                "Recent Domain Velocity": recent_velocity,
                "Predictability Score": predictability_score,
                "Topical Authority Score": topical_authority_score,
                "First Seen Timestamp": domain_first_seen.isoformat() if pd.notna(domain_first_seen) else None,
                "Page Last Seen Timestamp": domain_last_seen.isoformat() if pd.notna(domain_last_seen) else None,
                "Days Since Last Seen": domain_days_since_last,
            }

            domain_record = {
                **domain_base,
                "Page Title": None,
                "Full URL": None,
                "Total Page Citations": None,
                "Avg Page Rank": None,
                "% of Prompt Runs Citing Page": None,
                "% of Outputs Citing Page": None,
                "Avg Citations per Output (Page)": None,
            }
            domain_records.append(domain_record)

            for page_url, page_rows in domain_rows.groupby("clean_url"):
                page_total_cites = int(page_rows.shape[0])
                page_avg_rank = page_rows["citation_rank"].mean() if not page_rows["citation_rank"].dropna().empty else None
                page_prompt_runs = page_rows["prompt_run_id"].nunique()
                page_outputs = page_rows["run_id"].nunique()
                pct_prompt_runs_page = (page_prompt_runs / max(prompt_total_runs, 1)) * 100
                pct_outputs_page = (page_outputs / max(prompt_total_outputs, 1)) * 100
                avg_cites_per_output_page = page_total_cites / max(page_outputs, 1)
                page_first_seen = page_rows["row_timestamp"].min()
                page_last_seen = page_rows["row_timestamp"].max()
                days_since_last = (now_ts - page_last_seen).days if pd.notna(page_last_seen) else None
                page_title = page_rows["page_title"].dropna().iloc[0]

                record = {
                    **domain_base,
                    "Page Title": page_title,
                    "Full URL": page_url,
                    "Total Page Citations": page_total_cites,
                    "Avg Page Rank": page_avg_rank,
                    "% of Prompt Runs Citing Page": pct_prompt_runs_page,
                    "% of Outputs Citing Page": pct_outputs_page,
                    "Avg Citations per Output (Page)": avg_cites_per_output_page,
                    "First Seen Timestamp": page_first_seen.isoformat() if pd.notna(page_first_seen) else None,
                    "Page Last Seen Timestamp": page_last_seen.isoformat() if pd.notna(page_last_seen) else None,
                    "Days Since Last Seen": days_since_last,
                }
                page_records.append(record)

    if not page_records and not domain_records:
        empty = pd.DataFrame(columns=REPORT_COLUMNS)
        return empty, empty

    domain_df = pd.DataFrame(domain_records, columns=REPORT_COLUMNS) if domain_records else pd.DataFrame(columns=REPORT_COLUMNS)
    page_df = pd.DataFrame(page_records, columns=REPORT_COLUMNS) if page_records else pd.DataFrame(columns=REPORT_COLUMNS)
    return domain_df, page_df


def apply_custom_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    working = df.copy()
    if "citation_url" in working.columns:
        working["__domain"] = working["citation_url"].apply(
            lambda url: extract_domain(clean_url(url)) if isinstance(url, str) else ""
        )
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


def compute_summary(domain_df: pd.DataFrame, page_df: pd.DataFrame) -> dict[str, int | float]:
    if domain_df.empty and page_df.empty:
        return {
            "prompts": 0,
            "domains": 0,
            "pages": 0,
            "total_citations": 0,
        }
    return {
        "prompts": domain_df["Prompt Text"].nunique() if not domain_df.empty else page_df["Prompt Text"].nunique(),
        "domains": domain_df["Domain"].nunique() if not domain_df.empty else 0,
        "pages": page_df["Full URL"].nunique() if not page_df.empty else 0,
        "total_citations": int(page_df["Total Page Citations"].sum()) if not page_df.empty and "Total Page Citations" in page_df else 0,
    }


master_df = normalize_role_labels(refresh_master_df(force_rebuild=False))
FULL_DOMAIN_REPORT, FULL_PAGE_REPORT = build_prompt_insights(master_df)
filters = create_filter_panel(master_df)
if "role" in filters and isinstance(filters["role"], widgets.Widget):
    role_options = getattr(filters["role"], "options", [])
    option_values = [opt[1] if isinstance(opt, tuple) else opt for opt in role_options]
    if AI_ROLE in option_values:
        filters["role"].value = AI_ROLE
        filters["role"]._default = AI_ROLE  # type: ignore[attr-defined]
clear_filters_button = create_clear_filters_button(filters)

domain_dropdown = widgets.Dropdown(description="Domain:", options=_build_domain_options(master_df), value="All")
domain_search = widgets.Text(description="Domain contains:", placeholder="contains…")
page_search = widgets.Text(description="Page contains:", placeholder="url or slug…")
output_text_filter = widgets.Text(description="Output text:", placeholder="prompt/output contains…")
view_toggle = widgets.ToggleButtons(
    description="View:",
    options=[("Domain + Page", "with_pages"), ("Domain only", "domain_only")],
    value="with_pages",
)

DEFAULT_DISPLAY_COLUMNS = {
    "Prompt Text",
    "Domain",
    "Page Title",
    "Model",
    "Total Domain Citations",
    "Total Page Citations",
    "Avg Domain Rank",
    "Avg Page Rank",
    "Avg Citations per Output (Domain)",
    "Avg Citations per Output (Page)",
}

column_checkboxes = {
    col: widgets.Checkbox(description=col, value=(col in DEFAULT_DISPLAY_COLUMNS))
    for col in [
        "Prompt Text",
        "Scenario",
        "Persona",
        "Model",
        "Total Prompt Runs",
        "Unique Prompt Runs",
        "% of Total Prompt Runs",
        "Total Outputs",
        "% of Outputs with Citations",
        "Prompt Last Seen",
        "Domain",
        "Page Title",
        "Unique Pages Cited",
        "Full URL",
        "Total Domain Citations",
        "Total Page Citations",
        "Avg Domain Rank",
        "Avg Page Rank",
        "% of Prompt Runs Citing Domain",
        "% of Prompt Runs Citing Page",
        "% of Outputs Citing Domain",
        "% of Outputs Citing Page",
        "Avg Citations per Output (Domain)",
        "Avg Citations per Output (Page)",
        "Recent Domain Velocity",
        "First Seen Timestamp",
        "Predictability Score",
        "Page Last Seen Timestamp",
        "Topical Authority Score",
        "Days Since Last Seen",
    ]
}

column_picker_grid = widgets.GridBox(
    list(column_checkboxes.values()),
    layout=widgets.Layout(grid_template_columns="repeat(2, 50%)", grid_gap="4px 12px"),
)
column_picker_box = widgets.VBox([widgets.HTML("<b>Select columns to display:</b>"), column_picker_grid])
column_picker = widgets.Accordion(children=[column_picker_box])
column_picker.set_title(0, "+ Column picker")
column_picker.selected_index = None

sort_column = widgets.Dropdown(description="Sort by:", options=SORT_OPTIONS, value="Predictability Score")
sort_order = widgets.ToggleButtons(
    description="Order:",
    options=[("Desc", "desc"), ("Asc", "asc")],
    value="desc",
)
heading_html = widgets.HTML("<h3>Prompt-Level Insights Report</h3>")
refresh_button = widgets.Button(description="Refresh data", icon="refresh", button_style="primary")
rebuild_button = widgets.Button(description="Force rebuild", icon="repeat", button_style="danger")
export_table_button = widgets.Button(description="Export All", icon="table", button_style="warning")
export_view_button = widgets.Button(description="Export View", icon="eye", button_style="success")
export_highlights_button = widgets.Button(description="Export highlights", icon="star")
export_highlights_button.style.button_color = "#e0e0e0"

summary_output = widgets.Output()
table_header = widgets.HTML("<h4>Prompt / Domain / Page performance</h4>")
table_output = widgets.Output(layout=widgets.Layout(max_height="520px", overflow="auto"))
message_output = widgets.Output()
download_link_html = widgets.HTML()


def maybe_trigger_download(path: Path) -> None:
    if "google.colab" in sys.modules:
        try:
            from google.colab import files  # type: ignore

            files.download(str(path))
        except Exception:
            pass


def sort_report(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ascending = sort_order.value == "asc"
    return df.sort_values(sort_column.value, ascending=ascending)


def update_display(_=None):
    global LAST_REPORT, LAST_VIEW

    filtered = apply_filters(master_df, filters)
    filtered = apply_custom_filters(filtered)
    domain_df, page_df = build_prompt_insights(filtered)
    summary = compute_summary(domain_df, page_df)

    with summary_output:
        summary_output.clear_output()
        display(HTML("<h4>Summary metrics</h4>"))
        if domain_df.empty and page_df.empty:
            display(HTML("<p>No prompts with citations in the current view.</p>"))
            display(HTML("<br>"))
        else:
            summary_df = pd.DataFrame(
                [
                    {
                        "Prompts w/ Citations": f"{summary['prompts']:,}",
                        "Domains": f"{summary['domains']:,}",
                        "Pages": f"{summary['pages']:,}",
                        "Total Page Citations": f"{summary['total_citations']:,}",
                    }
                ]
            )
            display(HTML(summary_df.to_html(index=False, escape=False)))
        display(HTML("<br>"))

    with table_output:
        table_output.clear_output()
        active_df = domain_df if view_toggle.value == "domain_only" else page_df
        if active_df.empty:
            LAST_REPORT = pd.DataFrame(columns=REPORT_COLUMNS)
            LAST_VIEW = pd.DataFrame(columns=REPORT_COLUMNS)
            return

        sorted_df = sort_report(active_df)
        LAST_REPORT = sorted_df
        rows = filters["rows"].value
        display_df = sorted_df.head(rows).copy()
        LAST_VIEW = display_df.copy()

        if "Prompt Text" in display_df.columns:
            display_df["Prompt Text"] = display_df["Prompt Text"].apply(truncate_prompt_text)

        if view_toggle.value == "domain_only":
            for col in [
                "Page Title",
                "Full URL",
                "Total Page Citations",
                "Avg Page Rank",
                "% of Prompt Runs Citing Page",
                "% of Outputs Citing Page",
                "Avg Citations per Output (Page)",
                "First Seen Timestamp",
                "Page Last Seen Timestamp",
                "Days Since Last Seen",
            ]:
                if col in display_df.columns:
                    display_df[col] = "---"

        display_df["Domain"] = display_df["Domain"].apply(format_domain_link)
        display_df["Page Title"] = display_df.apply(
            lambda row: format_page_title_link(row["Page Title"], row["Full URL"])
            if isinstance(row["Page Title"], str) and row["Page Title"] not in ("", "---") and isinstance(row["Full URL"], str) and row["Full URL"] not in ("", "---")
            else (row["Page Title"] if isinstance(row["Page Title"], str) else ""),
            axis=1,
        )
        if "Full URL" in display_df.columns:
            display_df["Full URL"] = display_df["Full URL"].apply(
                lambda url: format_url_link(url) if isinstance(url, str) and url not in ("", "---") else url
            )

        percent_columns = [
            "% of Total Prompt Runs",
            "% of Outputs with Citations",
            "% of Prompt Runs Citing Domain",
            "% of Prompt Runs Citing Page",
            "% of Outputs Citing Domain",
            "% of Outputs Citing Page",
            "Recent Domain Velocity",
            "Predictability Score",
            "Topical Authority Score",
        ]
        for col in percent_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda v: f"{format_numeric(v, 2)}%" if pd.notna(v) and v != "---" else v)

        numeric_columns = [
            "Avg Domain Rank",
            "Avg Page Rank",
            "Avg Citations per Output (Domain)",
            "Avg Citations per Output (Page)",
        ]
        for col in numeric_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda v: format_numeric(v, 2) if pd.notna(v) and v != "---" else v)

        timestamp_columns = [
            "Prompt Last Seen",
            "First Seen Timestamp",
            "Page Last Seen Timestamp",
        ]
        for col in timestamp_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(
                    lambda v: format_timestamp_short(v) if isinstance(v, str) and v != "---" else v
                )

        selected_columns = [col for col, checkbox in column_checkboxes.items() if checkbox.value]
        if not selected_columns:
            selected_columns = [col for col in REPORT_COLUMNS if col in DEFAULT_DISPLAY_COLUMNS] or REPORT_COLUMNS

        valid_columns = [col for col in selected_columns if col in display_df.columns]
        table_html = display_df[valid_columns].to_html(index=False, escape=False, classes="prompt-insights-table")
        styled_html = """
        <style>
        .prompt-insights-table thead th {
            position: sticky;
            top: 0;
            background: #f6f6f6;
            z-index: 1;
        }
        .prompt-insights-table tbody td {
            vertical-align: top;
        }
        </style>
        """ + table_html
        display(HTML(styled_html))


def refresh_domain_options_widget() -> None:
    options = _build_domain_options(master_df)
    current = domain_dropdown.value if domain_dropdown.value in options else "All"
    domain_dropdown.options = options
    domain_dropdown.value = current


def _refresh_full_reports():
    global FULL_DOMAIN_REPORT, FULL_PAGE_REPORT
    FULL_DOMAIN_REPORT, FULL_PAGE_REPORT = build_prompt_insights(master_df)


def handle_refresh(_):
    global master_df
    master_df = normalize_role_labels(refresh_master_df(force_rebuild=False))
    _refresh_full_reports()
    refresh_domain_options_widget()
    update_display()


def handle_rebuild(_):
    global master_df
    master_df = normalize_role_labels(refresh_master_df(force_rebuild=True))
    _refresh_full_reports()
    refresh_domain_options_widget()
    update_display()


def _current_full_report() -> pd.DataFrame:
    return FULL_DOMAIN_REPORT if view_toggle.value == "domain_only" else FULL_PAGE_REPORT


def handle_export_table(_):
    full_df = _current_full_report()
    if full_df.empty:
        with message_output:
            message_output.clear_output()
            print("⚠️ Nothing to export yet.")
        download_link_html.value = ""
        return
    export_df = sort_report(full_df)
    filename = "prompt_insights_domain_all" if view_toggle.value == "domain_only" else "prompt_insights_page_all"
    path = export_dataframe(export_df, filename)
    maybe_trigger_download(path)
    with message_output:
        message_output.clear_output()
        print("📄 Exported full dataset.")
    download_link_html.value = f'<a href="{path}" target="_blank">Download CSV</a>'


def handle_export_view(_):
    if LAST_REPORT.empty:
        with message_output:
            message_output.clear_output()
            print("⚠️ No filtered rows to export yet.")
        download_link_html.value = ""
        return
    filename = "prompt_insights_domain_view" if view_toggle.value == "domain_only" else "prompt_insights_page_view"
    path = export_dataframe(LAST_REPORT, filename)
    maybe_trigger_download(path)
    with message_output:
        message_output.clear_output()
        print("📄 Exported current view.")
    download_link_html.value = f'<a href="{path}" target="_blank">Download CSV</a>'


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
            df.nlargest(5, "Predictability Score").assign(Highlight="predictability_top5"),
            df.nlargest(5, "Topical Authority Score").assign(Highlight="authority_top5"),
            df.nlargest(5, "Recent Domain Velocity").assign(Highlight="velocity_top5"),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["Prompt Text", "Domain", "Full URL", "Highlight"])
    path = export_dataframe(highlights, "prompt_insights_highlights")
    maybe_trigger_download(path)
    with message_output:
        message_output.clear_output()
        print("📄 Highlight tables exported.")
    download_link_html.value = f'<a href="{path}" target="_blank">Download CSV</a>'


refresh_button.on_click(handle_refresh)
rebuild_button.on_click(handle_rebuild)
export_table_button.on_click(handle_export_table)
export_view_button.on_click(handle_export_view)
export_highlights_button.on_click(handle_export_highlights)

for widget in list(filters.values()) + [sort_column, sort_order, view_toggle]:
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

filters["query_text"].description = "Prompt search:"
filters["message_text"].description = "Message search:"

for control in (
    domain_dropdown,
    domain_search,
    page_search,
    output_text_filter,
    filters["query_text"],
    filters["message_text"],
):
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
            [sort_column, sort_order, view_toggle],
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
