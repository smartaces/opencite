# cell_08_reports.py
"""
Reports - Generate domain, page, and prompt-level analysis.

This cell provides:
1. Domain Report - Aggregate citations by domain
2. Page Report - Aggregate citations by URL
3. Prompt Report - Analyze citation patterns by prompt
"""

import json
import os
from pathlib import Path
from urllib.parse import urlparse

import ipywidgets as widgets
import pandas as pd
from IPython.display import display, clear_output


def _ensure_paths():
    if 'PATHS' in globals() and globals()['PATHS']:
        return {k: Path(v) for k, v in globals()['PATHS'].items()}

    config_path = Path(os.environ.get('WORKSPACE_CONFIG', ''))
    if not config_path.is_file():
        raise RuntimeError("Workspace not configured. Run the workspace setup cell first.")

    with open(config_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return {k: Path(v) for k, v in data['paths'].items()}


PATHS = _ensure_paths()
CSV_OUTPUT_DIR = Path(PATHS['csv_output'])
REPORT_CACHE_DIR = CSV_OUTPUT_DIR / "report_cache"
REPORT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_detail_csvs() -> pd.DataFrame:
    """Load and combine all detail CSVs from the csv_output directory."""
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_files = sorted(CSV_OUTPUT_DIR.glob("*_detail_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not detail_files:
        return pd.DataFrame()

    dfs = []
    for file_path in detail_files:
        try:
            df = pd.read_csv(file_path)
            df["_source_file"] = file_path.name
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def get_citation_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract citation data from the master dataframe."""
    if df.empty:
        return pd.DataFrame()

    # Filter to rows with citations
    if "citation_url" not in df.columns:
        return pd.DataFrame()

    citations = df[df["citation_url"].notna()].copy()

    # Extract domain from URL
    if "domain" not in citations.columns:
        citations["domain"] = citations["citation_url"].apply(
            lambda x: urlparse(str(x)).netloc.replace("www.", "") if pd.notna(x) else ""
        )

    return citations


# ============================================================================
# DOMAIN REPORT
# ============================================================================

def generate_domain_report(df: pd.DataFrame, filter_provider: str = "All", filter_model: str = "All") -> pd.DataFrame:
    """Generate domain-level aggregation report."""
    citations = get_citation_data(df)

    if citations.empty:
        return pd.DataFrame(columns=["Domain", "Citations", "Unique URLs", "Prompts", "Avg Rank"])

    # Apply filters
    if filter_provider != "All" and "provider" in citations.columns:
        citations = citations[citations["provider"] == filter_provider]
    if filter_model != "All" and "model" in citations.columns:
        citations = citations[citations["model"] == filter_model]

    if citations.empty:
        return pd.DataFrame(columns=["Domain", "Citations", "Unique URLs", "Prompts", "Avg Rank"])

    # Aggregate by domain
    agg = citations.groupby("domain").agg({
        "citation_url": "count",
        "citation_rank": "mean",
        "query_or_topic": lambda x: x.nunique() if "query_or_topic" in citations.columns else 0,
    }).reset_index()

    agg.columns = ["Domain", "Citations", "Avg Rank", "Prompts"]

    # Get unique URLs per domain
    url_counts = citations.groupby("domain")["citation_url"].nunique().reset_index()
    url_counts.columns = ["Domain", "Unique URLs"]
    agg = agg.merge(url_counts, on="Domain", how="left")

    # Reorder and sort
    agg = agg[["Domain", "Citations", "Unique URLs", "Prompts", "Avg Rank"]]
    agg = agg.sort_values("Citations", ascending=False).reset_index(drop=True)
    agg["Avg Rank"] = agg["Avg Rank"].round(2)

    return agg


# ============================================================================
# PAGE REPORT
# ============================================================================

def generate_page_report(df: pd.DataFrame, filter_provider: str = "All", filter_model: str = "All") -> pd.DataFrame:
    """Generate page/URL-level aggregation report."""
    citations = get_citation_data(df)

    if citations.empty:
        return pd.DataFrame(columns=["URL", "Title", "Domain", "Citations", "Avg Rank", "Prompts"])

    # Apply filters
    if filter_provider != "All" and "provider" in citations.columns:
        citations = citations[citations["provider"] == filter_provider]
    if filter_model != "All" and "model" in citations.columns:
        citations = citations[citations["model"] == filter_model]

    if citations.empty:
        return pd.DataFrame(columns=["URL", "Title", "Domain", "Citations", "Avg Rank", "Prompts"])

    # Aggregate by URL
    agg = citations.groupby("citation_url").agg({
        "citation_title": "first",
        "domain": "first",
        "citation_rank": ["count", "mean"],
        "query_or_topic": lambda x: x.nunique() if "query_or_topic" in citations.columns else 0,
    }).reset_index()

    agg.columns = ["URL", "Title", "Domain", "Citations", "Avg Rank", "Prompts"]
    agg = agg.sort_values("Citations", ascending=False).reset_index(drop=True)
    agg["Avg Rank"] = agg["Avg Rank"].round(2)

    # Truncate title for display
    agg["Title"] = agg["Title"].apply(lambda x: str(x)[:60] + "..." if pd.notna(x) and len(str(x)) > 60 else x)

    return agg


# ============================================================================
# PROMPT REPORT
# ============================================================================

def generate_prompt_report(df: pd.DataFrame, filter_provider: str = "All", filter_model: str = "All") -> pd.DataFrame:
    """Generate prompt-level aggregation report."""
    if df.empty or "query_or_topic" not in df.columns:
        return pd.DataFrame(columns=["Prompt", "Runs", "Total Citations", "Unique URLs", "Unique Domains", "Top Domain"])

    # Filter to AI System responses (where citations appear)
    ai_data = df[df["role"] == "AI System"].copy() if "role" in df.columns else df.copy()

    # Apply filters
    if filter_provider != "All" and "provider" in ai_data.columns:
        ai_data = ai_data[ai_data["provider"] == filter_provider]
    if filter_model != "All" and "model" in ai_data.columns:
        ai_data = ai_data[ai_data["model"] == filter_model]

    if ai_data.empty:
        return pd.DataFrame(columns=["Prompt", "Runs", "Total Citations", "Unique URLs", "Unique Domains", "Top Domain"])

    results = []
    for prompt, group in ai_data.groupby("query_or_topic"):
        runs = group["execution_id"].nunique() if "execution_id" in group.columns else 1

        citations_data = group[group["citation_url"].notna()] if "citation_url" in group.columns else pd.DataFrame()
        total_citations = len(citations_data)
        unique_urls = citations_data["citation_url"].nunique() if not citations_data.empty else 0
        unique_domains = citations_data["domain"].nunique() if not citations_data.empty and "domain" in citations_data.columns else 0

        top_domain = ""
        if not citations_data.empty and "domain" in citations_data.columns:
            domain_counts = citations_data["domain"].value_counts()
            if len(domain_counts) > 0:
                top_domain = domain_counts.index[0]

        prompt_display = str(prompt)[:60] + "..." if len(str(prompt)) > 60 else str(prompt)

        results.append({
            "Prompt": prompt_display,
            "Full Prompt": prompt,
            "Runs": runs,
            "Total Citations": total_citations,
            "Unique URLs": unique_urls,
            "Unique Domains": unique_domains,
            "Top Domain": top_domain,
        })

    report = pd.DataFrame(results)
    report = report.sort_values("Total Citations", ascending=False).reset_index(drop=True)

    return report


# ============================================================================
# LOAD DATA
# ============================================================================

_master_df = load_all_detail_csvs()


def get_unique_providers(df: pd.DataFrame) -> list[str]:
    if df.empty or "provider" not in df.columns:
        return []
    return sorted(df["provider"].dropna().unique().tolist())


def get_unique_models(df: pd.DataFrame) -> list[str]:
    if df.empty or "model" not in df.columns:
        return []
    return sorted(df["model"].dropna().unique().tolist())


# ============================================================================
# WIDGETS
# ============================================================================

STYLE = {'description_width': '80px'}
DROPDOWN_LAYOUT = widgets.Layout(width='250px')

report_type_dropdown = widgets.Dropdown(
    options=[
        ("Domain Report", "domain"),
        ("Page Report", "page"),
        ("Prompt Report", "prompt"),
    ],
    value="domain",
    description="Report:",
    style=STYLE,
    layout=DROPDOWN_LAYOUT,
)

provider_filter = widgets.Dropdown(
    options=["All"] + get_unique_providers(_master_df),
    value="All",
    description="Provider:",
    style=STYLE,
    layout=DROPDOWN_LAYOUT,
)

model_filter = widgets.Dropdown(
    options=["All"] + get_unique_models(_master_df),
    value="All",
    description="Model:",
    style=STYLE,
    layout=DROPDOWN_LAYOUT,
)

generate_button = widgets.Button(
    description="Generate Report",
    button_style="primary",
    icon="table",
    layout=widgets.Layout(width='150px'),
)

export_button = widgets.Button(
    description="Export CSV",
    button_style="success",
    icon="download",
    layout=widgets.Layout(width='120px'),
    disabled=True,
)

refresh_button = widgets.Button(
    description="Refresh Data",
    icon="refresh",
    layout=widgets.Layout(width='120px'),
)

report_output = widgets.Output()
status_output = widgets.Output()

_current_report = {"df": None, "type": None}


# ============================================================================
# EVENT HANDLERS
# ============================================================================

def refresh_data(_=None):
    global _master_df
    _master_df = load_all_detail_csvs()
    provider_filter.options = ["All"] + get_unique_providers(_master_df)
    model_filter.options = ["All"] + get_unique_models(_master_df)
    with status_output:
        clear_output()
        print(f"Data refreshed. {len(_master_df)} rows loaded.")


def generate_report(_=None):
    global _current_report

    with report_output:
        clear_output()

    with status_output:
        clear_output()

    if _master_df.empty:
        with report_output:
            print("No data available. Run some batches first.")
        export_button.disabled = True
        return

    report_type = report_type_dropdown.value
    provider = provider_filter.value
    model = model_filter.value

    with status_output:
        print(f"Generating {report_type} report...")

    if report_type == "domain":
        report_df = generate_domain_report(_master_df, provider, model)
    elif report_type == "page":
        report_df = generate_page_report(_master_df, provider, model)
    else:
        report_df = generate_prompt_report(_master_df, provider, model)

    _current_report["df"] = report_df
    _current_report["type"] = report_type

    with report_output:
        if report_df.empty:
            print("No data matches the current filters.")
            export_button.disabled = True
        else:
            # Hide internal columns for display
            display_cols = [c for c in report_df.columns if not c.startswith("Full")]
            display(report_df[display_cols])
            export_button.disabled = False

    with status_output:
        clear_output()
        if not report_df.empty:
            print(f"Generated {len(report_df)} rows")


def export_report(_=None):
    if _current_report["df"] is None or _current_report["df"].empty:
        with status_output:
            clear_output()
            print("No report to export.")
        return

    report_type = _current_report["type"]
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{report_type}_report_{timestamp}.csv"
    filepath = REPORT_CACHE_DIR / filename

    _current_report["df"].to_csv(filepath, index=False)

    with status_output:
        clear_output()
        print(f"Exported to: {filepath}")


# ============================================================================
# EVENT BINDINGS
# ============================================================================

generate_button.on_click(generate_report)
export_button.on_click(export_report)
refresh_button.on_click(refresh_data)


# ============================================================================
# LAYOUT
# ============================================================================

controls = widgets.VBox([
    widgets.HTML("<h3>Reports</h3>"),
    widgets.HTML("<p>Generate aggregated reports from your batch run data.</p>"),

    widgets.HTML("<hr style='margin: 8px 0;'>"),
    widgets.HTML("<b>Report Options</b>"),
    widgets.HBox([report_type_dropdown, provider_filter, model_filter], layout=widgets.Layout(gap='12px')),
    widgets.HBox([generate_button, export_button, refresh_button], layout=widgets.Layout(gap='8px')),
    status_output,

    widgets.HTML("<hr style='margin: 12px 0;'>"),
    widgets.HTML("<b>Report Results</b>"),
    report_output,
])

display(controls)

# Show initial stats
with status_output:
    if _master_df.empty:
        print("No data available. Run some batches first, then click 'Refresh Data'.")
    else:
        citations = get_citation_data(_master_df)
        print(f"Data loaded: {len(_master_df)} rows, {len(citations)} citations")
        print("Select a report type and click 'Generate Report'.")
