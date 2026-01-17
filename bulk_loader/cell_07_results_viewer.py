# @title Results Viewer - Browse and inspect batch run outputs

import json
import os
from pathlib import Path

import ipywidgets as widgets
import pandas as pd
from IPython.display import display, clear_output, HTML


def _ensure_paths():
    if 'PATHS' in globals():
        return {k: Path(v) for k, v in PATHS.items()}

    config_path = Path(os.environ.get('WORKSPACE_CONFIG', ''))
    if not config_path.is_file():
        raise RuntimeError("Workspace not configured. Run the workspace setup cell first.")

    with open(config_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return {k: Path(v) for k, v in data['paths'].items()}


PATHS = _ensure_paths()
CSV_OUTPUT_DIR = Path(PATHS['csv_output'])


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


def get_unique_prompts(df: pd.DataFrame) -> list[str]:
    """Get unique prompts from the dataframe."""
    if df.empty or "query_or_topic" not in df.columns:
        return []
    prompts = df["query_or_topic"].dropna().unique().tolist()
    return sorted(set(str(p) for p in prompts if str(p).strip()))


def get_unique_run_labels(df: pd.DataFrame) -> list[str]:
    """Get unique run labels from the dataframe."""
    if df.empty or "run_label" not in df.columns:
        return []
    labels = df["run_label"].dropna().unique().tolist()
    return sorted(set(str(l) for l in labels if str(l).strip()))


def get_unique_executions(df: pd.DataFrame) -> list[str]:
    """Get unique execution IDs from the dataframe."""
    if df.empty or "execution_id" not in df.columns:
        return []
    execs = df["execution_id"].dropna().unique().tolist()
    return sorted(set(str(e) for e in execs if str(e).strip()), reverse=True)


def filter_dataframe(
    df: pd.DataFrame,
    prompt_filter: str = "All",
    run_label_filter: str = "All",
    prompt_keyword: str = "",
    response_keyword: str = "",
) -> pd.DataFrame:
    """Apply filters to the dataframe."""
    if df.empty:
        return df

    filtered = df.copy()

    # Filter by prompt
    if prompt_filter != "All" and "query_or_topic" in filtered.columns:
        filtered = filtered[filtered["query_or_topic"] == prompt_filter]

    # Filter by run label
    if run_label_filter != "All" and "run_label" in filtered.columns:
        filtered = filtered[filtered["run_label"] == run_label_filter]

    # Filter by keyword in prompt
    if prompt_keyword.strip() and "query_or_topic" in filtered.columns:
        keyword = prompt_keyword.strip().lower()
        filtered = filtered[filtered["query_or_topic"].astype(str).str.lower().str.contains(keyword, na=False)]

    # Filter by keyword in response
    if response_keyword.strip() and "message_text" in filtered.columns:
        keyword = response_keyword.strip().lower()
        filtered = filtered[filtered["message_text"].astype(str).str.lower().str.contains(keyword, na=False)]

    return filtered


def build_runs_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary table of unique runs."""
    if df.empty or "execution_id" not in df.columns:
        return pd.DataFrame(columns=["Execution ID", "Prompt", "Run Label", "Turns", "Citations", "Timestamp"])

    runs = []
    for exec_id, group in df.groupby("execution_id"):
        prompt = group["query_or_topic"].iloc[0] if "query_or_topic" in group.columns else "Unknown"
        run_label = group["run_label"].iloc[0] if "run_label" in group.columns else ""

        # Count turns (unique turn numbers)
        if "turn_or_run" in group.columns:
            turns = group["turn_or_run"].nunique()
        else:
            turns = 1

        # Count citations
        if "citation_url" in group.columns:
            citations = group["citation_url"].dropna().nunique()
        else:
            citations = 0

        # Get timestamp
        if "row_timestamp" in group.columns:
            ts = pd.to_datetime(group["row_timestamp"], errors="coerce").min()
            timestamp = ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else ""
        else:
            timestamp = ""

        # Truncate prompt for display
        prompt_display = str(prompt)[:60] + "..." if len(str(prompt)) > 60 else str(prompt)

        runs.append({
            "Execution ID": exec_id,
            "Prompt": prompt_display,
            "Full Prompt": prompt,
            "Run Label": run_label if pd.notna(run_label) else "",
            "Turns": turns,
            "Citations": citations,
            "Timestamp": timestamp,
        })

    runs_df = pd.DataFrame(runs)

    # Sort by timestamp descending (most recent first)
    if not runs_df.empty and "Timestamp" in runs_df.columns:
        runs_df = runs_df.sort_values("Timestamp", ascending=False).reset_index(drop=True)

    return runs_df


def get_run_details(df: pd.DataFrame, execution_id: str) -> pd.DataFrame:
    """Get all rows for a specific execution ID, sorted by turn."""
    if df.empty or "execution_id" not in df.columns:
        return pd.DataFrame()

    run_df = df[df["execution_id"] == execution_id].copy()

    if "turn_or_run" in run_df.columns:
        run_df = run_df.sort_values("turn_or_run")

    return run_df


def format_citation_link(url: str, title: str = "") -> str:
    """Format a citation as a clickable HTML link."""
    if not url or pd.isna(url):
        return ""
    display_text = title if title and not pd.isna(title) else url
    # Truncate display text if too long
    if len(display_text) > 80:
        display_text = display_text[:77] + "..."
    return f'<a href="{url}" target="_blank">{display_text}</a>'


def render_run_detail(df: pd.DataFrame, execution_id: str) -> str:
    """Render the full conversation for a run as HTML."""
    run_df = get_run_details(df, execution_id)

    if run_df.empty:
        return "<p>No data found for this run.</p>"

    # Get run metadata
    prompt = run_df["query_or_topic"].iloc[0] if "query_or_topic" in run_df.columns else "Unknown"
    run_label = run_df["run_label"].iloc[0] if "run_label" in run_df.columns else ""

    html_parts = []

    # Header
    html_parts.append(f"""
    <div style="background: #f5f5f5; padding: 12px; border-radius: 8px; margin-bottom: 16px;">
        <strong>Execution ID:</strong> {execution_id}<br>
        <strong>Run Label:</strong> {run_label if pd.notna(run_label) else '(none)'}<br>
        <strong>Prompt:</strong> {prompt}
    </div>
    """)

    # Group by turn
    if "turn_or_run" in run_df.columns:
        turns = sorted(run_df["turn_or_run"].dropna().unique())
    else:
        turns = [1]

    for turn in turns:
        if "turn_or_run" in run_df.columns:
            turn_df = run_df[run_df["turn_or_run"] == turn]
        else:
            turn_df = run_df

        html_parts.append(f"""
        <div style="border-left: 3px solid #2196F3; padding-left: 12px; margin-bottom: 16px;">
            <h4 style="margin: 0 0 8px 0; color: #2196F3;">Turn {int(turn)}</h4>
        """)

        # Persona message (user query)
        persona_rows = turn_df[turn_df["role"] == "persona"] if "role" in turn_df.columns else pd.DataFrame()
        if not persona_rows.empty and "message_text" in persona_rows.columns:
            persona_msg = persona_rows["message_text"].iloc[0]
            html_parts.append(f"""
            <div style="background: #e3f2fd; padding: 10px; border-radius: 6px; margin-bottom: 8px;">
                <strong>🗣️ User/Persona:</strong><br>
                <p style="margin: 4px 0 0 0; white-space: pre-wrap;">{persona_msg}</p>
            </div>
            """)

        # AI response
        ai_rows = turn_df[turn_df["role"] == "AI System"] if "role" in turn_df.columns else turn_df
        if not ai_rows.empty and "message_text" in ai_rows.columns:
            ai_msg = ai_rows["message_text"].iloc[0]
            html_parts.append(f"""
            <div style="background: #f5f5f5; padding: 10px; border-radius: 6px; margin-bottom: 8px;">
                <strong>🤖 AI Response:</strong><br>
                <p style="margin: 4px 0 0 0; white-space: pre-wrap;">{ai_msg}</p>
            </div>
            """)

            # Citations for this turn
            citations = ai_rows[ai_rows["citation_url"].notna()] if "citation_url" in ai_rows.columns else pd.DataFrame()
            if not citations.empty:
                html_parts.append("""
                <div style="background: #fff3e0; padding: 10px; border-radius: 6px; margin-bottom: 8px;">
                    <strong>🔗 Citations:</strong><br>
                    <ul style="margin: 4px 0 0 0; padding-left: 20px;">
                """)

                seen_urls = set()
                for _, row in citations.iterrows():
                    url = row.get("citation_url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        title = row.get("citation_title", "")
                        rank = row.get("citation_rank", "")
                        rank_str = f" (rank {int(rank)})" if pd.notna(rank) else ""
                        link = format_citation_link(url, title)
                        html_parts.append(f"<li>{link}{rank_str}</li>")

                html_parts.append("</ul></div>")

        html_parts.append("</div>")  # Close turn div

    return "".join(html_parts)


# ============================================================================
# LOAD INITIAL DATA
# ============================================================================

_master_df = load_all_detail_csvs()
_runs_summary = build_runs_summary(_master_df)


# ============================================================================
# WIDGETS
# ============================================================================

STYLE = {'description_width': '120px'}
DROPDOWN_LAYOUT = widgets.Layout(width='300px')
TEXT_LAYOUT = widgets.Layout(width='300px')

# Filter widgets
prompt_dropdown = widgets.Dropdown(
    options=["All"] + get_unique_prompts(_master_df),
    value="All",
    description="Prompt:",
    style=STYLE,
    layout=DROPDOWN_LAYOUT,
)

run_label_dropdown = widgets.Dropdown(
    options=["All"] + get_unique_run_labels(_master_df),
    value="All",
    description="Run label:",
    style=STYLE,
    layout=DROPDOWN_LAYOUT,
)

prompt_keyword_input = widgets.Text(
    value="",
    description="Prompt contains:",
    placeholder="keyword...",
    style=STYLE,
    layout=TEXT_LAYOUT,
)

response_keyword_input = widgets.Text(
    value="",
    description="Response contains:",
    placeholder="keyword...",
    style=STYLE,
    layout=TEXT_LAYOUT,
)

refresh_button = widgets.Button(
    description="Refresh Data",
    icon="refresh",
    button_style="primary",
    layout=widgets.Layout(width='120px'),
)

# Results table output
results_output = widgets.Output()

# Run detail output
detail_output = widgets.Output()

# State tracking
_selected_execution_id = {"value": None}


# ============================================================================
# EVENT HANDLERS
# ============================================================================

def refresh_data(_=None):
    """Reload all data from CSV files."""
    global _master_df, _runs_summary

    _master_df = load_all_detail_csvs()
    _runs_summary = build_runs_summary(_master_df)

    # Update dropdown options
    prompt_dropdown.options = ["All"] + get_unique_prompts(_master_df)
    run_label_dropdown.options = ["All"] + get_unique_run_labels(_master_df)

    update_results_table()

    with detail_output:
        clear_output()
        print("ℹ️ Data refreshed. Select a run from the table above to view details.")


def update_results_table(_=None):
    """Update the results table based on current filters."""
    global _runs_summary

    # Apply filters to master data
    filtered_df = filter_dataframe(
        _master_df,
        prompt_filter=prompt_dropdown.value,
        run_label_filter=run_label_dropdown.value,
        prompt_keyword=prompt_keyword_input.value,
        response_keyword=response_keyword_input.value,
    )

    # Build summary of matching runs
    _runs_summary = build_runs_summary(filtered_df)

    with results_output:
        clear_output()

        if _runs_summary.empty:
            print("No runs match the current filters.")
            return

        print(f"Found {len(_runs_summary)} matching run(s). Click a row to view details.\n")

        # Display table without Full Prompt column (used internally)
        display_cols = ["Execution ID", "Prompt", "Run Label", "Turns", "Citations", "Timestamp"]
        display_df = _runs_summary[[c for c in display_cols if c in _runs_summary.columns]].copy()

        # Create clickable buttons for each row
        for idx, row in display_df.iterrows():
            exec_id = _runs_summary.loc[idx, "Execution ID"]
            btn = widgets.Button(
                description=f"View",
                button_style="info",
                layout=widgets.Layout(width='60px', height='24px'),
            )
            btn.on_click(lambda b, eid=exec_id: load_run_detail(eid))

            row_html = f"""
            <div style="display: flex; align-items: center; padding: 6px 0; border-bottom: 1px solid #eee;">
                <span style="width: 180px; font-family: monospace; font-size: 12px;">{row['Execution ID']}</span>
                <span style="width: 280px; overflow: hidden; text-overflow: ellipsis;">{row['Prompt']}</span>
                <span style="width: 120px;">{row['Run Label']}</span>
                <span style="width: 60px; text-align: center;">{row['Turns']}</span>
                <span style="width: 60px; text-align: center;">{row['Citations']}</span>
                <span style="width: 120px;">{row['Timestamp']}</span>
            </div>
            """
            display(widgets.HBox([
                widgets.HTML(row_html),
                btn,
            ]))


def load_run_detail(execution_id: str):
    """Load and display the full detail for a specific run."""
    _selected_execution_id["value"] = execution_id

    with detail_output:
        clear_output()
        html_content = render_run_detail(_master_df, execution_id)
        display(HTML(html_content))


# ============================================================================
# EVENT BINDINGS
# ============================================================================

refresh_button.on_click(refresh_data)
prompt_dropdown.observe(update_results_table, names="value")
run_label_dropdown.observe(update_results_table, names="value")
prompt_keyword_input.observe(update_results_table, names="value")
response_keyword_input.observe(update_results_table, names="value")


# ============================================================================
# LAYOUT
# ============================================================================

controls = widgets.VBox([
    widgets.HTML("<h3>Results Viewer</h3>"),
    widgets.HTML("<p>Browse and inspect your batch run outputs. Use filters to narrow down, then click a run to view the full conversation.</p>"),

    widgets.HTML("<hr style='margin: 8px 0;'>"),
    widgets.HTML("<b>Filters</b>"),
    widgets.HBox([prompt_dropdown, run_label_dropdown], layout=widgets.Layout(gap='12px')),
    widgets.HBox([prompt_keyword_input, response_keyword_input], layout=widgets.Layout(gap='12px')),
    refresh_button,

    widgets.HTML("<hr style='margin: 12px 0;'>"),
    widgets.HTML("<b>Matching Runs</b>"),
    results_output,

    widgets.HTML("<hr style='margin: 12px 0;'>"),
    widgets.HTML("<b>Run Details</b>"),
    detail_output,
])

display(controls)

# Initial load
update_results_table()
with detail_output:
    print("ℹ️ Select a run from the table above to view the full conversation.")