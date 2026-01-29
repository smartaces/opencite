# =============================================================================
# CELL 8: ANALYSE CITATIONS (REPORTS)
# =============================================================================
# Generate domain, page, or prompt-level reports from your batch results.
# =============================================================================

import json
import os
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display, clear_output


def _get_scripts_path() -> Path:
    """Get the scripts folder path from workspace config."""
    if 'PATHS' in globals():
        return Path(PATHS['scripts'])

    config_path = Path(os.environ.get('WORKSPACE_CONFIG', ''))
    if not config_path.is_file():
        raise RuntimeError("Workspace not configured. Run the workspace setup cell first.")

    with open(config_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return Path(data['paths']['scripts'])


scripts_path = _get_scripts_path()

# Report options
REPORT_OPTIONS = [
    ("Domain Report - Aggregate citations by domain", "domain"),
    ("Page Report - Aggregate citations by URL", "page"),
    ("Prompt Report - Analyse citations by prompt", "prompt"),
]

# UI Components
report_dropdown = widgets.Dropdown(
    options=REPORT_OPTIONS,
    value="domain",
    description="Report:",
    style={'description_width': '60px'},
    layout=widgets.Layout(width='450px'),
)

load_button = widgets.Button(
    description="Load Report",
    button_style="primary",
    icon="chart-bar",
    layout=widgets.Layout(width="140px"),
)

report_output = widgets.Output()

_loaded_reports = set()


def _handle_load(_):
    report_type = report_dropdown.value

    with report_output:
        clear_output()
        print(f"Loading {report_type} report...")

        try:
            if report_type == "domain":
                script_path = scripts_path / "cell_09_openrouter_domain_report.py"
            elif report_type == "page":
                script_path = scripts_path / "cell_10_openrouter_page_report.py"
            elif report_type == "prompt":
                script_path = scripts_path / "cell_11_openrouter_prompt_report.py"
            else:
                raise ValueError(f"Unknown report type: {report_type}")

            if not script_path.exists():
                raise RuntimeError(f"Report script not found: {script_path}")

            # Execute the report script
            exec(compile(open(script_path, encoding='utf-8').read(), script_path, 'exec'), globals())
            _loaded_reports.add(report_type)

        except Exception as exc:
            print(f"Error loading report: {exc}")


load_button.on_click(_handle_load)

# Layout
controls = widgets.VBox([
    widgets.HTML("<h3>Analyse Citations</h3>"),
    widgets.HTML("<p>Select a report type and click Load to generate the analysis.</p>"),
    widgets.HBox([report_dropdown, load_button], layout=widgets.Layout(gap='8px', align_items='center')),
    widgets.HTML("<hr style='margin: 12px 0;'>"),
    report_output,
])

display(controls)
