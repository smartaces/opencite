# cell_07_google_url_resolver.py
"""
Google URL Resolver - Post-processing for Vertex redirect URLs.

This cell resolves Google's Vertex redirect URLs to actual page URLs.
Run this AFTER completing a batch with Google/Gemini provider.

The resolver:
1. Reads your batch output CSV
2. Extracts unique Vertex redirect URLs
3. Resolves each URL (with delays to avoid rate limiting)
4. Creates a backup of the original file
5. Saves a resolved version with actual page URLs

Note: This is only needed for Google/Gemini. OpenAI returns actual URLs directly.
"""

import os
import sys
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display, clear_output

# Ensure scripts directory is in path
if 'PATHS' in globals():
    scripts_dir = Path(PATHS.get('scripts', ''))
    if scripts_dir.exists():
        sys.path.insert(0, str(scripts_dir))

# Check for required globals
if 'PATHS' not in globals():
    raise RuntimeError(
        "Workspace not configured. Run the workspace setup cell first."
    )

# Import the resolver
from core.google_url_resolver import GoogleURLResolver, resolve_google_urls


# =============================================================================
# UI Components
# =============================================================================

output_area = widgets.Output()

# File selector
csv_file_input = widgets.Text(
    value="",
    placeholder="Path to detail CSV file (or drag & drop)",
    description="CSV File:",
    style={'description_width': '80px'},
    layout=widgets.Layout(width='500px'),
)

# File browser button
def list_recent_csvs():
    """List recent detail CSVs in the csv_output folder."""
    csv_dir = Path(PATHS.get('csv_output', ''))
    if not csv_dir.exists():
        return []

    csvs = list(csv_dir.glob("*detail*.csv"))
    # Sort by modification time, most recent first
    csvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[:10]  # Last 10

recent_csvs = list_recent_csvs()
csv_dropdown = widgets.Dropdown(
    options=[("Select a recent CSV...", "")] + [(f.name, str(f)) for f in recent_csvs],
    value="",
    description="Recent:",
    style={'description_width': '80px'},
    layout=widgets.Layout(width='500px'),
)

def on_dropdown_change(change):
    if change['new']:
        csv_file_input.value = change['new']

csv_dropdown.observe(on_dropdown_change, names='value')

# Resolve button
resolve_button = widgets.Button(
    description="Resolve URLs",
    button_style="primary",
    icon="refresh",
    layout=widgets.Layout(width='150px'),
)

# Status
status_label = widgets.HTML(value="")


def on_resolve_click(_):
    """Handle resolve button click."""
    csv_path = csv_file_input.value.strip()

    if not csv_path:
        status_label.value = "<span style='color: red;'>Please select or enter a CSV file path.</span>"
        return

    csv_path = Path(csv_path)
    if not csv_path.exists():
        status_label.value = f"<span style='color: red;'>File not found: {csv_path}</span>"
        return

    # Check if it looks like a Google output (has Vertex URLs)
    status_label.value = "<span style='color: blue;'>Starting resolution...</span>"

    with output_area:
        clear_output()

        # Run the resolver
        resolver = GoogleURLResolver()
        result = resolver.resolve_batch_output(csv_path)

        if result:
            status_label.value = f"<span style='color: green;'>Resolution complete! Output: {result.name}</span>"
        else:
            status_label.value = "<span style='color: red;'>Resolution failed. Check output above for details.</span>"


resolve_button.on_click(on_resolve_click)


# =============================================================================
# Display UI
# =============================================================================

form = widgets.VBox([
    widgets.HTML("<h3>Google URL Resolver</h3>"),
    widgets.HTML("<p>Resolve Vertex redirect URLs to actual page URLs. <b>Only needed for Google/Gemini outputs.</b></p>"),
    widgets.HTML("<hr>"),
    widgets.HTML("<b>Select CSV file:</b>"),
    csv_dropdown,
    widgets.HTML("<p style='color: #666; font-size: 12px;'>Or enter path manually:</p>"),
    csv_file_input,
    widgets.HTML("<hr>"),
    resolve_button,
    status_label,
    widgets.HTML("<hr>"),
    widgets.HTML("<b>Output:</b>"),
    output_area,
])

display(form)
