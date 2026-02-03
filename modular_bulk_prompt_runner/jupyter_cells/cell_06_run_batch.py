# cell_06_run_batch.py
"""
Batch Runner - Execute prompts through the configured search agent.

This cell:
1. Loads a CSV of prompts from the terms_lists folder
2. Executes each prompt through the selected search agent
3. Supports multi-turn conversations with optional personas
4. Saves results to CSV files in csv_output folder

Prerequisites:
- Cell 04a: Search agent must be initialized
- Cell 04b: Conversation agent (optional, for multi-turn)
- Cell 05: Prompts CSV must be uploaded
"""

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
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
TERMS_DIR = Path(PATHS['terms_lists'])

# Check for search agent
if '_search_agent_state' not in globals():
    raise RuntimeError("Search agent not initialized. Run cell 04a first.")

if not globals().get('_search_agent_state', {}).get('initialized'):
    raise RuntimeError("Search agent not initialized. Run cell 04a and click 'Confirm Selection'.")

# Add scripts to path if needed
scripts_dir = Path(PATHS.get('scripts', ''))
if scripts_dir.exists() and str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Import core components
from core import ReportHelper, BatchRunner


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _list_csv_files():
    TERMS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(TERMS_DIR).glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in ["prompt", "persona", "runs", "turns"] if col not in df.columns]
    if missing:
        raise ValueError(f"The selected CSV is missing required columns: {missing}. Re-run the CSV loader cell.")
    return df


def _format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def _format_location(country: str = "US", city: str = "New York", region: str = "New York") -> dict:
    """Format location dict for API calls."""
    return {
        "country": country,
        "city": city,
        "region": region,
        "type": "approximate",
    }


# ============================================================================
# SPINNER CLASS
# ============================================================================

class Spinner:
    """A simple spinner that runs in a background thread."""

    FRAMES = ['|', '/', '-', '\\']

    def __init__(self, label_widget):
        self._label = label_widget
        self._running = False
        self._thread = None
        self._message = "Running..."
        self._frame_idx = 0

    def start(self, message: str = "Running..."):
        """Start the spinner with a message."""
        self._message = message
        self._running = True
        self._frame_idx = 0
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def update(self, message: str):
        """Update the spinner message."""
        self._message = message

    def stop(self, final_message: str = None):
        """Stop the spinner and optionally display a final message."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        if final_message:
            self._label.value = final_message

    def _spin(self):
        """Background thread that updates the spinner."""
        while self._running:
            frame = self.FRAMES[self._frame_idx % len(self.FRAMES)]
            self._label.value = f"{frame} {self._message}"
            self._frame_idx += 1
            time.sleep(0.1)


# ============================================================================
# WIDGET STYLING
# ============================================================================

STYLE = {'description_width': '100px'}
DROPDOWN_LAYOUT = widgets.Layout(width='420px')
TEXT_LAYOUT = widgets.Layout(width='420px')


# ============================================================================
# CSV SELECTION WIDGETS
# ============================================================================

csv_files = _list_csv_files()
csv_dropdown = widgets.Dropdown(
    options=[(file.name, str(file)) for file in csv_files] or [("No CSV files found", "")],
    value=str(csv_files[0]) if csv_files else "",
    description="CSV file:",
    style=STYLE,
    layout=DROPDOWN_LAYOUT,
)
refresh_button = widgets.Button(description="Refresh", icon="refresh", layout=widgets.Layout(width='90px'))
csv_info_output = widgets.Output()
_csv_state = {"path": None, "df": None}


# ============================================================================
# RUN LABEL WIDGET
# ============================================================================

run_label_input = widgets.Text(
    value="",
    description="Run label:",
    placeholder="Optional tag (auto-generated if blank)",
    style=STYLE,
    layout=TEXT_LAYOUT,
)


# ============================================================================
# SEARCH DEPTH WIDGETS (only shown if model supports it)
# ============================================================================

context_dropdown = widgets.Dropdown(
    options=[
        ("Low - Minimal web content, fastest", "low"),
        ("Medium - Balanced retrieval (recommended)", "medium"),
        ("High - Extensive web content, most thorough", "high"),
    ],
    value="medium",
    description="Level:",
    style=STYLE,
    layout=DROPDOWN_LAYOUT,
)


# ============================================================================
# REASONING EFFORT WIDGETS (only shown if model supports it)
# ============================================================================

reasoning_dropdown = widgets.Dropdown(
    options=[
        ("Low - Quick responses, minimal deliberation", "low"),
        ("Medium - Balanced thinking", "medium"),
        ("High - Deep analysis, slower but more nuanced", "high"),
    ],
    value="low",
    description="Level:",
    style=STYLE,
    layout=DROPDOWN_LAYOUT,
)


# ============================================================================
# PARALLEL RUNS WIDGETS
# ============================================================================

parallel_dropdown = widgets.Dropdown(
    options=[
        ("1 - Sequential (no parallelism)", 1),
        ("2 - Run 2 queries at once", 2),
        ("3 - Run 3 queries at once (recommended)", 3),
        ("4 - Run 4 queries at once", 4),
        ("5 - Run 5 queries at once (fastest)", 5),
    ],
    value=1,
    description="Batch size:",
    style=STYLE,
    layout=DROPDOWN_LAYOUT,
)


# ============================================================================
# LOCATION BIAS WIDGETS
# ============================================================================

location_toggle = widgets.Checkbox(
    value=False,
    description="Enable location bias",
    style={'description_width': 'auto'},
    layout=widgets.Layout(width='200px'),
)

location_presets = widgets.Dropdown(
    options=[
        ("US - New York", "us_ny"),
        ("UK - London", "uk_london"),
        ("US - San Francisco", "us_sf"),
        ("France - Paris", "fr_paris"),
        ("Germany - Berlin", "de_berlin"),
        ("Custom location", "custom"),
    ],
    value="uk_london",
    description="Location:",
    style=STYLE,
    layout=DROPDOWN_LAYOUT,
    disabled=True,
)

custom_country = widgets.Text(value="", description="Country:", placeholder="e.g. GB", style={'description_width': '60px'}, layout=widgets.Layout(width='130px'), disabled=True)
custom_city = widgets.Text(value="", description="City:", placeholder="e.g. London", style={'description_width': '40px'}, layout=widgets.Layout(width='150px'), disabled=True)
custom_region = widgets.Text(value="", description="Region:", placeholder="e.g. England", style={'description_width': '50px'}, layout=widgets.Layout(width='160px'), disabled=True)

preset_locations = {
    "us_ny": ("US", "New York", "New York"),
    "uk_london": ("GB", "London", "England"),
    "us_sf": ("US", "San Francisco", "California"),
    "fr_paris": ("FR", "Paris", "Ile-de-France"),
    "de_berlin": ("DE", "Berlin", "Berlin"),
}


# ============================================================================
# RUN BUTTON & OUTPUT WIDGETS
# ============================================================================

run_button = widgets.Button(
    description="Run Batch",
    button_style="primary",
    icon="play",
    layout=widgets.Layout(width='120px'),
)

stop_button = widgets.Button(
    description="Stop",
    button_style="danger",
    icon="stop",
    layout=widgets.Layout(width='80px'),
    disabled=True,
)

# Spinner label
spinner_label = widgets.HTML(value="", layout=widgets.Layout(width='100%'))

# Summary output
summary_output = widgets.Output(layout=widgets.Layout(width='100%'))

# Detail output (scrollable)
detail_output = widgets.Output(layout=widgets.Layout(
    width='100%',
    max_height='400px',
    overflow_y='auto',
    border='1px solid #ccc',
    padding='8px',
))

detail_container = widgets.VBox([
    widgets.HTML("<b>Detailed Output</b> <span style='color: #666; font-size: 0.9em;'>(scroll to view)</span>"),
    detail_output
], layout=widgets.Layout(display='none'))

# Batch runner instance
_batch_runner = None


# ============================================================================
# CSV LOADING & SUMMARY
# ============================================================================

def _refresh_csv_list(_=None):
    files = _list_csv_files()
    if files:
        csv_dropdown.options = [(file.name, str(file)) for file in files]
        csv_dropdown.value = str(files[0])
    else:
        csv_dropdown.options = [("No CSV files found", "")]
        csv_dropdown.value = ""
    _load_selected_csv()


def _analyze_numeric_column(series: pd.Series) -> dict:
    blank = 0
    invalid = 0
    valid = 0
    total_value = 0
    for raw in series:
        if pd.isna(raw) or str(raw).strip() == "":
            blank += 1
            continue
        try:
            number = int(float(raw))
        except (ValueError, TypeError):
            invalid += 1
            continue
        if number >= 1:
            valid += 1
            total_value += number
        else:
            invalid += 1
    return {"valid": valid, "blank": blank, "invalid": invalid, "total": total_value}


def _analyze_persona(series: pd.Series) -> dict:
    text = series.fillna("").astype(str).str.strip()
    blank = (text == "").sum()
    return {"filled": len(series) - blank, "blank": int(blank)}


def _update_csv_summary(df: pd.DataFrame):
    persona_stats = _analyze_persona(df["persona"])
    runs_stats = _analyze_numeric_column(df["runs"])
    turns_stats = _analyze_numeric_column(df["turns"])

    with csv_info_output:
        clear_output()

        lines = [f"Loaded {len(df)} prompts"]

        if persona_stats['filled'] == len(df):
            lines.append(f"Personas: all {persona_stats['filled']} filled")
        elif persona_stats['filled'] == 0:
            lines.append(f"Personas: all blank (will run without persona)")
        else:
            lines.append(f"Personas: {persona_stats['filled']} filled, {persona_stats['blank']} blank")

        if runs_stats['blank'] == 0 and runs_stats['invalid'] == 0:
            lines.append(f"Runs: all valid (total: {runs_stats['total']} runs across all prompts)")
        else:
            parts = [f"{runs_stats['valid']} valid"]
            if runs_stats['blank'] > 0:
                parts.append(f"{runs_stats['blank']} blank -> fallback to 1")
            if runs_stats['invalid'] > 0:
                parts.append(f"{runs_stats['invalid']} invalid -> fallback to 1")
            lines.append(f"Runs: {', '.join(parts)}")

        if turns_stats['blank'] == 0 and turns_stats['invalid'] == 0:
            lines.append(f"Turns: all valid (total: {turns_stats['total']} turns across all prompts)")
        else:
            parts = [f"{turns_stats['valid']} valid"]
            if turns_stats['blank'] > 0:
                parts.append(f"{turns_stats['blank']} blank -> fallback to 1")
            if turns_stats['invalid'] > 0:
                parts.append(f"{turns_stats['invalid']} invalid -> fallback to 1")
            lines.append(f"Turns: {', '.join(parts)}")

        for line in lines:
            print(line)


def _load_selected_csv(change=None):
    path = csv_dropdown.value
    if not path:
        _csv_state["path"] = None
        _csv_state["df"] = None
        with csv_info_output:
            clear_output()
            print("No CSV selected")
        return None
    try:
        df = _load_csv(Path(path))
    except Exception as exc:
        _csv_state["path"] = None
        _csv_state["df"] = None
        with csv_info_output:
            clear_output()
            print(f"Failed to load: {exc}")
        return None
    _csv_state["path"] = path
    _csv_state["df"] = df
    _update_csv_summary(df)
    return df


# ============================================================================
# LOCATION WIDGET UPDATES
# ============================================================================

def _update_location_widgets(change=None):
    enabled = location_toggle.value
    location_presets.disabled = not enabled
    is_custom = enabled and location_presets.value == "custom"
    for w in (custom_country, custom_city, custom_region):
        w.disabled = not is_custom


def _get_location_dict():
    if not location_toggle.value:
        return None
    if location_presets.value != "custom":
        country, city, region = preset_locations[location_presets.value]
        return _format_location(country=country, city=city, region=region)
    country = custom_country.value.strip() or "US"
    city = custom_city.value.strip() or "New York"
    region = custom_region.value.strip() or city
    return _format_location(country=country, city=city, region=region)


# ============================================================================
# BATCH EXECUTION
# ============================================================================

def _run_batch(_):
    global _batch_runner

    df = _csv_state.get("df")
    if df is None or df.empty:
        with summary_output:
            clear_output()
            print("Load a CSV before running.")
        return

    # Clear outputs
    with summary_output:
        clear_output()
    with detail_output:
        clear_output()
    detail_container.layout.display = 'none'

    # Get search agent state
    search_state = globals().get('_search_agent_state', {})
    if not search_state.get('initialized'):
        with summary_output:
            print("Search agent not initialized. Run cell 04a first.")
        return

    # Get conversation agent state (optional)
    conv_state = globals().get('_conversation_agent_state', {})
    conv_initialized = conv_state.get('initialized', False)

    # Build batch runner
    _batch_runner = BatchRunner(
        search_cartridge=search_state['cartridge'],
        search_client=search_state['client'],
        search_model=search_state['model_id'],
        conversation_cartridge=conv_state.get('cartridge') if conv_initialized else None,
        conversation_client=conv_state.get('client') if conv_initialized else None,
        conversation_model=conv_state.get('model_id') if conv_initialized else None,
        report_helper_class=ReportHelper,
        paths=PATHS,
    )

    # Build parameters
    user_location = _get_location_dict()
    user_label = run_label_input.value.strip()
    default_label = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_label = user_label or default_label

    params = {
        'search_context_size': context_dropdown.value,
        'reasoning_effort': reasoning_dropdown.value,
    }
    if user_location:
        params['location'] = user_location

    # UI state
    run_button.disabled = True
    stop_button.disabled = False

    # Start spinner
    spinner = Spinner(spinner_label)
    spinner.start(f"Starting batch of {len(df)} prompts...")

    batch_start = time.time()

    # Print header to summary
    with summary_output:
        print(f"Batch: {len(df)} prompts from {Path(csv_dropdown.value).name}")
        print(f"   Provider: {search_state['cartridge'].name}")
        print(f"   Model: {search_state['model_id']}")
        print(f"   Search: {context_dropdown.value} | Reasoning: {reasoning_dropdown.value}")
        if user_location:
            print(f"   Location: {user_location['city']}, {user_location['country']}")
        print(f"   Label: {run_label}")
        if conv_initialized:
            print(f"   Conversation Agent: {conv_state['cartridge'].name} / {conv_state['model_id']}")
        else:
            print(f"   Conversation Agent: Not configured (single-turn only)")
        print("-" * 60)

    all_detail_output = []

    def progress_callback(msg, current, total):
        spinner.update(f"[{current}/{total}] {msg}")

    # Run batch
    try:
        result = _batch_runner.run_batch(
            prompts_df=df,
            params=params,
            run_label=run_label,
            parallel_workers=parallel_dropdown.value,
            on_progress=progress_callback,
        )

        # Collect detail lines
        for r in result.get('results', []):
            all_detail_output.extend(r.get('detail_lines', []))

        # Summary
        batch_duration = time.time() - batch_start
        spinner.stop(f"Complete! {result['successful']}/{result['total_runs']} runs in {_format_duration(batch_duration)}")

        with summary_output:
            print("-" * 60)
            print(f"Finished: {result['successful']}/{result['total_runs']} runs successful")
            print(f"Total citations: {result['total_citations']}")
            print(f"Total time: {_format_duration(batch_duration)}")
            if result['failed'] > 0:
                print(f"Failed: {result['failed']} runs")

    except Exception as exc:
        spinner.stop(f"Error: {exc}")
        with summary_output:
            print(f"Batch failed: {exc}")

    # Dump detail output
    with detail_output:
        for line in all_detail_output:
            print(line)

    detail_container.layout.display = 'block'

    # Reset UI state
    run_button.disabled = False
    stop_button.disabled = True


def _stop_batch(_):
    global _batch_runner
    if _batch_runner:
        _batch_runner.stop()
        with summary_output:
            print("\n[STOP REQUESTED]")


# ============================================================================
# EVENT BINDINGS
# ============================================================================

refresh_button.on_click(_refresh_csv_list)
csv_dropdown.observe(_load_selected_csv, names="value")
location_toggle.observe(_update_location_widgets, names="value")
location_presets.observe(_update_location_widgets, names="value")
run_button.on_click(_run_batch)
stop_button.on_click(_stop_batch)

_update_location_widgets()
_load_selected_csv()


# ============================================================================
# LAYOUT
# ============================================================================

# Check model capabilities to show/hide certain options
search_state = globals().get('_search_agent_state', {})
cartridge = search_state.get('cartridge')
model_id = search_state.get('model_id', '')

# Get model schema if available
model_schema = None
if cartridge:
    for m in cartridge.models:
        if m.id == model_id:
            model_schema = m
            break

# Build dynamic controls based on model capabilities
controls_list = [
    widgets.HTML("<h3>Batch Runner</h3>"),
    widgets.HTML("<p>Run your prompts through the AI search assistant and save the results.</p>"),

    widgets.HTML("<hr style='margin: 8px 0;'>"),
    widgets.HTML("<b>Prompt file</b>"),
    widgets.HBox([csv_dropdown, refresh_button], layout=widgets.Layout(gap='8px', align_items='center')),
    csv_info_output,

    widgets.HTML("<hr style='margin: 12px 0;'>"),
    widgets.HTML("<b>Run label</b> - Optional tag for organising output files"),
    run_label_input,
]

# Only show search depth if model supports it
if model_schema is None or model_schema.search_context_options:
    controls_list.extend([
        widgets.HTML("<hr style='margin: 12px 0;'>"),
        widgets.HTML("<b>Search depth</b> - Controls how much web content is retrieved per query"),
        context_dropdown,
    ])

# Only show reasoning if model supports it
if model_schema is None or model_schema.supports_reasoning:
    controls_list.extend([
        widgets.HTML("<hr style='margin: 12px 0;'>"),
        widgets.HTML("<b>Reasoning effort</b> - Controls how deeply the model analyses content"),
        reasoning_dropdown,
    ])

controls_list.extend([
    widgets.HTML("<hr style='margin: 12px 0;'>"),
    widgets.HTML("<b>Parallel runs</b> - Run multiple queries simultaneously for faster processing"),
    parallel_dropdown,

    widgets.HTML("<hr style='margin: 8px 0;'>"),
    widgets.HTML("<b>Location bias</b> - Personalises results as if searching from this location"),
    location_toggle,
    location_presets,
    widgets.HBox([custom_country, custom_city, custom_region], layout=widgets.Layout(gap='8px')),

    widgets.HTML("<hr style='margin: 12px 0;'>"),
    widgets.HBox([run_button, stop_button], layout=widgets.Layout(gap='8px')),

    widgets.HTML("<hr style='margin: 12px 0;'>"),
    spinner_label,
    summary_output,

    widgets.HTML("<br/>"),
    detail_container,
])

controls = widgets.VBox(controls_list)

display(controls)
