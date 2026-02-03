# cell_05_upload_prompts.py
"""
CSV Loader & Persona Mapper - Upload prompts for batch execution.

This cell:
1. Uploads your prompt list CSV
2. Auto-detects column mappings (prompt, persona, runs, turns)
3. Saves a normalized file to the terms_lists folder

Required columns: prompt
Optional columns: persona, runs, turns
"""

import io
import json
import os
import re
from datetime import datetime
from pathlib import Path

import ipywidgets as widgets
import pandas as pd
from IPython.display import display, clear_output


def _load_paths():
    """Ensure PATHS is populated even if the notebook kernel was restarted."""
    if 'PATHS' in globals() and globals()['PATHS']:
        return {k: Path(v) for k, v in globals()['PATHS'].items()}

    config_path = Path(os.environ.get('WORKSPACE_CONFIG', ''))
    if not config_path.is_file():
        raise RuntimeError("Workspace not configured. Run the workspace setup cell first.")

    with open(config_path, 'r', encoding='utf-8') as fp:
        config = json.load(fp)
    return {k: Path(v) for k, v in config['paths'].items()}


PATHS = _load_paths()
TERMS_DIR = Path(PATHS['terms_lists'])
TERMS_DIR.mkdir(parents=True, exist_ok=True)

persona_presets = [
    ("(blank) No persona", ""),
    ("Budget-conscious student", "A budget-conscious university student researching affordable tech."),
    ("Enterprise IT lead", "An enterprise IT leader focused on security, compliance, and lifecycle planning."),
    ("Health & fitness coach", "A health and fitness coach advising clients on reliable wearables and services."),
    ("Travel storyteller", "A travel storyteller who values photography, roaming support, and battery life."),
    ("Custom persona", "custom"),
]

# ============================================================================
# AUTO-DETECTION PATTERNS
# ============================================================================

COLUMN_PATTERNS = {
    "prompt": [
        r"^prompt[s]?$",
        r"^query$",
        r"^question[s]?$",
        r"^search[_\s]?term[s]?$",
        r"^term[s]?$",
        r"^keyword[s]?$",
        r"^input[s]?$",
        r"^text$",
        r"^message[s]?$",
        r"^instruction[s]?$",
    ],
    "persona": [
        r"^persona[s]?$",
        r"^audience$",
        r"^user[_\s]?type$",
        r"^role$",
        r"^profile$",
        r"^character$",
        r"^context$",
    ],
    "runs": [
        r"^run[s]?$",
        r"^iteration[s]?$",
        r"^repeat[s]?$",
        r"^count$",
        r"^num[_\s]?run[s]?$",
        r"^n[_\s]?run[s]?$",
    ],
    "turns": [
        r"^turn[s]?$",
        r"^round[s]?$",
        r"^step[s]?$",
        r"^num[_\s]?turn[s]?$",
        r"^n[_\s]?turn[s]?$",
        r"^conversation[_\s]?turn[s]?$",
    ],
}


def _detect_column(columns: list[str], field: str) -> str | None:
    """Attempt to find the best matching column for a given field type."""
    patterns = COLUMN_PATTERNS.get(field, [])
    if not patterns:
        return None

    for col in columns:
        col_lower = col.lower().strip()
        for pattern in patterns:
            if re.match(pattern, col_lower):
                return col
    return None


def _auto_detect_columns(columns: list[str]) -> dict[str, str | None]:
    """Auto-detect likely columns for prompt, persona, runs, turns."""
    return {
        "prompt": _detect_column(columns, "prompt"),
        "persona": _detect_column(columns, "persona"),
        "runs": _detect_column(columns, "runs"),
        "turns": _detect_column(columns, "turns"),
    }


# ============================================================================
# WIDGETS
# ============================================================================

COLUMN_STYLE = {'description_width': 'initial'}
COLUMN_LAYOUT = widgets.Layout(width="460px")
SHORT_LAYOUT = widgets.Layout(width="220px")

upload_widget = widgets.FileUpload(accept=".csv", multiple=False, description="Upload CSV")
has_headers_checkbox = widgets.Checkbox(
    value=True,
    description="File has headers",
    style=COLUMN_STYLE,
    layout=widgets.Layout(width="200px"),
)
detection_output = widgets.Output()
preview_output = widgets.Output()

prompt_dropdown = widgets.Dropdown(
    description="Prompt column:",
    options=[("Select column...", None)],
    style=COLUMN_STYLE,
    layout=COLUMN_LAYOUT,
)

persona_mode = widgets.Dropdown(
    options=[("Manual input", "manual"), ("Use CSV column", "csv")],
    value="manual",
    description="Persona source:",
    style=COLUMN_STYLE,
    layout=widgets.Layout(width="320px"),
)
persona_dropdown = widgets.Dropdown(
    options=persona_presets,
    description="Persona preset:",
    style=COLUMN_STYLE,
    layout=COLUMN_LAYOUT,
)
persona_textarea = widgets.Textarea(
    value="",
    description="Custom persona:",
    layout=widgets.Layout(width="100%", height="80px"),
    disabled=True,
    style=COLUMN_STYLE,
)
persona_column_dropdown = widgets.Dropdown(
    description="Persona column:",
    options=[("Select column...", None)],
    disabled=True,
    style=COLUMN_STYLE,
    layout=COLUMN_LAYOUT,
)

runs_column_dropdown = widgets.Dropdown(
    description="Runs column:",
    options=[("Use default (1)", None)],
    style=COLUMN_STYLE,
    layout=COLUMN_LAYOUT,
)
default_runs_input = widgets.BoundedIntText(
    value=1,
    min=1,
    description="Default runs:",
    layout=SHORT_LAYOUT,
    style=COLUMN_STYLE,
)

turns_column_dropdown = widgets.Dropdown(
    description="Turns column:",
    options=[("Use default (1)", None)],
    style=COLUMN_STYLE,
    layout=COLUMN_LAYOUT,
)
default_turns_input = widgets.BoundedIntText(
    value=1,
    min=1,
    description="Default turns:",
    layout=SHORT_LAYOUT,
    style=COLUMN_STYLE,
)

filename_input = widgets.Text(
    value="normalized_prompts",
    description="Filename prefix:",
    placeholder="normalized_prompts",
    layout=COLUMN_LAYOUT,
    style=COLUMN_STYLE,
)
save_button = widgets.Button(description="Save Settings", button_style="success", disabled=True)
status_output = widgets.Output()

_state = {"df": None, "columns": []}

manual_persona_box = widgets.VBox(
    [
        widgets.HTML("<b>Manual persona</b>"),
        persona_dropdown,
        persona_textarea,
    ],
    layout=widgets.Layout(width="100%"),
)

csv_persona_box = widgets.VBox(
    [
        widgets.HTML("<b>Persona column</b>"),
        persona_column_dropdown,
    ],
    layout=widgets.Layout(width="100%"),
)


def _build_options(columns, include_none_label="Select column..."):
    options = []
    if include_none_label:
        options.append((include_none_label, None))
    options.extend([(col, col) for col in columns])
    return options


def _update_column_widgets(columns: list[str], detected: dict[str, str | None]):
    """Update dropdown options and pre-select detected columns."""
    # Prompt dropdown
    prompt_dropdown.options = _build_options(columns)
    if detected["prompt"]:
        prompt_dropdown.value = detected["prompt"]

    # Persona column dropdown
    persona_column_dropdown.options = _build_options(columns)
    if detected["persona"]:
        persona_column_dropdown.value = detected["persona"]
        # Auto-switch to CSV mode if persona column detected
        persona_mode.value = "csv"

    # Runs dropdown
    runs_column_dropdown.options = [("Use default", None)] + [(col, col) for col in columns]
    if detected["runs"]:
        runs_column_dropdown.value = detected["runs"]

    # Turns dropdown
    turns_column_dropdown.options = [("Use default", None)] + [(col, col) for col in columns]
    if detected["turns"]:
        turns_column_dropdown.value = detected["turns"]


def _display_detection_summary(detected: dict[str, str | None]):
    """Show what columns were auto-detected."""
    with detection_output:
        clear_output()
        found = {k: v for k, v in detected.items() if v is not None}
        if found:
            items = [f"<b>{k}</b> -> {v}" for k, v in found.items()]
            html = f"<div style='background:#e8f5e9; padding:8px 12px; border-radius:4px; margin:8px 0;'>Auto-detected: {', '.join(items)}</div>"
            display(widgets.HTML(html))
        else:
            html = "<div style='background:#fff3e0; padding:8px 12px; border-radius:4px; margin:8px 0;'>No columns auto-detected. Please map manually below.</div>"
            display(widgets.HTML(html))


def _handle_upload(change):
    if not upload_widget.value:
        return

    uploaded = list(upload_widget.value.values())[0]
    content = uploaded["content"]

    try:
        if has_headers_checkbox.value:
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content), header=None)
            df.columns = [f"Column {i+1}" for i in range(len(df.columns))]
    except Exception as exc:
        with preview_output:
            clear_output()
            print(f"Failed to read CSV: {exc}")
        with detection_output:
            clear_output()
        _state.update({"df": None, "columns": []})
        save_button.disabled = True
        return

    columns = list(df.columns)
    _state.update({"df": df, "columns": columns})

    # Auto-detect columns
    detected = _auto_detect_columns(columns)

    # Display detection summary first
    _display_detection_summary(detected)

    # Update dropdowns with detected values
    _update_column_widgets(columns, detected)

    save_button.disabled = False

    # Show preview
    with preview_output:
        clear_output()
        display(widgets.HTML(f"<div style='margin-bottom:8px;'><b>{len(df)} rows</b> loaded with columns: <code>{', '.join(columns)}</code></div>"))
        display(df.head())


def _update_persona_widgets(change=None):
    is_manual = persona_mode.value == "manual"
    persona_dropdown.disabled = not is_manual
    persona_textarea.disabled = not (is_manual and persona_dropdown.value == "custom")
    persona_column_dropdown.disabled = is_manual
    manual_persona_box.layout.display = "flex" if is_manual else "none"
    csv_persona_box.layout.display = "none" if is_manual else "flex"


def _update_persona_dropdown(change=None):
    persona_textarea.disabled = persona_dropdown.value != "custom" or persona_mode.value != "manual"


def _validate_persona_value(value: str) -> str:
    value = (value or "").strip()
    if len(value) > 250:
        raise ValueError("Persona/audience descriptions must be 250 characters or fewer.")
    return value


def _coerce_positive_integers(series: pd.Series, label: str, default_value: int) -> pd.Series:
    values = []
    for idx, raw in series.items():
        if pd.isna(raw) or str(raw).strip() == "":
            values.append(default_value)
            continue
        try:
            if isinstance(raw, str) and raw.strip() == "":
                values.append(default_value)
                continue
            num = int(float(raw))
        except (ValueError, TypeError):
            raise ValueError(f"{label}: Row {idx + 2 if has_headers_checkbox.value else idx + 1} contains a non-numeric value '{raw}'.")
        if num < 1:
            raise ValueError(f"{label}: Row {idx + 2 if has_headers_checkbox.value else idx + 1} must be >= 1 (found {num}).")
        values.append(num)
    return pd.Series(values, index=series.index, dtype=int)


def _get_manual_persona() -> pd.Series:
    preset_value = persona_dropdown.value
    if preset_value == "custom":
        persona_value = _validate_persona_value(persona_textarea.value)
    else:
        persona_value = preset_value.strip()
        _ = _validate_persona_value(persona_value)
    return persona_value


def _handle_save(_):
    df = _state["df"]
    if df is None or df.empty:
        with status_output:
            clear_output()
            print("Upload a CSV before saving.")
        return

    prompt_col = prompt_dropdown.value
    if not prompt_col:
        with status_output:
            clear_output()
            print("Select which column contains the prompt text.")
        return

    try:
        prompts = df[prompt_col].astype(str).str.strip()
    except Exception as exc:
        with status_output:
            clear_output()
            print(f"Unable to read prompts: {exc}")
        return

    prompts = prompts.replace({"nan": ""})
    prompts = prompts[prompts != ""]
    if prompts.empty:
        with status_output:
            clear_output()
            print("No valid prompt rows were found after trimming empty values.")
        return

    persona_series = None
    try:
        if persona_mode.value == "manual":
            persona_value = _get_manual_persona()
            persona_series = pd.Series([persona_value] * len(df), index=df.index)
        else:
            persona_col = persona_column_dropdown.value
            if not persona_col:
                raise ValueError("Select the persona column or switch to manual entry.")
            persona_series = df[persona_col].fillna("").astype(str).str.strip()
            persona_series.apply(_validate_persona_value)
    except ValueError as exc:
        with status_output:
            clear_output()
            print(f"Persona validation error: {exc}")
        return

    try:
        if runs_column_dropdown.value:
            runs_series = _coerce_positive_integers(df[runs_column_dropdown.value], "Runs column", default_runs_input.value)
        else:
            runs_series = pd.Series([default_runs_input.value] * len(df), index=df.index, dtype=int)

        if turns_column_dropdown.value:
            turns_series = _coerce_positive_integers(df[turns_column_dropdown.value], "Turns column", default_turns_input.value)
        else:
            turns_series = pd.Series([default_turns_input.value] * len(df), index=df.index, dtype=int)
    except ValueError as exc:
        with status_output:
            clear_output()
            print(f"{exc}")
        return

    normalized = pd.DataFrame(
        {
            "prompt": df[prompt_col].astype(str).str.strip(),
            "persona": persona_series,
            "runs": runs_series.astype(int),
            "turns": turns_series.astype(int),
        }
    )

    normalized = normalized[normalized["prompt"] != ""]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = filename_input.value.strip() or "normalized_prompts"
    filename = f"{prefix}_{timestamp}.csv"
    output_path = TERMS_DIR / filename

    normalized.to_csv(output_path, index=False)

    # Set global for use in batch runner
    import __main__
    __main__.PROMPTS_CSV_PATH = output_path

    with status_output:
        clear_output()
        display(widgets.HTML(f"<div style='background:#e8f5e9; padding:8px 12px; border-radius:4px;'>Saved <b>{len(normalized)} prompts</b> to:<br><code style='font-size:11px;'>{output_path}</code></div>"))


upload_widget.observe(_handle_upload, names="value")
persona_mode.observe(_update_persona_widgets, names="value")
persona_dropdown.observe(_update_persona_dropdown, names="value")
save_button.on_click(_handle_save)
_update_persona_widgets()

# ============================================================================
# LAYOUT
# ============================================================================

form = widgets.VBox(
    [
        widgets.HTML("<h3>CSV Loader & Persona Mapper</h3>"),
        widgets.HTML("<p style='margin-bottom:12px;'>Upload your prompt list, map the columns, and save a normalized file to <code>terms_lists/</code>.</p>"),
        widgets.HBox([upload_widget, has_headers_checkbox], layout=widgets.Layout(width="100%", justify_content="flex-start", gap="10px")),
        detection_output,
        preview_output,
        widgets.HTML("<hr style='margin:16px 0 12px 0;'><b>Column Mapping</b> <small style='color:#666;'>(auto-detected where possible)</small>"),
        prompt_dropdown,
        persona_mode,
        manual_persona_box,
        csv_persona_box,
        widgets.HBox([runs_column_dropdown, default_runs_input], layout=widgets.Layout(width="100%", justify_content="space-between", gap="12px")),
        widgets.HBox([turns_column_dropdown, default_turns_input], layout=widgets.Layout(width="100%", justify_content="space-between", gap="12px")),
        widgets.HTML("<hr style='margin:16px 0 12px 0;'><b>Save Options</b>"),
        filename_input,
        save_button,
        status_output,
    ]
)

display(form)
