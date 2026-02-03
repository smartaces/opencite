# cell_02_workspace_setup.py
"""
Workspace Setup - Configure storage location for the modular bulk prompt runner.

This cell:
1. Lets you choose where to store your project (Google Drive recommended)
2. Creates all necessary subfolders including the modular scripts folder
3. Sets up global PATHS dict used by all subsequent cells

Run this cell, then click "Select Location" and follow the prompts.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import ipywidgets as widgets
from IPython.display import clear_output, display

try:
    from google.colab import drive  # type: ignore
except ImportError:
    drive = None

IN_COLAB = "google.colab" in sys.modules


def _default_project_name() -> str:
    return f"modular_project_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"


def _ensure_drive_mounted() -> Path:
    if drive is None:
        raise RuntimeError("google.colab.drive is unavailable in this environment.")
    mount_point = Path("/content/drive")
    if not mount_point.exists() or not os.path.ismount(mount_point):
        print("Mounting Google Drive...")
        drive.mount(str(mount_point))
    return mount_point / "MyDrive"


def _scan_existing_projects(workspace_root: Path) -> list[tuple[str, str]]:
    """Scan workspace_root for existing project folders, sorted by most recently modified."""
    if not workspace_root.exists():
        return []

    folders = []
    try:
        for item in workspace_root.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                try:
                    mtime = item.stat().st_mtime
                    folders.append((item, mtime))
                except OSError:
                    continue
    except PermissionError:
        return []

    folders.sort(key=lambda x: x[1], reverse=True)

    result = []
    for folder_path, mtime in folders:
        modified_date = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
        label = f"{folder_path.name}  (modified: {modified_date})"
        result.append((label, str(folder_path)))

    return result


def _build_location_options():
    if IN_COLAB:
        return [
            ("Google Drive Folder (/content/drive/MyDrive)", "drive"),
            ("Google Colab Temporary Folder (/content)", "colab"),
            ("Local Folder (current directory)", "local"),
        ]
    else:
        return [
            ("Local Folder (current directory)", "local"),
        ]


def _create_workspace_structure(workspace_path: Path) -> dict:
    """Create all workspace subfolders and return PATHS dict.

    Note: Uses 'scripts/modular' subfolder to avoid conflicts with
    existing bulk_loader scripts.
    """
    subfolders = {
        "scripts": workspace_path / "scripts" / "modular",  # Separate from bulk_loader
        "search_results": workspace_path / "search_results",
        "extracted_raw": workspace_path / "extracted_raw",
        "csv_output": workspace_path / "csv_output",
        "grabbed": workspace_path / "grabbed",
        "terms_lists": workspace_path / "terms_lists",
        "logs": workspace_path / "logs",
    }

    for path in subfolders.values():
        path.mkdir(parents=True, exist_ok=True)

    return {name: str(path) for name, path in subfolders.items()}


def _write_workspace_config(workspace_path: Path, paths: dict) -> Path:
    """Write workspace config file and return its path."""
    config = {
        "workspace_root": str(workspace_path),
        "paths": paths,
        "version": "modular_v1",
    }
    config_path = workspace_path / "modular_workspace_config.json"
    with open(config_path, "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)
    return config_path


def configure_workspace(default_folder: str = "opencite_modular_workspace") -> None:
    state: dict[str, Path | None] = {"base_path": None}

    header = widgets.HTML("<h3>Workspace Setup</h3>")

    location_dropdown = widgets.Dropdown(
        options=_build_location_options(),
        value=_build_location_options()[0][1],
        description="Location:",
        style={'description_width': '80px'},
        layout=widgets.Layout(width="450px"),
    )
    select_base_button = widgets.Button(
        description="Select Location",
        icon="map",
        button_style="info",
        layout=widgets.Layout(width="140px"),
    )
    base_status_output = widgets.Output()

    stage_container = widgets.VBox(
        [
            header,
            widgets.HTML("<p>Choose where to store your project. Google Drive is recommended for persistence.</p>"),
            widgets.HBox([location_dropdown, select_base_button], layout=widgets.Layout(gap='8px')),
            base_status_output,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    def _render_folder_stage(base_path: Path):
        workspace_root = (base_path / default_folder).expanduser().resolve()
        workspace_root.mkdir(parents=True, exist_ok=True)

        existing_projects = _scan_existing_projects(workspace_root)
        has_existing = len(existing_projects) > 0

        mode_state = {"mode": "existing" if has_existing else "new"}

        # Existing project widgets
        existing_dropdown = widgets.Dropdown(
            options=[("-- Select a project --", "")] + existing_projects,
            value="",
            style={'description_width': '100px'},
            layout=widgets.Layout(width="500px"),
        )
        switch_to_new_button = widgets.Button(
            description="Create new project instead",
            button_style="",
            icon="plus",
            layout=widgets.Layout(width="220px"),
        )
        existing_section = widgets.VBox([
            widgets.HTML(f"<b>Continue an existing project</b> ({len(existing_projects)} found)"),
            existing_dropdown,
            switch_to_new_button,
        ])

        # New project widgets
        project_name_input = widgets.Text(
            value=_default_project_name(),
            placeholder="project_YYYYMMDD_HHMMSS",
            style={'description_width': '100px'},
            layout=widgets.Layout(width="450px"),
        )
        switch_to_existing_button = widgets.Button(
            description="Select existing project instead",
            button_style="",
            icon="folder-open",
            layout=widgets.Layout(width="240px"),
        )
        new_section = widgets.VBox([
            widgets.HTML("<b>Create a new project</b>"),
            project_name_input,
            switch_to_existing_button if has_existing else widgets.HTML(""),
        ])

        base_path_label = widgets.HTML(f"<b>Workspace location:</b> {workspace_root}")

        action_button = widgets.Button(
            description="Open Project" if has_existing else "Create Project",
            button_style="primary",
            icon="folder-open" if has_existing else "plus",
            layout=widgets.Layout(width="150px"),
        )
        status_output = widgets.Output()

        mode_container = widgets.VBox([existing_section if has_existing else new_section])

        def _switch_to_new(_):
            mode_state["mode"] = "new"
            mode_container.children = [new_section]
            action_button.description = "Create Project"
            action_button.icon = "plus"
            existing_dropdown.value = ""

        def _switch_to_existing(_):
            mode_state["mode"] = "existing"
            mode_container.children = [existing_section]
            action_button.description = "Open Project"
            action_button.icon = "folder-open"

        def _handle_action(_):
            with status_output:
                clear_output()
                try:
                    if mode_state["mode"] == "existing":
                        selected = existing_dropdown.value
                        if not selected:
                            raise RuntimeError("Select a project from the dropdown.")
                        workspace_path = Path(selected).expanduser().resolve()
                    else:
                        project_name = project_name_input.value.strip()
                        if not project_name:
                            raise RuntimeError("Enter a project folder name.")
                        workspace_path = (workspace_root / project_name).expanduser().resolve()

                    # Create workspace structure
                    paths = _create_workspace_structure(workspace_path)

                    # Write config
                    config_path = _write_workspace_config(workspace_path, paths)

                    # Set up environment
                    os.environ["WORKSPACE_ROOT"] = str(workspace_path)
                    os.environ["WORKSPACE_CONFIG"] = str(config_path)

                    # Set globals
                    import __main__
                    __main__.WORKSPACE_ROOT = workspace_path
                    __main__.PATHS = {k: Path(v) for k, v in paths.items()}
                    __main__.WORKSPACE_CONFIG = config_path

                    # Also set in current namespace
                    globals()["WORKSPACE_ROOT"] = workspace_path
                    globals()["PATHS"] = {k: Path(v) for k, v in paths.items()}
                    globals()["WORKSPACE_CONFIG"] = config_path

                    print(f"Workspace ready: {workspace_path}")
                    print(f"Folders created:")
                    for name in paths:
                        print(f"   - {name}")
                    print(f"\nRun the next cell to download/update modular scripts.")

                except Exception as exc:
                    print(f"Error: {exc}")

        switch_to_new_button.on_click(_switch_to_new)
        switch_to_existing_button.on_click(_switch_to_existing)
        action_button.on_click(_handle_action)

        stage_container.children = [
            header,
            widgets.HTML("<p><strong>Location selected.</strong> Now choose or create a project.</p>"),
            base_path_label,
            widgets.HTML("<hr style='margin: 12px 0;'>"),
            mode_container,
            widgets.HTML("<hr style='margin: 12px 0;'>"),
            action_button,
            status_output,
        ]

    def _handle_base_select(_):
        with base_status_output:
            clear_output()
            try:
                selection = location_dropdown.value
                if selection == "colab":
                    base_path = Path("/content")
                elif selection == "drive":
                    base_path = _ensure_drive_mounted()
                elif selection == "local":
                    base_path = Path.cwd()
                else:
                    base_path = Path.cwd()

                base_path = base_path.expanduser().resolve()
                if not base_path.exists():
                    base_path.mkdir(parents=True, exist_ok=True)
                state["base_path"] = base_path
                print(f"Location ready: {base_path}")
                _render_folder_stage(base_path)
            except Exception as exc:
                state["base_path"] = None
                print(f"Error: {exc}")

    select_base_button.on_click(_handle_base_select)

    display(stage_container)


# Run the workspace configuration UI
configure_workspace()
