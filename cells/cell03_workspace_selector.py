import json
import os
import sys
from datetime import datetime
from pathlib import Path

import ipywidgets as widgets
from IPython.display import clear_output, display

try:
    from google.colab import drive  # type: ignore
except ImportError:  # Not running inside Colab
    drive = None

IN_COLAB = "google.colab" in sys.modules
HISTORY_PATH = Path.home() / ".opencite_workspace_history.json"


def _load_history() -> list[str]:
    if HISTORY_PATH.is_file():
        try:
            data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(Path(item).expanduser()) for item in data]
        except Exception:
            return []
    return []


def _save_history(paths: list[str]) -> None:
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_PATH.write_text(json.dumps(paths[:10], indent=2), encoding="utf-8")
    except Exception:
        pass


def _add_to_history(path: Path) -> None:
    existing = _load_history()
    normalized = str(path)
    updated = [normalized] + [p for p in existing if p != normalized]
    _save_history(updated)


def _ensure_drive_mounted() -> Path:
    if drive is None:
        raise RuntimeError("google.colab.drive is unavailable in this environment.")
    mount_point = Path("/content/drive")
    if not mount_point.exists() or not os.path.ismount(mount_point):
        print("🔌 Mounting Google Drive...")
        drive.mount(str(mount_point))
    return mount_point / "MyDrive"


def _build_location_options():
    options = []
    if IN_COLAB:
        options = [
            ("Google Colab Temporary Folder (/content)", "colab"),
            ("Google Drive Folder (/content/drive/MyDrive)", "drive"),
            ("Local Folder (current directory)", "local"),
        ]
    else:
        options = [
            ("Local Folder (current directory)", "local"),
        ]
    return options


def configure_workspace(default_folder: str = "opencite_workspace") -> None:
    recent_paths = _load_history()
    state: dict[str, Path | None] = {"base_path": None}

    header = widgets.HTML("<h3>Cell 03 · Workspace Selector</h3>")

    location_dropdown = widgets.Dropdown(
        options=_build_location_options(),
        value=_build_location_options()[0][1],
        description="Base location:",
        layout=widgets.Layout(width="420px"),
    )
    select_base_button = widgets.Button(description="Select Base Location", icon="map", button_style="info")
    base_status_output = widgets.Output()
    stage_container = widgets.VBox(
        [
            header,
            widgets.HTML("Step 1: Choose where this project should be stored."),
            location_dropdown,
            select_base_button,
            base_status_output,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    def _render_folder_stage(base_path: Path):
        workspace_root = (base_path / default_folder).expanduser().resolve()
        workspace_root.mkdir(parents=True, exist_ok=True)
        project_name_input = widgets.Text(
            value=_default_project_name(),
            placeholder="project_YYYYMMDD_HHMMSS",
            layout=widgets.Layout(width="420px"),
        )
        recent_dropdown = widgets.Dropdown(
            options=[("Please Select", "")] + [(path, path) for path in recent_paths],
            value="",
            layout=widgets.Layout(width="420px"),
        )
        base_path_label = widgets.HTML(
            f"<b>Base path:</b> {base_path}<br><b>Projects stored in:</b> {workspace_root}"
        )
        configure_button = widgets.Button(description="Use Workspace Folder", button_style="primary", icon="folder-open")
        status_output = widgets.Output()

        def _handle_config(_):
            with status_output:
                clear_output()
                try:
                    selected_recent = recent_dropdown.value
                    if selected_recent:
                        workspace_path = Path(selected_recent).expanduser()
                    else:
                        project_name = project_name_input.value.strip()
                        if not project_name:
                            raise RuntimeError("Enter a project folder name (e.g., project_20240101).")
                        workspace_path = (workspace_root / project_name).expanduser()
                    workspace_path.mkdir(parents=True, exist_ok=True)
                    resolved = workspace_path.resolve()
                    os.environ["WORKSPACE_ROOT"] = str(resolved)
                    globals()["WORKSPACE_ROOT"] = resolved
                    print(f"📁 Workspace ready at: {resolved}")
                    print("ℹ️ Run the workspace-setup cell next to build the subfolders/config.")
                    _add_to_history(resolved)
                except Exception as exc:
                    print(f"❌ {exc}")

        configure_button.on_click(_handle_config)

        stage_container.children = (
            [
                header,
                widgets.HTML("<strong>Step 1 complete.</strong> Choose or create a project folder below:"),
                base_path_label,
                widgets.HTML("<b>Create a new project folder</b>"),
                project_name_input,
                widgets.HTML("<b>Or continue a recent project</b>"),
                recent_dropdown,
                configure_button,
                status_output,
            ]
        )

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
                base_path = base_path.expanduser().resolve()
                if not base_path.exists():
                    base_path.mkdir(parents=True, exist_ok=True)
                state["base_path"] = base_path
                print(f"📍 Base location ready: {base_path}")
                _render_folder_stage(base_path)
            except Exception as exc:
                state["base_path"] = None
                print(f"❌ {exc}")

    select_base_button.on_click(_handle_base_select)

    display(stage_container)


configure_workspace()
def _default_project_name() -> str:
    return f"project_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
