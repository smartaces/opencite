import os
import sys
from pathlib import Path

try:
    from google.colab import drive  # type: ignore
except ImportError:  # Not running inside Colab
    drive = None

IN_COLAB = "google.colab" in sys.modules


def configure_workspace(default_folder: str = "opencite_workspace") -> None:
    """Guide the user through selecting a workspace directory."""
    global WORKSPACE_ROOT

    if IN_COLAB:
        print("Select where you want to store project files:")
        print("  1 - Temporary Colab storage (/content)")
        print("  2 - Google Drive (/content/drive/MyDrive)")
        print("  3 - Custom absolute path")
        choice = input("Enter 1, 2, or 3 [default 1]: ").strip() or "1"

        if choice == "2":
            if drive is None:
                raise RuntimeError("google.colab.drive is unavailable in this environment.")
            mount_point = "/content/drive"
            if not Path(mount_point).exists() or not os.path.ismount(mount_point):
                print("🔌 Mounting Google Drive...")
                drive.mount(mount_point)
            base_path = Path(mount_point) / "MyDrive"
        elif choice == "3":
            base_input = input("Enter the full path you want to use: ").strip()
            if not base_input:
                raise RuntimeError("No path provided.")
            base_path = Path(base_input).expanduser()
        else:
            base_path = Path("/content")
    else:
        default_base = Path.cwd()
        prompt = f"Enter workspace path [{default_base}]: "
        base_input = input(prompt).strip()
        base_path = Path(base_input).expanduser() if base_input else default_base

    folder_name = input(f"Folder name inside {base_path} [{default_folder}]: ").strip() or default_folder
    workspace_path = base_path / folder_name
    workspace_path.mkdir(parents=True, exist_ok=True)

    WORKSPACE_ROOT = workspace_path.resolve()
    os.environ["WORKSPACE_ROOT"] = str(WORKSPACE_ROOT)
    print(f"📁 Workspace ready at: {WORKSPACE_ROOT}")


configure_workspace()
