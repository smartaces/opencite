import builtins
import json
import os
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("WORKSPACE_CONFIG", ""))
if not CONFIG_PATH.is_file():
    raise RuntimeError("workspace_config.json not found. Run the setup cells first.")

with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    workspace_config = json.load(fp)

WORKSPACE_ROOT = Path(workspace_config["workspace_root"])
PATHS = {key: Path(value) for key, value in workspace_config["paths"].items()}

_original_open = builtins.open


def smart_open(file, *args, **kwargs):
    """Resolve relative file paths into the workspace automatically."""
    file_path = Path(file)
    if not file_path.is_absolute():
        for prefix in PATHS.values():
            # If the path already includes a known folder name, leave as-is
            pass
        # Default: put file under workspace root
        file_path = WORKSPACE_ROOT / file_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return _original_open(file_path, *args, **kwargs)


builtins.open = smart_open

print("✅ Smart file handling enabled. Relative paths are rooted at:")
print(f"   {WORKSPACE_ROOT}")
for name, path in PATHS.items():
    print(f"   • {name}: {path}")
