import json
import os
from pathlib import Path

if "WORKSPACE_ROOT" in globals():
    workspace_root = Path(WORKSPACE_ROOT)
else:
    env_root = os.environ.get("WORKSPACE_ROOT")
    if not env_root:
        raise RuntimeError("Workspace not configured. Run the previous cell first to select a location.")
    workspace_root = Path(env_root)

workspace_root = workspace_root.expanduser().resolve()
os.environ["WORKSPACE_ROOT"] = str(workspace_root)

subfolders = {
    "search_results": workspace_root / "search_results",
    "extracted_raw": workspace_root / "extracted_raw",
    "csv_output": workspace_root / "csv_output",
    "grabbed": workspace_root / "grabbed",
    "terms_lists": workspace_root / "terms_lists",
    "logs": workspace_root / "logs"
}

for path in subfolders.values():
    path.mkdir(parents=True, exist_ok=True)

PATHS = {name: str(path) for name, path in subfolders.items()}

config = {
    "workspace_root": str(workspace_root),
    "paths": PATHS
}

config_path = workspace_root / "workspace_config.json"
with open(config_path, "w", encoding="utf-8") as fp:
    json.dump(config, fp, indent=2)

os.environ["WORKSPACE_CONFIG"] = str(config_path)

print(f"📁 Workspace root: {workspace_root}")
for name, path in PATHS.items():
    print(f"  • {name}: {path}")
print(f"🗂️ Config saved to: {config_path}")
print("ℹ️ Later cells: from pathlib import Path; PATHS.get('csv_output') etc.")
