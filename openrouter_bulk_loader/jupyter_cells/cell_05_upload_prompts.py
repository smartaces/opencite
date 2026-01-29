# =============================================================================
# CELL 5: UPLOAD PROMPTS
# =============================================================================
# Upload your CSV file with prompts, personas, and run/turn configurations.
# =============================================================================

import json
import os
from pathlib import Path

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

# Load CSV Loader script
csv_loader_script = scripts_path / "cell_05_openrouter_csv_loader.py"
if csv_loader_script.exists():
    exec(compile(open(csv_loader_script, encoding='utf-8').read(), csv_loader_script, 'exec'))
else:
    raise RuntimeError(f"CSV loader script not found: {csv_loader_script}")
