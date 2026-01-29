# =============================================================================
# CELL 7: BROWSE RESULTS
# =============================================================================
# Browse and filter your batch run results.
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

# Load Results Viewer script
results_viewer_script = scripts_path / "cell_07_openrouter_results_viewer.py"
if results_viewer_script.exists():
    exec(compile(open(results_viewer_script, encoding='utf-8').read(), results_viewer_script, 'exec'))
else:
    raise RuntimeError(f"Results viewer script not found: {results_viewer_script}")
