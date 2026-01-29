# =============================================================================
# CELL 6: RUN BATCH
# =============================================================================
# Execute your prompts through the OpenRouter search agent.
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

# Load Batch Runner script
batch_runner_script = scripts_path / "cell_06_openrouter_batch_runner.py"
if batch_runner_script.exists():
    exec(compile(open(batch_runner_script, encoding='utf-8').read(), batch_runner_script, 'exec'))
else:
    raise RuntimeError(f"Batch runner script not found: {batch_runner_script}")
