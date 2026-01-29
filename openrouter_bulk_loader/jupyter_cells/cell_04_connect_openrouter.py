# =============================================================================
# CELL 4: CONNECT TO OPENROUTER
# =============================================================================
# This cell loads the core scripts and connects to the OpenRouter API.
# It will display a model selector UI when complete.
# =============================================================================

import json
import os
import subprocess
import sys
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

# 1. Install dependencies (silent)
print("Installing dependencies...")
pip_script = scripts_path / "cell_00_openrouter_pip_installs.py"
if pip_script.exists():
    # Run pip install silently
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "openai", "rich", "python-dotenv", "ipywidgets", "plotly", "posthog"],
        capture_output=True
    )
    print("Dependencies installed.")
else:
    print("Warning: pip install script not found, attempting to continue...")

# 2. Load API key configuration
print("Loading API key...")
api_key_script = scripts_path / "cell_02_openrouter_api_key.py"
if api_key_script.exists():
    exec(compile(open(api_key_script, encoding='utf-8').read(), api_key_script, 'exec'))
else:
    raise RuntimeError(f"API key script not found: {api_key_script}")

# 3. Load ReportHelper class
print("Loading ReportHelper...")
report_helper_script = scripts_path / "cell_03_openrouter_report_helper.py"
if report_helper_script.exists():
    exec(compile(open(report_helper_script, encoding='utf-8').read(), report_helper_script, 'exec'))
else:
    raise RuntimeError(f"Report helper script not found: {report_helper_script}")

# 4. Load Dataset Builder (for filter helpers used by reports)
print("Loading Dataset Builder...")
dataset_builder_script = scripts_path / "cell_08_openrouter_dataset_builder.py"
if dataset_builder_script.exists():
    exec(compile(open(dataset_builder_script, encoding='utf-8').read(), dataset_builder_script, 'exec'))
else:
    print("Warning: Dataset builder script not found, reports may have limited functionality.")

# 5. Load Search Agent (displays model selector UI)
print("Loading Search Agent...")
search_agent_script = scripts_path / "cell_04_openrouter_search_agent.py"
if search_agent_script.exists():
    exec(compile(open(search_agent_script, encoding='utf-8').read(), search_agent_script, 'exec'))
else:
    raise RuntimeError(f"Search agent script not found: {search_agent_script}")

print("\nOpenRouter connection ready!")
print("Run the next cell to upload your prompts CSV.")
