# =============================================================================
# CELL 3: DOWNLOAD SCRIPTS FROM GITHUB
# =============================================================================
# Downloads all cell_*.py scripts from the GitHub repository to your
# workspace/scripts/ folder. Run this after setting up your workspace.
# =============================================================================

import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

import ipywidgets as widgets
from IPython.display import display, clear_output

# GitHub repository configuration
GITHUB_REPO = "smartaces/opencite"
GITHUB_BRANCH = "main"
GITHUB_FOLDER = "openrouter_bulk_loader/github_scripts"

# Scripts to download
SCRIPT_FILES = [
    "cell_00_openrouter_pip_installs.py",
    "cell_02_openrouter_api_key.py",
    "cell_03_openrouter_report_helper.py",
    "cell_04_openrouter_search_agent.py",
    "cell_05_openrouter_csv_loader.py",
    "cell_06_openrouter_batch_runner.py",
    "cell_07_openrouter_results_viewer.py",
    "cell_08_openrouter_dataset_builder.py",
    "cell_09_openrouter_domain_report.py",
    "cell_10_openrouter_page_report.py",
    "cell_11_openrouter_prompt_report.py",
]


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


def _get_github_raw_url(filename: str) -> str:
    """Build GitHub raw content URL for a file."""
    return f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_FOLDER}/{filename}"


def _download_file(url: str, dest_path: Path) -> bool:
    """Download a file from URL to destination path."""
    try:
        req = Request(url, headers={'User-Agent': 'OpenCite-Bulk-Loader/1.0'})
        with urlopen(req, timeout=30) as response:
            content = response.read()
        dest_path.write_bytes(content)
        return True
    except URLError as e:
        print(f"Failed to download {url}: {e}")
        return False
    except Exception as e:
        print(f"Error saving {dest_path.name}: {e}")
        return False


def _get_file_hash(path: Path) -> str:
    """Get MD5 hash of a file."""
    if not path.exists():
        return ""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _download_all_scripts(scripts_path: Path, force: bool = False) -> dict:
    """Download all scripts, optionally forcing re-download."""
    results = {"downloaded": [], "skipped": [], "failed": []}

    scripts_path.mkdir(parents=True, exist_ok=True)

    for filename in SCRIPT_FILES:
        dest_path = scripts_path / filename
        url = _get_github_raw_url(filename)

        if dest_path.exists() and not force:
            # Check if remote is different
            try:
                req = Request(url, headers={'User-Agent': 'OpenCite-Bulk-Loader/1.0'})
                with urlopen(req, timeout=30) as response:
                    remote_content = response.read()
                remote_hash = hashlib.md5(remote_content).hexdigest()
                local_hash = _get_file_hash(dest_path)

                if remote_hash == local_hash:
                    results["skipped"].append(filename)
                    continue
                else:
                    # Update needed
                    dest_path.write_bytes(remote_content)
                    results["downloaded"].append(filename)
            except Exception:
                results["skipped"].append(filename)
                continue
        else:
            # Download new file
            if _download_file(url, dest_path):
                results["downloaded"].append(filename)
            else:
                results["failed"].append(filename)

    return results


# UI Components
scripts_path = _get_scripts_path()
status_output = widgets.Output()

check_button = widgets.Button(
    description="Check for Updates",
    button_style="info",
    icon="refresh",
    layout=widgets.Layout(width="180px"),
)

force_button = widgets.Button(
    description="Force Re-download All",
    button_style="warning",
    icon="download",
    layout=widgets.Layout(width="180px"),
)


def _handle_check(_):
    with status_output:
        clear_output()
        print("Checking for updates...")
        results = _download_all_scripts(scripts_path, force=False)
        print(f"\nResults:")
        if results["downloaded"]:
            print(f"  Updated: {len(results['downloaded'])} files")
            for f in results["downloaded"]:
                print(f"    - {f}")
        if results["skipped"]:
            print(f"  Up to date: {len(results['skipped'])} files")
        if results["failed"]:
            print(f"  Failed: {len(results['failed'])} files")
            for f in results["failed"]:
                print(f"    - {f}")
        print(f"\nScripts location: {scripts_path}")


def _handle_force(_):
    with status_output:
        clear_output()
        print("Force downloading all scripts...")
        results = _download_all_scripts(scripts_path, force=True)
        print(f"\nResults:")
        print(f"  Downloaded: {len(results['downloaded'])} files")
        if results["failed"]:
            print(f"  Failed: {len(results['failed'])} files")
            for f in results["failed"]:
                print(f"    - {f}")
        print(f"\nScripts location: {scripts_path}")


check_button.on_click(_handle_check)
force_button.on_click(_handle_force)

# Layout
controls = widgets.VBox([
    widgets.HTML("<h3>Download Scripts</h3>"),
    widgets.HTML("<p>Download or update the OpenCite scripts from GitHub.</p>"),
    widgets.HTML(f"<p><b>Scripts folder:</b> {scripts_path}</p>"),
    widgets.HBox([check_button, force_button], layout=widgets.Layout(gap='8px')),
    status_output,
])

display(controls)

# Auto-download on first run
print("Downloading scripts...")
results = _download_all_scripts(scripts_path, force=False)
if results["downloaded"]:
    print(f"Downloaded {len(results['downloaded'])} scripts")
if results["skipped"]:
    print(f"Already up to date: {len(results['skipped'])} scripts")
if results["failed"]:
    print(f"Failed to download: {results['failed']}")
print(f"\nScripts ready in: {scripts_path}")
print("Run the next cell to connect to OpenRouter.")
