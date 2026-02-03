# cell_03_download_scripts.py
"""
Download Scripts - Fetch the latest modular scripts from GitHub.

This cell downloads ALL modular provider cartridges, schemas, and UI components
from the GitHub repository to your workspace's scripts folder.

Scripts are downloaded dynamically - no hardcoded file list.
"""

import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

import ipywidgets as widgets
from IPython.display import display, clear_output


def _load_paths():
    """Ensure PATHS is populated even if the notebook kernel was restarted."""
    if 'PATHS' in globals() and globals()['PATHS']:
        return {k: Path(v) for k, v in globals()['PATHS'].items()}

    config_path = Path(os.environ.get('WORKSPACE_CONFIG', ''))
    if not config_path.is_file():
        raise RuntimeError("Workspace not configured. Run the workspace setup cell first.")

    with open(config_path, 'r', encoding='utf-8') as fp:
        config = json.load(fp)
    return {k: Path(v) for k, v in config['paths'].items()}


# GitHub repository configuration
GITHUB_REPO = "smartaces/opencite"
GITHUB_BRANCH = "main"
GITHUB_API_BASE = f"https://api.github.com/repos/{GITHUB_REPO}/contents"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}"
SCRIPTS_PATH = "modular_bulk_prompt_runner/github_scripts"

# Subdirectories to download
SCRIPT_SUBDIRS = ["core", "schemas", "providers", "ui", "reports"]


def get_github_files(subdir: str) -> list:
    """Fetch list of files in a GitHub directory using the API.

    Args:
        subdir: Subdirectory name (e.g., "core", "providers")

    Returns:
        List of filenames in that directory
    """
    api_url = f"{GITHUB_API_BASE}/{SCRIPTS_PATH}/{subdir}?ref={GITHUB_BRANCH}"

    try:
        request = Request(api_url)
        request.add_header('Accept', 'application/vnd.github.v3+json')
        request.add_header('User-Agent', 'OpenCite-Downloader')

        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))

        # Filter for .py files only
        files = [item['name'] for item in data if item['type'] == 'file' and item['name'].endswith('.py')]
        return files

    except URLError as e:
        print(f"  Warning: Could not list {subdir}/ via API: {e}")
        return []
    except json.JSONDecodeError:
        print(f"  Warning: Invalid response for {subdir}/")
        return []


def download_scripts(paths: dict, status_output: widgets.Output, force: bool = False) -> bool:
    """Download all modular scripts from GitHub dynamically.

    Args:
        paths: PATHS dict with 'scripts' key
        status_output: Widget output for status messages
        force: If True, overwrite existing files

    Returns:
        True if all downloads succeeded
    """
    scripts_dir = Path(paths["scripts"])
    scripts_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skip_count = 0
    fail_count = 0

    with status_output:
        clear_output()
        print(f"Downloading modular scripts to: {scripts_dir}")
        print(f"Repository: {GITHUB_REPO} (branch: {GITHUB_BRANCH})")
        print(f"\nFetching file lists from GitHub API...\n")

        for subdir in SCRIPT_SUBDIRS:
            subdir_path = scripts_dir / subdir
            subdir_path.mkdir(parents=True, exist_ok=True)

            # Get list of files from GitHub API
            files = get_github_files(subdir)

            if not files:
                print(f"\n{subdir}/ (no files found or API error)")
                continue

            print(f"\n{subdir}/ ({len(files)} files)")

            for filename in files:
                file_path = subdir_path / filename

                # Skip if exists and not forcing
                if file_path.exists() and not force:
                    print(f"  [skip] {filename} (already exists)")
                    skip_count += 1
                    continue

                # Build URL and download
                url = f"{GITHUB_RAW_BASE}/{SCRIPTS_PATH}/{subdir}/{filename}"

                try:
                    with urlopen(url, timeout=30) as response:
                        content = response.read().decode('utf-8')

                    file_path.write_text(content, encoding='utf-8')
                    print(f"  [ok] {filename}")
                    success_count += 1

                except URLError as e:
                    print(f"  [FAIL] {filename}: {e}")
                    fail_count += 1
                except Exception as e:
                    print(f"  [FAIL] {filename}: {e}")
                    fail_count += 1

        print(f"\n{'='*50}")
        print(f"Downloaded: {success_count}")
        print(f"Skipped: {skip_count}")
        print(f"Failed: {fail_count}")

        if fail_count > 0:
            print(f"\nSome downloads failed. Check your internet connection and try again.")
            return False

        # Add scripts directory to Python path
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
            print(f"\nAdded to Python path: {scripts_dir}")

        print(f"\nRun the next cell to set up your search agent.")
        return True


def copy_local_scripts(paths: dict, source_dir: Path, status_output: widgets.Output) -> bool:
    """Copy scripts from a local directory instead of downloading.

    Args:
        paths: PATHS dict with 'scripts' key
        source_dir: Local directory containing the scripts
        status_output: Widget output for status messages

    Returns:
        True if copy succeeded
    """
    import shutil

    scripts_dir = Path(paths["scripts"])
    scripts_dir.mkdir(parents=True, exist_ok=True)

    with status_output:
        clear_output()
        print(f"Copying scripts from: {source_dir}")
        print(f"To: {scripts_dir}\n")

        try:
            for subdir in SCRIPT_SUBDIRS:
                src = source_dir / subdir
                dst = scripts_dir / subdir

                if src.exists():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                    file_count = len(list(dst.glob("*.py")))
                    print(f"  [ok] {subdir}/ ({file_count} files)")
                else:
                    print(f"  [skip] {subdir}/ (not found in source)")

            # Add scripts directory to Python path
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
                print(f"\nAdded to Python path: {scripts_dir}")

            print(f"\nScripts copied successfully!")
            print(f"Run the next cell to set up your search agent.")
            return True

        except Exception as e:
            print(f"Error copying scripts: {e}")
            return False


# =============================================================================
# UI WIDGETS
# =============================================================================

try:
    PATHS = _load_paths()
except RuntimeError as e:
    print(f"Error: {e}")
    PATHS = None

if PATHS:
    status_output = widgets.Output()

    download_button = widgets.Button(
        description="Download from GitHub",
        button_style="primary",
        icon="download",
        layout=widgets.Layout(width="200px"),
    )

    force_checkbox = widgets.Checkbox(
        value=False,
        description="Overwrite existing files",
        style={'description_width': 'auto'},
        layout=widgets.Layout(width="200px"),
    )

    # Local copy option (for development)
    local_source_input = widgets.Text(
        value="",
        placeholder="Path to local github_scripts folder",
        description="Local source:",
        style={'description_width': '100px'},
        layout=widgets.Layout(width="400px"),
    )

    copy_button = widgets.Button(
        description="Copy from Local",
        button_style="",
        icon="copy",
        layout=widgets.Layout(width="150px"),
    )

    def _on_download(_):
        download_scripts(PATHS, status_output, force=force_checkbox.value)

    def _on_copy(_):
        source = local_source_input.value.strip()
        if not source:
            with status_output:
                clear_output()
                print("Enter a local source path first.")
            return
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            with status_output:
                clear_output()
                print(f"Source path not found: {source_path}")
            return
        copy_local_scripts(PATHS, source_path, status_output)

    download_button.on_click(_on_download)
    copy_button.on_click(_on_copy)

    form = widgets.VBox([
        widgets.HTML("<h3>Download Modular Scripts</h3>"),
        widgets.HTML(f"<p>Scripts will be saved to: <code>{PATHS['scripts']}</code></p>"),
        widgets.HTML(f"<p style='color: #666;'>Repository: {GITHUB_REPO} (fetches all .py files dynamically)</p>"),
        widgets.HTML("<hr style='margin: 12px 0;'>"),
        widgets.HTML("<b>Option 1: Download from GitHub</b>"),
        widgets.HBox([download_button, force_checkbox], layout=widgets.Layout(gap='12px')),
        widgets.HTML("<hr style='margin: 12px 0;'>"),
        widgets.HTML("<b>Option 2: Copy from Local (development)</b>"),
        widgets.HBox([local_source_input, copy_button], layout=widgets.Layout(gap='8px')),
        widgets.HTML("<hr style='margin: 12px 0;'>"),
        status_output,
    ])

    display(form)
