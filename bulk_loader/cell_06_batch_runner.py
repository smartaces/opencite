import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import ipywidgets as widgets
import pandas as pd
from IPython.display import display, clear_output

PERSONA_MODEL = "gpt-4o-mini"


def _ensure_paths():
    if 'PATHS' in globals():
        return {k: Path(v) for k, v in PATHS.items()}

    config_path = Path(os.environ.get('WORKSPACE_CONFIG', ''))
    if not config_path.is_file():
        raise RuntimeError("Workspace not configured. Run the workspace setup cell first.")

    with open(config_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return {k: Path(v) for k, v in data['paths'].items()}


PATHS = _ensure_paths()
TERMS_DIR = Path(PATHS['terms_lists'])

if 'search_agent' not in globals():
    raise RuntimeError("Search agent not initialized. Run the OpenAI agent cell first.")

if 'ReportHelper' not in globals():
    raise RuntimeError("ReportHelper missing. Run the reporting helper cell before this one.")

if 'format_location' not in globals():
    raise RuntimeError("Location helper missing. Run the location helper cell before this one.")

if 'OpenAISearchAgent' not in globals():
    raise RuntimeError("OpenAISearchAgent class not found. Run the OpenAI agent cell first.")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _list_csv_files():
    TERMS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(TERMS_DIR).glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in ["prompt", "persona", "runs", "turns"] if col not in df.columns]
    if missing:
        raise ValueError(f"The selected CSV is missing required columns: {missing}. Re-run Cell 12 to regenerate it.")
    return df


def _validate_persona_text(value: str) -> str:
    value = (value or "").strip()
    if len(value) > 250:
        raise ValueError("Persona/audience descriptions must be 250 characters or fewer.")
    return value


def _coerce_persona(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _convert_positive(value, fallback: int = 1) -> int:
    """Convert value to positive integer, defaulting to fallback (1) if blank/invalid."""
    if pd.isna(value) or str(value).strip() == "":
        return fallback
    try:
        number = int(float(value))
    except (ValueError, TypeError):
        return fallback
    return max(1, number)


def _simulate_persona_message(client, persona_profile: str, history: list, topic: str) -> str:
    """Simulate a persona follow-up message. Uses provided client for thread safety."""
    prompt = (
        "You are role-playing as the user described as: "
        + persona_profile
        + ". You are conversing with an AI assistant about: "
        + topic
        + ". Read the assistant's latest reply and respond naturally as this user would—"
          "acknowledging what you just learned, sharing reactions or preferences, ALWAYS REMEMBER YOU ARE THE SHOPPER OR BUYER OR POTENTIAL CUSTOMER"
          " or asking a follow-up question that moves the conversation forward. If the conversation is about a product, service, company or brand, seek to get relevant further information to inform your perception, understanding, knowledge or buying decision. Ask pertinent questions, that would enable you to make a good choice."
          " Stay in character and never interview the real user. Respond with a single message. Focus on learning more about the product or service or company as part of a buying research and consideration process. Keep your queries and follow ons relatively succinct - don't be verbose - or over-imagine - reflect the tone of someone conducting web research via an llm search assistant to get the info you need e.g. Instead of saying something like: It sounds like the battery life is decent, but I'm a bit concerned about how it will hold up during heavy use. Say something like: how long does the battery last for a heavy user? "
    )
    response = client.responses.create(
        model=PERSONA_MODEL,
        input=[{"role": "system", "content": prompt}] + history,
    )
    # Use the global search_agent just for the extract method (stateless)
    return search_agent.extract_text_response(response).strip()


def _get_location_dict():
    if not location_toggle.value:
        return None
    if location_presets.value != "custom":
        country, city, region = preset_locations[location_presets.value]
        return format_location(country=country, city=city, region=region)
    country = custom_country.value.strip() or "US"
    city = custom_city.value.strip() or "New York"
    region = custom_region.value.strip() or city
    return format_location(country=country, city=city, region=region)


def _create_thread_local_agent():
    """Create a fresh search agent instance for thread-safe parallel execution."""
    model = getattr(search_agent, "model", "gpt-5.2")
    client = search_agent.client  # Share the client (it's thread-safe)
    return OpenAISearchAgent(client, model=model)


# ============================================================================
# WIDGET STYLING
# ============================================================================

STYLE = {'description_width': '100px'}
DROPDOWN_LAYOUT = widgets.Layout(width='420px')
TEXT_LAYOUT = widgets.Layout(width='420px')


# ============================================================================
# CSV SELECTION WIDGETS
# ============================================================================

csv_files = _list_csv_files()
csv_dropdown = widgets.Dropdown(
    options=[(file.name, str(file)) for file in csv_files] or [("No CSV files found", "")],
    value=str(csv_files[0]) if csv_files else "",
    description="CSV file:",
    style=STYLE,
    layout=DROPDOWN_LAYOUT,
)
refresh_button = widgets.Button(description="Refresh", icon="refresh", layout=widgets.Layout(width='90px'))
csv_info_output = widgets.Output()
_csv_state = {"path": None, "df": None}


# ============================================================================
# RUN LABEL WIDGET
# ============================================================================

run_label_input = widgets.Text(
    value="",
    description="Run label:",
    placeholder="Optional tag (auto-generated if blank)",
    style=STYLE,
    layout=TEXT_LAYOUT,
)


# ============================================================================
# SEARCH DEPTH WIDGETS
# ============================================================================

context_dropdown = widgets.Dropdown(
    options=[
        ("Low – Minimal web content, fastest", "low"),
        ("Medium – Balanced retrieval (recommended)", "medium"),
        ("High – Extensive web content, most thorough", "high"),
    ],
    value="medium",
    description="Level:",
    style=STYLE,
    layout=DROPDOWN_LAYOUT,
)


# ============================================================================
# REASONING EFFORT WIDGETS
# ============================================================================

reasoning_dropdown = widgets.Dropdown(
    options=[
        ("Low – Quick responses, minimal deliberation", "low"),
        ("Medium – Balanced thinking", "medium"),
        ("High – Deep analysis, slower but more nuanced", "high"),
    ],
    value="low",
    description="Level:",
    style=STYLE,
    layout=DROPDOWN_LAYOUT,
)


# ============================================================================
# PARALLEL RUNS WIDGETS
# ============================================================================

parallel_dropdown = widgets.Dropdown(
    options=[
        ("1 – Sequential (no parallelism)", 1),
        ("2 – Run 2 queries at once", 2),
        ("3 – Run 3 queries at once (recommended)", 3),
        ("4 – Run 4 queries at once", 4),
        ("5 – Run 5 queries at once (fastest)", 5),
    ],
    value=3,
    description="Batch size:",
    style=STYLE,
    layout=DROPDOWN_LAYOUT,
)


# ============================================================================
# LOCATION BIAS WIDGETS
# ============================================================================

location_toggle = widgets.Checkbox(
    value=False,
    description="Enable location bias",
    style={'description_width': 'auto'},
    layout=widgets.Layout(width='200px'),
)

location_presets = widgets.Dropdown(
    options=[
        ("US · New York", "us_ny"),
        ("UK · London", "uk_london"),
        ("US · San Francisco", "us_sf"),
        ("France · Paris", "fr_paris"),
        ("Germany · Berlin", "de_berlin"),
        ("Custom location", "custom"),
    ],
    value="uk_london",
    description="Location:",
    style=STYLE,
    layout=DROPDOWN_LAYOUT,
    disabled=True,
)

custom_country = widgets.Text(value="", description="Country:", placeholder="e.g. GB", style={'description_width': '60px'}, layout=widgets.Layout(width='130px'), disabled=True)
custom_city = widgets.Text(value="", description="City:", placeholder="e.g. London", style={'description_width': '40px'}, layout=widgets.Layout(width='150px'), disabled=True)
custom_region = widgets.Text(value="", description="Region:", placeholder="e.g. England", style={'description_width': '50px'}, layout=widgets.Layout(width='160px'), disabled=True)

preset_locations = {
    "us_ny": ("US", "New York", "New York"),
    "uk_london": ("GB", "London", "England"),
    "us_sf": ("US", "San Francisco", "California"),
    "fr_paris": ("FR", "Paris", "Île-de-France"),
    "de_berlin": ("DE", "Berlin", "Berlin"),
}


# ============================================================================
# RUN BUTTON & OUTPUT
# ============================================================================

run_button = widgets.Button(
    description="Run Batch",
    button_style="primary",
    icon="play",
    layout=widgets.Layout(width='120px'),
)
output = widgets.Output()


# ============================================================================
# CSV LOADING & SUMMARY
# ============================================================================

def _refresh_csv_list(_=None):
    files = _list_csv_files()
    if files:
        csv_dropdown.options = [(file.name, str(file)) for file in files]
        csv_dropdown.value = str(files[0])
    else:
        csv_dropdown.options = [("No CSV files found", "")]
        csv_dropdown.value = ""
    _load_selected_csv()


def _analyze_numeric_column(series: pd.Series) -> dict:
    blank = 0
    invalid = 0
    valid = 0
    total_value = 0
    for raw in series:
        if pd.isna(raw) or str(raw).strip() == "":
            blank += 1
            continue
        try:
            number = int(float(raw))
        except (ValueError, TypeError):
            invalid += 1
            continue
        if number >= 1:
            valid += 1
            total_value += number
        else:
            invalid += 1
    return {"valid": valid, "blank": blank, "invalid": invalid, "total": total_value}


def _analyze_persona(series: pd.Series) -> dict:
    text = series.fillna("").astype(str).str.strip()
    blank = (text == "").sum()
    return {"filled": len(series) - blank, "blank": int(blank)}


def _update_csv_summary(df: pd.DataFrame):
    persona_stats = _analyze_persona(df["persona"])
    runs_stats = _analyze_numeric_column(df["runs"])
    turns_stats = _analyze_numeric_column(df["turns"])

    with csv_info_output:
        clear_output()

        lines = [f"✓ Loaded {len(df)} prompts"]

        if persona_stats['filled'] == len(df):
            lines.append(f"✓ Personas: all {persona_stats['filled']} filled")
        elif persona_stats['filled'] == 0:
            lines.append(f"⚠ Personas: all blank (will run without persona)")
        else:
            lines.append(f"✓ Personas: {persona_stats['filled']} filled, {persona_stats['blank']} blank")

        if runs_stats['blank'] == 0 and runs_stats['invalid'] == 0:
            lines.append(f"✓ Runs: all valid (total: {runs_stats['total']} runs across all prompts)")
        else:
            parts = [f"{runs_stats['valid']} valid"]
            if runs_stats['blank'] > 0:
                parts.append(f"{runs_stats['blank']} blank → fallback to 1")
            if runs_stats['invalid'] > 0:
                parts.append(f"{runs_stats['invalid']} invalid → fallback to 1")
            lines.append(f"⚠ Runs: {', '.join(parts)}")

        if turns_stats['blank'] == 0 and turns_stats['invalid'] == 0:
            lines.append(f"✓ Turns: all valid (total: {turns_stats['total']} turns across all prompts)")
        else:
            parts = [f"{turns_stats['valid']} valid"]
            if turns_stats['blank'] > 0:
                parts.append(f"{turns_stats['blank']} blank → fallback to 1")
            if turns_stats['invalid'] > 0:
                parts.append(f"{turns_stats['invalid']} invalid → fallback to 1")
            lines.append(f"⚠ Turns: {', '.join(parts)}")

        for line in lines:
            print(line)


def _display_csv_message(message: str):
    with csv_info_output:
        clear_output()
        print(message)


def _load_selected_csv(change=None):
    path = csv_dropdown.value
    if not path:
        _csv_state["path"] = None
        _csv_state["df"] = None
        _display_csv_message("⚠ No CSV selected")
        return None
    try:
        df = _load_csv(Path(path))
    except Exception as exc:
        _csv_state["path"] = None
        _csv_state["df"] = None
        _display_csv_message(f"✗ Failed to load: {exc}")
        return None
    _csv_state["path"] = path
    _csv_state["df"] = df
    _update_csv_summary(df)
    return df


def _ensure_dataframe():
    path = csv_dropdown.value
    if _csv_state["df"] is not None and _csv_state["path"] == path:
        return _csv_state["df"].copy()
    return _load_selected_csv()


# ============================================================================
# LOCATION WIDGET UPDATES
# ============================================================================

def _update_location_widgets(change=None):
    enabled = location_toggle.value
    location_presets.disabled = not enabled
    is_custom = enabled and location_presets.value == "custom"
    for w in (custom_country, custom_city, custom_region):
        w.disabled = not is_custom


# ============================================================================
# SINGLE RUN EXECUTION (used by both sequential and parallel modes)
# ============================================================================

def _execute_single_run(
    run_idx: int,
    total_runs: int,
    topic: str,
    persona_profile: str,
    turns: int,
    context_level: str,
    reasoning_level: str,
    user_location: dict,
    run_label: str,
    use_local_agent: bool = False,
) -> dict:
    """
    Execute a single run (all turns for one run).
    Returns a dict with results for logging.

    If use_local_agent is True, creates a fresh search agent instance
    for thread-safe parallel execution.
    """
    # Create thread-local agent for parallel execution, or use global for sequential
    if use_local_agent:
        local_agent = _create_thread_local_agent()
    else:
        local_agent = search_agent

    model_name = getattr(local_agent, "model", "unknown")
    persona_model_name = PERSONA_MODEL
    log_lines = []

    reporter = ReportHelper("multi_turn_batch", PATHS, run_label=run_label)
    execution_id = reporter.execution_id
    log_lines.append(f"\n▶ Run {run_idx}/{total_runs} — ID: {execution_id}")

    persona_history = []
    advisor_history = []

    for turn in range(1, turns + 1):
        turn_id = f"{execution_id}_turn_{turn}"
        if turn == 1:
            persona_msg = topic
            log_lines.append(f"  Turn {turn}/{turns} — Initial: {persona_msg}")
        else:
            persona_msg = _simulate_persona_message(
                local_agent.client,
                persona_profile,
                persona_history + advisor_history,
                topic
            )
            log_lines.append(f"  Turn {turn}/{turns} — Persona: {persona_msg}")
        persona_history.append({"role": "user", "content": persona_msg})

        reporter.add_detail_row(
            unit_id=turn_id,
            turn_or_run=turn,
            role="persona",
            model=persona_model_name,
            query_or_topic=topic,
            message_text=persona_msg,
            citation_rank=None,
            citation_title=None,
            citation_url=None,
            domain=None,
            context=context_level,
            reasoning=reasoning_level,
            location_country=user_location['country'] if user_location else None,
            location_city=user_location['city'] if user_location else None,
            location_region=user_location['region'] if user_location else None,
            response_file=None,
            persona_profile=persona_profile,
            persona_model=persona_model_name,
        )

        log_lines.append(f"  Querying...")
        try:
            response = local_agent.search(
                query=persona_msg,
                search_context_size=context_level,
                reasoning_effort=reasoning_level,
                verbosity="medium",
                user_location=user_location,
                use_previous_reasoning=True,  # Enable within this run's agent
            )
        except Exception as exc:
            log_lines.append(f"  ✗ Search failed: {exc}")
            return {"success": False, "log": log_lines, "error": str(exc)}

        raw_path = reporter.save_raw_response(f"turn_{turn}", response)
        advisor_text = local_agent.extract_text_response(response)
        advisor_history.append({"role": "assistant", "content": advisor_text})

        # Log response summary (truncated for readability)
        response_preview = advisor_text[:200] + "..." if len(advisor_text) > 200 else advisor_text
        log_lines.append(f"  Response: {response_preview}")

        citations = local_agent.extract_citations(response)
        if citations:
            log_lines.append(f"  Citations: {len(citations)} found")
            for rank, cite in enumerate(citations, 1):
                url = cite.get("url", "")
                parsed = urlparse(url) if url else None
                domain = parsed.netloc.replace("www.", "") if parsed and parsed.netloc else None
                reporter.add_detail_row(
                    unit_id=turn_id,
                    turn_or_run=turn,
                    role="AI System",
                    model=model_name,
                    query_or_topic=topic,
                    message_text=advisor_text,
                    citation_rank=rank,
                    citation_title=cite.get("title", ""),
                    citation_url=url,
                    domain=domain,
                    context=context_level,
                    reasoning=reasoning_level,
                    location_country=user_location['country'] if user_location else None,
                    location_city=user_location['city'] if user_location else None,
                    location_region=user_location['region'] if user_location else None,
                    response_file=str(raw_path),
                    persona_profile=persona_profile,
                    persona_model=persona_model_name,
                )
        else:
            log_lines.append(f"  Citations: none")
            reporter.add_detail_row(
                unit_id=turn_id,
                turn_or_run=turn,
                role="AI System",
                model=model_name,
                query_or_topic=topic,
                message_text=advisor_text,
                citation_rank=None,
                citation_title=None,
                citation_url=None,
                domain=None,
                context=context_level,
                reasoning=reasoning_level,
                location_country=user_location['country'] if user_location else None,
                location_city=user_location['city'] if user_location else None,
                location_region=user_location['region'] if user_location else None,
                response_file=str(raw_path),
                persona_profile=persona_profile,
                persona_model=persona_model_name,
            )

    detail_path = reporter.write_detail_csv()
    df = pd.DataFrame(reporter._detail_rows)
    advisor_citations = df[df["role"] == "AI System"].dropna(subset=["citation_url"])
    domain_counts = advisor_citations["domain"].value_counts().head(5)

    summary_row = {
        "model": model_name,
        "topic": topic,
        "turns": turns,
        "persona_profile": persona_profile,
        "persona_model": persona_model_name,
        "location_country": user_location['country'] if user_location else None,
        "location_city": user_location['city'] if user_location else None,
        "location_region": user_location['region'] if user_location else None,
        "total_citations": len(advisor_citations),
        "unique_citation_urls": advisor_citations["citation_url"].nunique(),
        "unique_domains": advisor_citations["domain"].nunique(),
    }

    for idx, (domain, count) in enumerate(domain_counts.items(), start=1):
        summary_row[f"top_domain_{idx}"] = domain
        summary_row[f"top_domain_{idx}_count"] = count

    summary_path = reporter.write_summary_csv(summary_row)
    log_lines.append(f"  💾 Detail: {detail_path.name}")
    log_lines.append(f"  💾 Summary: {summary_path.name}")

    return {"success": True, "log": log_lines, "detail_path": detail_path, "summary_path": summary_path}


# ============================================================================
# BATCH EXECUTION (supports both sequential and parallel)
# ============================================================================

def _execute_runs_for_prompt(
    prompt_idx: int,
    total_prompts: int,
    topic: str,
    persona_profile: str,
    turns: int,
    runs: int,
    context_level: str,
    reasoning_level: str,
    user_location: dict,
    run_label: str,
    max_workers: int,
) -> dict:
    """
    Execute all runs for a single prompt.
    If max_workers > 1, runs execute in parallel with thread-local agents.
    """
    log_lines = []
    log_lines.append(f"\n[{prompt_idx}/{total_prompts}] {topic[:50]}{'...' if len(topic) > 50 else ''}")
    log_lines.append(f"    Persona: {persona_profile[:40] if persona_profile else '(none)'}{'...' if len(persona_profile) > 40 else ''}")
    log_lines.append(f"    Runs: {runs} | Turns: {turns}")

    if runs == 1 or max_workers == 1:
        # Sequential execution - use global agent
        successes = 0
        for run_idx in range(1, runs + 1):
            result = _execute_single_run(
                run_idx=run_idx,
                total_runs=runs,
                topic=topic,
                persona_profile=persona_profile,
                turns=turns,
                context_level=context_level,
                reasoning_level=reasoning_level,
                user_location=user_location,
                run_label=run_label,
                use_local_agent=False,  # Use global agent
            )
            log_lines.extend(result["log"])
            if result["success"]:
                successes += 1
        return {"success": successes == runs, "successes": successes, "total": runs, "log": log_lines}
    else:
        # Parallel execution - each run gets its own agent
        log_lines.append(f"    ⚡ Running {runs} runs in parallel (max {max_workers} workers)")
        successes = 0

        with ThreadPoolExecutor(max_workers=min(max_workers, runs)) as executor:
            futures = {}
            for run_idx in range(1, runs + 1):
                future = executor.submit(
                    _execute_single_run,
                    run_idx=run_idx,
                    total_runs=runs,
                    topic=topic,
                    persona_profile=persona_profile,
                    turns=turns,
                    context_level=context_level,
                    reasoning_level=reasoning_level,
                    user_location=user_location,
                    run_label=run_label,
                    use_local_agent=True,  # Create fresh agent per thread
                )
                futures[future] = run_idx

            for future in as_completed(futures):
                run_idx = futures[future]
                try:
                    result = future.result()
                    log_lines.extend(result["log"])
                    if result["success"]:
                        successes += 1
                except Exception as exc:
                    log_lines.append(f"\n  ✗ Run {run_idx} failed with exception: {exc}")

        return {"success": successes == runs, "successes": successes, "total": runs, "log": log_lines}


def _run_batch(_):
    df = _ensure_dataframe()
    if df is None or df.empty:
        with output:
            clear_output()
            print("⚠ Load a CSV before running.")
        return

    total_rows = len(df)
    successful_prompts = 0
    skipped = []
    user_location = _get_location_dict()
    user_label = run_label_input.value.strip()
    default_label = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_label = user_label or default_label
    max_workers = parallel_dropdown.value

    with output:
        clear_output()
        print(f"🚀 Batch: {total_rows} prompts from {Path(csv_dropdown.value).name}")
        print(f"   Search: {context_dropdown.value} | Reasoning: {reasoning_dropdown.value}")
        if max_workers > 1:
            print(f"   ⚡ Parallel: up to {max_workers} runs simultaneously")
        else:
            print(f"   Sequential: 1 run at a time")
        if user_location:
            print(f"   📍 Location: {user_location['city']}, {user_location['country']}")
        print(f"   🏷️ Label: {run_label}")
        print("=" * 60)

        for idx, row in df.iterrows():
            topic = str(row.get("prompt", "") or "").strip()
            if not topic:
                skipped.append((idx + 1, "Missing prompt"))
                continue

            try:
                persona_profile = _validate_persona_text(_coerce_persona(row.get("persona", "")))
            except ValueError as exc:
                skipped.append((idx + 1, str(exc)))
                continue

            turns = _convert_positive(row.get("turns"), fallback=1)
            runs = _convert_positive(row.get("runs"), fallback=1)

            result = _execute_runs_for_prompt(
                prompt_idx=idx + 1,
                total_prompts=total_rows,
                topic=topic,
                persona_profile=persona_profile,
                turns=turns,
                runs=runs,
                context_level=context_dropdown.value,
                reasoning_level=reasoning_dropdown.value,
                user_location=user_location,
                run_label=run_label,
                max_workers=max_workers,
            )

            # Print logs
            for line in result["log"]:
                print(line)

            if result["success"]:
                successful_prompts += 1
            else:
                skipped.append((idx + 1, f"Partial failure: {result['successes']}/{result['total']} runs succeeded"))

        print("\n" + "=" * 60)
        print(f"✓ Complete: {successful_prompts}/{total_rows} prompts fully successful")
        if skipped:
            print(f"✗ Skipped/failed:")
            for row_num, reason in skipped:
                print(f"   Row {row_num}: {reason}")


# ============================================================================
# EVENT BINDINGS
# ============================================================================

refresh_button.on_click(_refresh_csv_list)
csv_dropdown.observe(_load_selected_csv, names="value")
location_toggle.observe(_update_location_widgets, names="value")
location_presets.observe(_update_location_widgets, names="value")
run_button.on_click(_run_batch)

_update_location_widgets()
_load_selected_csv()


# ============================================================================
# LAYOUT
# ============================================================================

controls = widgets.VBox([
    widgets.HTML("<h3>Cell 13 · Multi-turn Batch Runner</h3>"),
    widgets.HTML("<p>Run your prompts through the AI search assistant and save the results.</p>"),

    widgets.HTML("<hr style='margin: 8px 0;'>"),
    widgets.HTML("<b>Prompt file</b>"),
    widgets.HBox([csv_dropdown, refresh_button], layout=widgets.Layout(gap='8px', align_items='center')),
    csv_info_output,

    widgets.HTML("<hr style='margin: 12px 0;'>"),
    widgets.HTML("<b>Run label</b> — Optional tag for organising output files"),
    run_label_input,

    widgets.HTML("<hr style='margin: 12px 0;'>"),
    widgets.HTML("<b>Search depth</b> — Controls how much web content is retrieved per query"),
    context_dropdown,

    widgets.HTML("<hr style='margin: 12px 0;'>"),
    widgets.HTML("<b>Reasoning effort</b> — Controls how deeply the model analyses content before responding"),
    reasoning_dropdown,

    widgets.HTML("<hr style='margin: 12px 0;'>"),
    widgets.HTML("<b>Parallel runs</b> — Run multiple queries simultaneously for faster processing"),
    parallel_dropdown,

    widgets.HTML("<hr style='margin: 12px 0;'>"),
    widgets.HTML("<b>Location bias</b> — Personalises results as if searching from this location"),
    location_toggle,
    location_presets,
    widgets.HBox([custom_country, custom_city, custom_region], layout=widgets.Layout(gap='8px')),

    widgets.HTML("<hr style='margin: 12px 0;'>"),
    run_button,
    output,
])

display(controls)