import ipywidgets as widgets
from IPython.display import display, clear_output
from urllib.parse import urlparse
from pathlib import Path
import json
import os

if 'PATHS' not in globals():
    config_path = Path(os.environ.get('WORKSPACE_CONFIG', ''))
    if not config_path.is_file():
        raise RuntimeError('Workspace not configured. Run the setup cells first.')
    with open(config_path, 'r', encoding='utf-8') as fp:
        workspace_config = json.load(fp)
    PATHS = {k: Path(v) for k, v in workspace_config['paths'].items()}

if 'search_agent' not in globals():
    raise RuntimeError('Search agent not initialized. Run the agent cell first.')

if 'ReportHelper' not in globals():
    raise RuntimeError('ReportHelper not defined. Run the reporting helper cell first.')

# --- Query & model controls ---
query_input = widgets.Textarea(
    value="What is the best value smartphone in 2026 for $1000",
    description="Query:",
    layout=widgets.Layout(width="100%", height="80px"),
    placeholder="Type your search question..."
)

context_dropdown = widgets.Dropdown(
    options=[("Low", "low"), ("Medium", "medium"), ("High", "high")],
    value="medium",
    description="Context:",
)

reasoning_dropdown = widgets.Dropdown(
    options=[("Low", "low"), ("Medium", "medium"), ("High", "high")],
    value="low",
    description="Reasoning:",
)

# --- Location controls ---
location_toggle = widgets.Checkbox(value=False, description="Apply location bias")

location_presets = widgets.Dropdown(
    options=[
        ("US · New York", "us_ny"),
        ("UK · London", "uk_london"),
        ("US · San Francisco", "us_sf"),
        ("France · Paris", "fr_paris"),
        ("Germany · Berlin", "de_berlin"),
        ("Custom", "custom"),
    ],
    value="us_ny",
    description="Preset:",
    disabled=True,
)

custom_country = widgets.Text(value="US", description="Country:", disabled=True)
custom_city = widgets.Text(value="New York", description="City:", disabled=True)
custom_region = widgets.Text(value="New York", description="Region:", disabled=True)

run_button = widgets.Button(description="Run Search", button_style="primary", icon="search")
output = widgets.Output()

preset_map = {
    "us_ny": ("US", "New York", "New York"),
    "uk_london": ("GB", "London", "London"),
    "us_sf": ("US", "San Francisco", "California"),
    "fr_paris": ("FR", "Paris", "Île-de-France"),
    "de_berlin": ("DE", "Berlin", "Berlin"),
}


def update_location_visibility(change=None):
    enabled = location_toggle.value
    location_presets.disabled = not enabled
    is_custom = enabled and location_presets.value == "custom"
    for widget in (custom_country, custom_city, custom_region):
        widget.disabled = not is_custom


def get_location_dict():
    if not location_toggle.value:
        return None
    if location_presets.value != "custom":
        country, city, region = preset_map[location_presets.value]
    else:
        country = custom_country.value or "US"
        city = custom_city.value or "New York"
        region = custom_region.value or city
    return format_location(country=country, city=city, region=region)


location_toggle.observe(update_location_visibility, "value")
location_presets.observe(update_location_visibility, "value")
update_location_visibility()


def run_single_search(_):
    query = query_input.value.strip()
    with output:
        clear_output()
        if not query:
            print("⚠️ Please enter a query before running the search.")
            return
        user_location = get_location_dict()
        location_note = (
            f"📍 Location bias: {user_location}" if user_location else "📍 Location bias: none"
        )
        reporter = ReportHelper('single_run', PATHS)
        execution_id = reporter.execution_id
        model_name = getattr(search_agent, 'model', 'unknown')
        print(f"▶ Execution ID: {execution_id}")
        print("⏳ Running search, hang tight...\n")
        reporter.add_detail_row(
            unit_id=f"{execution_id}_query",
            turn_or_run=0,
            role='user',
            model=None,
            query_or_topic=query,
            message_text=query,
            citation_rank=None,
            citation_title=None,
            citation_url=None,
            domain=None,
            context=context_dropdown.value,
            reasoning=reasoning_dropdown.value,
            location_country=user_location['country'] if user_location else None,
            location_city=user_location['city'] if user_location else None,
            location_region=user_location['region'] if user_location else None,
            response_file=None,
        )
        response = search_agent.search(
            query,
            search_context_size=context_dropdown.value,
            reasoning_effort=reasoning_dropdown.value,
            verbosity="medium",
            user_location=user_location,
        )
        raw_path = reporter.save_raw_response("run", response)
        text = search_agent.extract_text_response(response)
        citations = search_agent.extract_citations(response) or []
        if citations:
            for rank, cite in enumerate(citations, 1):
                url = cite.get('url', '')
                domain = urlparse(url).netloc.replace('www.', '') if url else ''
                reporter.add_detail_row(
                    unit_id=f"{execution_id}_advisor_{rank}",
                    turn_or_run=1,
                    role='AI System',
                    model=model_name,
                    query_or_topic=query,
                    message_text=text,
                    citation_rank=rank,
                    citation_title=cite.get('title', ''),
                    citation_url=url,
                    domain=domain,
                    context=context_dropdown.value,
                    reasoning=reasoning_dropdown.value,
                    location_country=user_location['country'] if user_location else None,
                    location_city=user_location['city'] if user_location else None,
                    location_region=user_location['region'] if user_location else None,
                    response_file=str(raw_path),
                )
        else:
            reporter.add_detail_row(
                unit_id=f"{execution_id}_advisor_1",
                turn_or_run=1,
                role='AI System',
                model=model_name,
                query_or_topic=query,
                message_text=text,
                citation_rank=None,
                citation_title=None,
                citation_url=None,
                domain=None,
                context=context_dropdown.value,
                reasoning=reasoning_dropdown.value,
                location_country=user_location['country'] if user_location else None,
                location_city=user_location['city'] if user_location else None,
                location_region=user_location['region'] if user_location else None,
                response_file=str(raw_path),
            )
        detail_path = reporter.write_detail_csv()
        summary_row = {
            'model': model_name,
            'query': query,
            'context': context_dropdown.value,
            'reasoning': reasoning_dropdown.value,
            'location_country': user_location['country'] if user_location else None,
            'location_city': user_location['city'] if user_location else None,
            'location_region': user_location['region'] if user_location else None,
            'total_citations': len(citations),
            'unique_citation_urls': len({c.get('url') for c in citations if c.get('url')}) if citations else 0,
            'unique_domains': len({urlparse(c.get('url', '')).netloc.replace('www.', '') for c in citations if c.get('url')}) if citations else 0,
        }
        summary_path = reporter.write_summary_csv(summary_row)
        clear_output()
        print(f"🔍 Query: {query}\n{location_note}\n")
        print_search_result(response)
        print(f"\n💾 Detail CSV: {detail_path}")
        print(f"💾 Summary CSV: {summary_path}")


run_button.on_click(run_single_search)

location_box = widgets.VBox([
    location_toggle,
    location_presets,
    widgets.HBox([custom_country, custom_city, custom_region]),
])

controls = widgets.VBox([
    query_input,
    widgets.HBox([context_dropdown, reasoning_dropdown]),
    location_box,
    run_button,
])

display(widgets.VBox([controls, output]))
