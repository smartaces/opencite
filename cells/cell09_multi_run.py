import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
import json
import os
import re

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

query_input_multi = widgets.Textarea(
    value="Compare Samsung S25 vs Google Pixel",
    description="Query:",
    layout=widgets.Layout(width="100%", height="80px"),
)

context_dropdown_multi = widgets.Dropdown(
    options=[("Low", "low"), ("Medium", "medium"), ("High", "high")],
    value="medium",
    description="Context:",
)

reasoning_dropdown_multi = widgets.Dropdown(
    options=[("Low", "low"), ("Medium", "medium"), ("High", "high")],
    value="low",
    description="Reasoning:",
)

run_count_dropdown = widgets.Dropdown(
    options=[("1 run", 1), ("3 runs", 3), ("5 runs", 5), ("10 runs", 10), ("20 runs", 20)],
    value=3,
    description="# Runs:",
)

location_toggle_multi = widgets.Checkbox(value=False, description="Apply location bias")
location_presets_multi = widgets.Dropdown(
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
custom_country_multi = widgets.Text(value="US", description="Country:", disabled=True)
custom_city_multi = widgets.Text(value="New York", description="City:", disabled=True)
custom_region_multi = widgets.Text(value="New York", description="Region:", disabled=True)

run_multi_button = widgets.Button(description="Run Multi-Search", button_style="primary", icon="refresh")
output_multi = widgets.Output()

preset_map_multi = {
    "us_ny": ("US", "New York", "New York"),
    "uk_london": ("GB", "London", "London"),
    "us_sf": ("US", "San Francisco", "California"),
    "fr_paris": ("FR", "Paris", "Île-de-France"),
    "de_berlin": ("DE", "Berlin", "Berlin"),
}


def update_location_multi(change=None):
    enabled = location_toggle_multi.value
    location_presets_multi.disabled = not enabled
    is_custom = enabled and location_presets_multi.value == "custom"
    for widget in (custom_country_multi, custom_city_multi, custom_region_multi):
        widget.disabled = not is_custom


def get_location_multi():
    if not location_toggle_multi.value:
        return None
    if location_presets_multi.value != "custom":
        country, city, region = preset_map_multi[location_presets_multi.value]
    else:
        country = custom_country_multi.value or "US"
        city = custom_city_multi.value or "New York"
        region = custom_region_multi.value or city
    return format_location(country=country, city=city, region=region)


location_toggle_multi.observe(update_location_multi, "value")
location_presets_multi.observe(update_location_multi, "value")
update_location_multi()


def slugify(text: str, length: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:length] or "query"


def run_multi(_):
    query = query_input_multi.value.strip()
    run_count = run_count_dropdown.value
    user_location = get_location_multi()

    with output_multi:
        clear_output()
        if not query:
            print("⚠️ Please enter a query before running the analysis.")
            return

        reporter = ReportHelper('multi_run', PATHS)
        execution_id = reporter.execution_id
        model_name = getattr(search_agent, 'model', 'unknown')

        print(f"▶ Execution ID: {execution_id}")
        print(f"▶ Running {run_count} searches for: {query}")
        if user_location:
            print(f"📍 Location bias: {user_location}\n")
        else:
            print("📍 Location bias: none\n")

        summaries = []

        for idx in range(1, run_count + 1):
            run_id = f"{execution_id}_run_{idx}"
            print(f"--- Run {idx}/{run_count} ---")
            print("⏳ Requesting response...\n")
            response = search_agent.search(
                query,
                search_context_size=context_dropdown_multi.value,
                reasoning_effort=reasoning_dropdown_multi.value,
                verbosity="medium",
                user_location=user_location,
            )

            raw_path = reporter.save_raw_response(f"run_{idx}", response)
            print_search_result(response)
            print()

            text = search_agent.extract_text_response(response)
            citations = search_agent.extract_citations(response)
            citation_urls = []

            if citations:
                for rank, cite in enumerate(citations, 1):
                    url = cite.get('url', '')
                    domain = urlparse(url).netloc.replace('www.', '') if url else ''
                    reporter.add_detail_row(
                        unit_id=run_id,
                        turn_or_run=idx,
                        role='AI System',
                        model=model_name,
                        query_or_topic=query,
                        message_text=text,
                        citation_rank=rank,
                        citation_title=cite.get('title', ''),
                        citation_url=url,
                        domain=domain,
                        context=context_dropdown_multi.value,
                        reasoning=reasoning_dropdown_multi.value,
                        location_country=user_location['country'] if user_location else None,
                        location_city=user_location['city'] if user_location else None,
                        location_region=user_location['region'] if user_location else None,
                        response_file=str(raw_path),
                    )
                    if url:
                        citation_urls.append(url)
            else:
                reporter.add_detail_row(
                    unit_id=run_id,
                    turn_or_run=idx,
                    role='AI System',
                    model=model_name,
                    query_or_topic=query,
                    message_text=text,
                    citation_rank=None,
                    citation_title=None,
                    citation_url=None,
                    domain=None,
                    context=context_dropdown_multi.value,
                    reasoning=reasoning_dropdown_multi.value,
                    location_country=user_location['country'] if user_location else None,
                    location_city=user_location['city'] if user_location else None,
                    location_region=user_location['region'] if user_location else None,
                    response_file=str(raw_path),
                )

            summaries.append({'run': idx, 'citations': citation_urls})
            print(f"Run {idx} complete — {len(citation_urls)} citations captured.\n")

        detail_path = reporter.write_detail_csv()

        df = pd.DataFrame(reporter._detail_rows)
        df_valid = df.dropna(subset=['citation_url'])
        citation_sets = [tuple(sorted(summary['citations'])) for summary in summaries]
        unique_sets = len(set(citation_sets))
        consistency_pct = 100.0 if len(citation_sets) <= 1 else 100 * (1 - (unique_sets - 1) / len(citation_sets))
        domain_counts = df_valid['domain'].value_counts().head(5)

        print("=== Consistency Summary ===")
        print(f"Runs executed: {run_count}")
        print(f"Unique citation sets: {unique_sets}")
        print(f"Citation consistency score: {consistency_pct:.1f}%\n")
        if not domain_counts.empty:
            print("Top cited domains:")
            for domain, count in domain_counts.items():
                print(f"  • {domain}: {count}")

        summary_row = {
            'model': model_name,
            'query': query,
            'run_count': run_count,
            'context': context_dropdown_multi.value,
            'reasoning': reasoning_dropdown_multi.value,
            'location_country': user_location['country'] if user_location else None,
            'location_city': user_location['city'] if user_location else None,
            'location_region': user_location['region'] if user_location else None,
            'total_citations': int(df_valid['citation_url'].count()),
            'unique_citation_urls': int(df_valid['citation_url'].nunique()),
            'unique_domains': int(df_valid['domain'].nunique()),
            'unique_citation_sets': unique_sets,
            'consistency_pct': consistency_pct,
        }
        for idx, (domain, count) in enumerate(domain_counts.items(), start=1):
            summary_row[f"top_domain_{idx}"] = domain
            summary_row[f"top_domain_{idx}_count"] = count

        summary_path = reporter.write_summary_csv(summary_row)

        print(f"\n💾 Detail CSV: {detail_path}")
        print(f"💾 Summary CSV: {summary_path}")


run_multi_button.on_click(run_multi)

location_box_multi = widgets.VBox([
    location_toggle_multi,
    location_presets_multi,
    widgets.HBox([custom_country_multi, custom_city_multi, custom_region_multi]),
])

controls_multi = widgets.VBox([
    query_input_multi,
    widgets.HBox([context_dropdown_multi, reasoning_dropdown_multi, run_count_dropdown]),
    location_box_multi,
    run_multi_button,
])

display(widgets.VBox([controls_multi, output_multi]))
