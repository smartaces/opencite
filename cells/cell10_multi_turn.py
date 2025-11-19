import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
import json
import os
import re

PERSONA_MODEL = "gpt-4o-mini"

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

start_prompt_input = widgets.Textarea(
    value="Compare the Samsung S25 Ultra versus the Google Pixel 10 Pro",
    description="Starting prompt:",
    layout=widgets.Layout(width="100%", height="80px"),
)

persona_presets = [
    ("Budget college student (US)", "Budget-minded US college student upgrading an aging Android."),
    ("Enterprise IT director", "Enterprise IT director focused on security and lifecycle support."),
    ("Travel photographer", "Travel photographer who values camera quality and battery life."),
    ("Marathon runner", "Marathon runner looking for wearable integration and durability."),
    ("Custom persona", "custom"),
]

persona_dropdown = widgets.Dropdown(options=persona_presets, value=persona_presets[0][1], description="Persona:")
persona_custom = widgets.Textarea(
    value="",
    description="Custom profile:",
    layout=widgets.Layout(width="100%", height="80px"),
    disabled=True,
)

turn_dropdown = widgets.Dropdown(options=[("1 turn", 1), ("3 turns", 3), ("5 turns", 5)], value=5, description="Turns:")
context_dropdown = widgets.Dropdown(options=[("Low", "low"), ("Medium", "medium"), ("High", "high")], value="medium", description="Context:")
reasoning_dropdown = widgets.Dropdown(options=[("Low", "low"), ("Medium", "medium"), ("High", "high")], value="low", description="Reasoning:")
run_count_dropdown = widgets.Dropdown(
    options=[("1 run", 1), ("3 runs", 3), ("5 runs", 5), ("10 runs", 10), ("20 runs", 20)],
    value=1,
    description="# Runs:",
)

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

run_button = widgets.Button(description="Run Simulation", button_style="primary", icon="comments")
output = widgets.Output()

preset_locations = {
    "us_ny": ("US", "New York", "New York"),
    "uk_london": ("GB", "London", "London"),
    "us_sf": ("US", "San Francisco", "California"),
    "fr_paris": ("FR", "Paris", "Île-de-France"),
    "de_berlin": ("DE", "Berlin", "Berlin"),
}


def update_persona(change=None):
    persona_custom.disabled = persona_dropdown.value != "custom"


def update_location(change=None):
    enabled = location_toggle.value
    location_presets.disabled = not enabled
    is_custom = enabled and location_presets.value == "custom"
    for widget in (custom_country, custom_city, custom_region):
        widget.disabled = not is_custom


persona_dropdown.observe(update_persona, "value")
location_toggle.observe(update_location, "value")
location_presets.observe(update_location, "value")
update_persona()
update_location()


def get_persona_description():
    if persona_dropdown.value == "custom":
        return persona_custom.value.strip() or "Curious consumer researching products."
    return persona_dropdown.value


def get_location_dict():
    if not location_toggle.value:
        return None
    if location_presets.value != "custom":
        country, city, region = preset_locations[location_presets.value]
        return format_location(country=country, city=city, region=region)
    country = custom_country.value or "US"
    city = custom_city.value or "New York"
    region = custom_region.value or city
    return format_location(country=country, city=city, region=region)

def simulate_persona_message(persona_profile: str, history: list, topic: str) -> str:
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
    response = search_agent.client.responses.create(
        model=PERSONA_MODEL,
        input=[{"role": "system", "content": prompt}] + history,
    )
    return search_agent.extract_text_response(response).strip()


def run_simulation(_):
    topic = start_prompt_input.value.strip()
    turns = turn_dropdown.value
    run_count = run_count_dropdown.value
    persona_profile = get_persona_description()
    user_location = get_location_dict()
    model_name = getattr(search_agent, 'model', 'unknown')
    persona_model_name = PERSONA_MODEL

    with output:
        clear_output()
        if not topic:
            print("⚠️ Please enter a starting prompt.")
            return

        print(f"▶ Topic: {topic}")
        print(f"▶ Persona: {persona_profile}")
        print(f"▶ Turns per run: {turns}")
        print(f"▶ Runs: {run_count}")
        if user_location:
            print(f"📍 Location bias: {user_location}\n")
        else:
            print("📍 Location bias: none\n")

        for run_idx in range(1, run_count + 1):
            print(f"\n===== Run {run_idx}/{run_count} =====")
            reporter = ReportHelper('multi_turn', PATHS)
            execution_id = reporter.execution_id
            print(f"▶ Execution ID: {execution_id}\n")

            persona_history = []
            advisor_history = []

            for turn in range(1, turns + 1):
                turn_id = f"{execution_id}_turn_{turn}"
                if turn == 1:
                    persona_msg = topic
                else:
                    persona_msg = simulate_persona_message(persona_profile, persona_history + advisor_history, topic)
                persona_history.append({"role": "user", "content": persona_msg})
                print(f"Turn {turn} — Persona")
                print(persona_msg)
                print()

                reporter.add_detail_row(
                    unit_id=turn_id,
                    turn_or_run=turn,
                    role='persona',
                    model=persona_model_name,
                    query_or_topic=topic,
                    message_text=persona_msg,
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
                    persona_profile=persona_profile,
                    persona_model=persona_model_name,
                )

                print("⏳ Advisor responding...\n")
                response = search_agent.search(
                    persona_msg,
                    search_context_size=context_dropdown.value,
                    reasoning_effort=reasoning_dropdown.value,
                    verbosity="medium",
                    user_location=user_location,
                )

                raw_path = reporter.save_raw_response(f"turn_{turn}", response)
                print_search_result(response)
                print()

                advisor_text = search_agent.extract_text_response(response)
                advisor_history.append({"role": "assistant", "content": advisor_text})

                citations = search_agent.extract_citations(response)
                if citations:
                    for rank, cite in enumerate(citations, 1):
                        url = cite.get("url", "")
                        domain = urlparse(url).netloc.replace("www.", "") if url else ""
                        reporter.add_detail_row(
                            unit_id=turn_id,
                            turn_or_run=turn,
                            role='AI System',
                            model=model_name,
                            query_or_topic=topic,
                            message_text=advisor_text,
                            citation_rank=rank,
                            citation_title=cite.get("title", ""),
                            citation_url=url,
                            domain=domain,
                            context=context_dropdown.value,
                            reasoning=reasoning_dropdown.value,
                            location_country=user_location['country'] if user_location else None,
                            location_city=user_location['city'] if user_location else None,
                            location_region=user_location['region'] if user_location else None,
                            response_file=str(raw_path),
                            persona_profile=persona_profile,
                            persona_model=persona_model_name,
                        )
                else:
                    reporter.add_detail_row(
                        unit_id=turn_id,
                        turn_or_run=turn,
                        role='AI System',
                        model=model_name,
                        query_or_topic=topic,
                        message_text=advisor_text,
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
                        persona_profile=persona_profile,
                        persona_model=persona_model_name,
                    )

            detail_path = reporter.write_detail_csv()
            df = pd.DataFrame(reporter._detail_rows)
            advisor_citations = df[df['role'] == 'AI System'].dropna(subset=['citation_url'])
            domain_counts = advisor_citations['domain'].value_counts().head(5)

            print("=== Citation Summary ===")
            print(f"Total AI System citations: {len(advisor_citations)}")
            print(f"Unique citation URLs: {advisor_citations['citation_url'].nunique()}")
            if not domain_counts.empty:
                print("Top cited domains:")
                for domain, count in domain_counts.items():
                    print(f"  • {domain}: {count}")

            summary_row = {
                'model': model_name,
                'topic': topic,
                'turns': turns,
                'persona_profile': persona_profile,
                'persona_model': persona_model_name,
                'location_country': user_location['country'] if user_location else None,
                'location_city': user_location['city'] if user_location else None,
                'location_region': user_location['region'] if user_location else None,
                'total_citations': len(advisor_citations),
                'unique_citation_urls': advisor_citations['citation_url'].nunique(),
                'unique_domains': advisor_citations['domain'].nunique(),
            }
            for idx, (domain, count) in enumerate(domain_counts.items(), start=1):
                summary_row[f"top_domain_{idx}"] = domain
                summary_row[f"top_domain_{idx}_count"] = count

            summary_path = reporter.write_summary_csv(summary_row)

            print(f"\n💾 Detail CSV: {detail_path}")
            print(f"💾 Summary CSV: {summary_path}")
            print("✅ Run complete.")


run_button.on_click(run_simulation)

controls = widgets.VBox([
    start_prompt_input,
    persona_dropdown,
    persona_custom,
    widgets.HBox([turn_dropdown, context_dropdown, reasoning_dropdown, run_count_dropdown]),
    widgets.VBox([
        location_toggle,
        location_presets,
        widgets.HBox([custom_country, custom_city, custom_region])
    ]),
    run_button,
])

display(widgets.VBox([controls, output]))
