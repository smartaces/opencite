from typing import Any, Dict, List, Optional

import ipywidgets as widgets
from IPython.display import display, clear_output
from openai import OpenAI

# Require client from cell_02
if 'client' not in globals():
    raise RuntimeError("OpenRouter client not initialized. Run the API key cell first.")


def format_location(country: str = "US", city: str = "New York", region: str = "New York") -> dict:
    """Return a location dict for location-biased queries."""
    return {
        "country": country,
        "city": city,
        "region": region,
    }


class OpenRouterSearchAgent:
    """OpenRouter Web Search Agent using chat.completions with :online suffix."""

    def __init__(self, client: OpenAI, model: str = "openai/gpt-4o:online"):
        self.client = client
        self.model = model
        self.conversation_history: List[Dict[str, Any]] = []

    def search(
        self,
        query: str,
        search_context_size: str = "medium",
        user_location: Optional[Dict[str, str]] = None,
        force_search: bool = False,
        reasoning_effort: str = "low",
        verbosity: str = "medium",
        use_previous_reasoning: bool = True,
    ) -> Any:
        """Execute a web search query via OpenRouter.

        Args:
            query: The search query text.
            search_context_size: Ignored (OpenRouter handles this internally).
            user_location: Optional location dict for location-biased results.
            force_search: Ignored (OpenRouter :online suffix always enables search).
            reasoning_effort: Ignored (OpenRouter uses model defaults).
            verbosity: Ignored (OpenRouter uses model defaults).
            use_previous_reasoning: Ignored (OpenRouter doesn't support response chaining).

        Returns:
            OpenRouter chat completion response object.
        """
        # Build the query with optional location context
        message_content = query
        if user_location:
            location_parts = []
            if user_location.get("city"):
                location_parts.append(user_location["city"])
            if user_location.get("region"):
                location_parts.append(user_location["region"])
            if user_location.get("country"):
                location_parts.append(user_location["country"])
            if location_parts:
                location_str = ", ".join(location_parts)
                message_content = f"{query}\n\n(Please provide information relevant to someone in {location_str})"

        messages = [{"role": "user", "content": message_content}]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
            )
            self.conversation_history.append({"query": query, "response": response})
            return response
        except Exception as exc:
            return {"error": str(exc)}

    def extract_text_response(self, response: Any) -> str:
        """Extract the main text content from the response.

        Args:
            response: OpenRouter chat completion response.

        Returns:
            The text content of the response, or error message.
        """
        try:
            if isinstance(response, dict) and "error" in response:
                return f"Error: {response['error']}"
            content = response.choices[0].message.content
            return content if content else ""
        except Exception as exc:
            return f"Error extracting response: {exc}"

    def extract_citations(self, response: Any) -> List[Dict[str, Any]]:
        """Extract citations from OpenRouter response annotations.

        Args:
            response: OpenRouter chat completion response.

        Returns:
            List of citation dicts with 'url' and 'title' keys.
        """
        citations: List[Dict[str, Any]] = []
        try:
            if isinstance(response, dict) and "error" in response:
                return citations

            # Get annotations from response
            annotations = getattr(response.choices[0].message, 'annotations', None)
            if annotations is None:
                return citations

            for ann in annotations:
                # Get url_citation object
                url_citation = getattr(ann, 'url_citation', None)
                if url_citation is None:
                    continue

                # Extract URL (required)
                url = getattr(url_citation, 'url', None)
                if not url:
                    continue

                # Extract title (optional, default to 'Untitled')
                title = getattr(url_citation, 'title', None) or 'Untitled'

                citations.append({
                    "url": url,
                    "title": title,
                })

        except Exception:
            # Graceful fallback - return whatever citations we collected
            pass

        return citations

    def get_search_actions(self, response: Any) -> List[Dict[str, Any]]:
        """Get search action metadata (not available in OpenRouter).

        Args:
            response: OpenRouter chat completion response.

        Returns:
            Empty list (OpenRouter doesn't expose search actions).
        """
        # OpenRouter doesn't expose search action details like OpenAI does
        return []

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        print("Conversation history cleared")


def print_search_result(response, show_citations: bool = True, show_actions: bool = False) -> None:
    """Pretty-print a response from the search agent."""
    text = search_agent.extract_text_response(response)

    print("=" * 80)
    print("RESPONSE:")
    print("=" * 80)
    print(text)
    print()

    if show_citations:
        citations = search_agent.extract_citations(response)
        if citations:
            print("=" * 80)
            print("CITATIONS:")
            print("=" * 80)
            for i, cite in enumerate(citations, 1):
                print(f"{i}. {cite.get('title', 'Untitled')}")
                print(f"   URL: {cite.get('url', 'Unknown')}")
                print()
        else:
            print("No citations found for this query.")


# ============================================================================
# MODEL SELECTION UI
# ============================================================================

MODEL_OPTIONS = [
    ("openai/gpt-4o – OpenAI GPT-4o with web search", "openai/gpt-4o:online"),
    ("openai/gpt-4o-mini – OpenAI GPT-4o Mini (faster, cheaper)", "openai/gpt-4o-mini:online"),
    ("anthropic/claude-sonnet-4 – Anthropic Claude Sonnet 4", "anthropic/claude-sonnet-4:online"),
    ("perplexity/sonar-pro – Perplexity Sonar Pro (native search)", "perplexity/sonar-pro:online"),
    ("perplexity/sonar – Perplexity Sonar (native search, faster)", "perplexity/sonar:online"),
    ("google/gemini-2.0-flash-001 – Google Gemini 2.0 Flash", "google/gemini-2.0-flash-001:online"),
    ("meta-llama/llama-3.3-70b-instruct – Meta Llama 3.3 70B", "meta-llama/llama-3.3-70b-instruct:online"),
]

DEFAULT_MODEL = "openai/gpt-4o:online"

# Track state
_agent_state = {"initialized": False, "current_model": None}

model_dropdown = widgets.Dropdown(
    options=MODEL_OPTIONS,
    value=DEFAULT_MODEL,
    description="Model:",
    style={'description_width': '60px'},
    layout=widgets.Layout(width='500px'),
)

confirm_button = widgets.Button(
    description="Confirm Model",
    button_style="primary",
    icon="check",
    layout=widgets.Layout(width='140px'),
)

status_output = widgets.Output()


def _handle_confirm(_):
    global search_agent
    selected_model = model_dropdown.value

    with status_output:
        clear_output()

        if not _agent_state["initialized"]:
            # First time initialization
            search_agent = OpenRouterSearchAgent(client, model=selected_model)
            _agent_state["initialized"] = True
            _agent_state["current_model"] = selected_model
            print(f"Model {selected_model} selection confirmed!")
            print("Search agent ready")
            print("format_location helper ready")
        elif selected_model != _agent_state["current_model"]:
            # Model changed
            search_agent = OpenRouterSearchAgent(client, model=selected_model)
            _agent_state["current_model"] = selected_model
            print(f"Model {selected_model} selection updated!")
            print("Search agent reconfigured")
        else:
            # Same model confirmed again
            print(f"Model {selected_model} already active")


confirm_button.on_click(_handle_confirm)

# Layout
controls = widgets.VBox([
    widgets.HTML("<h3>OpenRouter Search Agent Setup</h3>"),
    widgets.HTML("<p>Select a model for web search. The :online suffix enables web search for all models.</p>"),
    widgets.HBox([model_dropdown, confirm_button], layout=widgets.Layout(gap='8px', align_items='center')),
    status_output,
])

display(controls)

# Auto-initialize with default model
search_agent = OpenRouterSearchAgent(client, model=DEFAULT_MODEL)
_agent_state["initialized"] = True
_agent_state["current_model"] = DEFAULT_MODEL
print(f"Model {DEFAULT_MODEL} initialized (default)")
print("format_location helper ready")
print("Use the dropdown above to change models if needed")
