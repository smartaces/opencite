# cell_04a_search_agent_setup.py
"""
Search Agent Setup - Provider and model selection for web search.

This cell allows users to:
1. Select a provider (OpenAI, Anthropic, etc.)
2. Select a model from that provider (filtered to models with native web search)
3. Configure search parameters (reasoning, search depth, location)
4. Initialize the search agent

The search agent is used for the primary web search queries that extract citations.
"""

import os
import sys
from pathlib import Path

# Ensure scripts directory is in path
if 'PATHS' in globals():
    scripts_dir = Path(PATHS.get('scripts', '')) / 'modular'
    if scripts_dir.exists():
        sys.path.insert(0, str(scripts_dir))

# Check for required globals
if 'PATHS' not in globals():
    raise RuntimeError(
        "Workspace not configured. Run the workspace setup cell first."
    )

# Import UI components
from ui.provider_selector import ProviderModelSelector, load_cartridge
from core.base_cartridge import BaseCartridge


# =============================================================================
# API Key Getter
# =============================================================================

def get_api_key(api_key_name: str) -> str:
    """Get API key from environment or Colab secrets.

    Args:
        api_key_name: Name of the API key (e.g., "OPENAI_API_KEY")

    Returns:
        The API key string

    Raises:
        ValueError: If key not found
    """
    # Try environment variable first
    key = os.environ.get(api_key_name)
    if key:
        return key

    # Try Colab secrets
    try:
        from google.colab import userdata
        # Map environment variable name to Colab secret name
        secret_map = {
            "OPENAI_API_KEY": "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
            "PERPLEXITY_API_KEY": "PERPLEXITY_API_KEY",
            "GOOGLE_API_KEY": "GEMINI_API_KEY",
            "XAI_API_KEY": "XAI_API_KEY",
        }
        secret_name = secret_map.get(api_key_name, api_key_name)
        key = userdata.get(secret_name)
        if key:
            return key
    except (ImportError, Exception):
        pass  # Not in Colab or secret not found

    raise ValueError(
        f"API key not found: {api_key_name}\n"
        f"Set it as an environment variable or add it to Colab secrets."
    )


# =============================================================================
# Search Agent State
# =============================================================================

# Global state for the search agent
_search_agent_state = {
    "cartridge": None,
    "client": None,
    "model_id": None,
    "params": {},
    "initialized": False,
}


def on_search_agent_confirm(cartridge, client, model_id, params):
    """Callback when search agent is confirmed."""
    global search_agent_cartridge, search_agent_client, search_agent_model

    _search_agent_state["cartridge"] = cartridge
    _search_agent_state["client"] = client
    _search_agent_state["model_id"] = model_id
    _search_agent_state["params"] = params
    _search_agent_state["initialized"] = True

    # Set global variables for easy access
    search_agent_cartridge = cartridge
    search_agent_client = client
    search_agent_model = model_id

    print(f"\nSearch Agent Ready")
    print(f"  Provider: {cartridge.name}")
    print(f"  Model: {model_id}")
    if params.get('reasoning_effort'):
        print(f"  Reasoning: {params['reasoning_effort']}")
    if params.get('search_context_size'):
        print(f"  Search Depth: {params['search_context_size']}")
    if params.get('location'):
        loc = params['location']
        print(f"  Location: {loc.get('city', '')}, {loc.get('region', '')}, {loc.get('country', '')}")


def search(query: str, **kwargs) -> dict:
    """Execute a web search using the configured search agent.

    This is a convenience function that uses the global search agent state.

    Args:
        query: The search query
        **kwargs: Additional parameters to override defaults

    Returns:
        SearchResponse object with text, citations, and metadata
    """
    if not _search_agent_state["initialized"]:
        raise RuntimeError("Search agent not initialized. Run this cell and click 'Confirm Selection'.")

    cartridge = _search_agent_state["cartridge"]
    client = _search_agent_state["client"]
    model_id = _search_agent_state["model_id"]
    params = _search_agent_state["params"].copy()
    params.update(kwargs)

    return cartridge.search(
        client=client,
        model_id=model_id,
        query=query,
        **params
    )


# =============================================================================
# Display UI
# =============================================================================

search_selector = ProviderModelSelector(
    title="Search Agent Setup",
    description="Select a provider and model for web search. Only models with native web search support are shown.",
    filter_search_models=True,
    default_provider="openai",
    default_model="gpt-5.2",
    on_confirm=on_search_agent_confirm,
    api_key_getter=get_api_key,
)

search_selector.display()
