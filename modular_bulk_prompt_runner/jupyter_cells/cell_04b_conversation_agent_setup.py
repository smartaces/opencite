# cell_04b_conversation_agent_setup.py
"""
Conversation Agent Setup - Provider and model selection for multi-turn follow-ups.

This cell allows users to:
1. Select a provider (OpenAI, Anthropic, etc.)
2. Select a model from that provider (all models, not filtered)
3. Initialize the conversation agent

The conversation agent is used for generating persona-based follow-up questions
in multi-turn conversations. It does NOT perform web search.

If you're only running single-turn queries, you can skip this cell.
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

# Check for search agent (should be set up first)
if '_search_agent_state' not in globals() or not _search_agent_state.get("initialized"):
    print("Note: Search agent not yet initialized. Run cell 4a first for full functionality.")

# Import UI components
from ui.provider_selector import ProviderModelSelector, load_cartridge
from core.base_cartridge import BaseCartridge


# =============================================================================
# API Key Getter (reuse from search agent cell if available)
# =============================================================================

if 'get_api_key' not in globals():
    def get_api_key(api_key_name: str) -> str:
        """Get API key from environment or Colab secrets."""
        key = os.environ.get(api_key_name)
        if key:
            return key

        try:
            from google.colab import userdata
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
            pass

        raise ValueError(f"API key not found: {api_key_name}")


# =============================================================================
# Conversation Agent State
# =============================================================================

_conversation_agent_state = {
    "cartridge": None,
    "client": None,
    "model_id": None,
    "params": {},
    "initialized": False,
}


def on_conversation_agent_confirm(cartridge, client, model_id, params):
    """Callback when conversation agent is confirmed."""
    global conversation_agent_cartridge, conversation_agent_client, conversation_agent_model

    _conversation_agent_state["cartridge"] = cartridge
    _conversation_agent_state["client"] = client
    _conversation_agent_state["model_id"] = model_id
    _conversation_agent_state["params"] = params
    _conversation_agent_state["initialized"] = True

    # Set global variables for easy access
    conversation_agent_cartridge = cartridge
    conversation_agent_client = client
    conversation_agent_model = model_id

    print(f"\nConversation Agent Ready")
    print(f"  Provider: {cartridge.name}")
    print(f"  Model: {model_id}")
    print(f"\nThis agent will be used for generating follow-up questions in multi-turn conversations.")


def chat(messages, **kwargs):
    """Execute a chat completion using the configured conversation agent.

    This is a convenience function that uses the global conversation agent state.
    No web search is performed - this is for generating follow-up questions.

    Args:
        messages: List of message dicts or a single string
        **kwargs: Additional parameters

    Returns:
        ChatResponse object with text and metadata
    """
    if not _conversation_agent_state["initialized"]:
        raise RuntimeError("Conversation agent not initialized. Run this cell and click 'Confirm Selection'.")

    cartridge = _conversation_agent_state["cartridge"]
    client = _conversation_agent_state["client"]
    model_id = _conversation_agent_state["model_id"]
    params = _conversation_agent_state["params"].copy()
    params.update(kwargs)

    # Normalize messages
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    return cartridge.chat(
        client=client,
        model_id=model_id,
        messages=messages,
        **params
    )


def generate_followup(persona: str, conversation_history: list, topic: str) -> str:
    """Generate a persona-based follow-up question.

    Args:
        persona: The persona description
        conversation_history: List of previous messages
        topic: The original topic/query

    Returns:
        Generated follow-up question text
    """
    if not _conversation_agent_state["initialized"]:
        raise RuntimeError("Conversation agent not initialized.")

    prompt = f"""You are acting as a user with this persona: {persona}

Based on the conversation so far about "{topic}", generate a natural follow-up question
that this persona would ask. The question should:
1. Build on the previous response
2. Reflect the persona's perspective and interests
3. Seek additional relevant information

Previous conversation:
{_format_history(conversation_history)}

Generate only the follow-up question, nothing else."""

    response = chat(prompt)
    return response.text.strip()


def _format_history(history: list) -> str:
    """Format conversation history for the prompt."""
    lines = []
    for msg in history[-4:]:  # Last 4 messages to keep context manageable
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')[:500]  # Truncate long messages
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


# =============================================================================
# Display UI
# =============================================================================

# Determine default values based on search agent if available
default_provider = "openai"
default_model = "gpt-5.2"

if '_search_agent_state' in globals() and _search_agent_state.get("initialized"):
    # Use same provider as search agent by default
    if _search_agent_state.get("cartridge"):
        for key, (module, _) in [("openai", ("providers.openai_cartridge", "OpenAICartridge"))]:
            if _search_agent_state["cartridge"].name == "OpenAI":
                default_provider = "openai"
                default_model = _search_agent_state.get("model_id", "gpt-5.2")
                break

conversation_selector = ProviderModelSelector(
    title="Conversation Agent Setup (Optional)",
    description="Select a model for generating follow-up questions in multi-turn conversations. "
                "This agent does NOT perform web search. Skip this if only running single-turn queries.",
    filter_search_models=False,  # Show all models, not just search-capable
    default_provider=default_provider,
    default_model=default_model,
    on_confirm=on_conversation_agent_confirm,
    api_key_getter=get_api_key,
)

conversation_selector.display()
