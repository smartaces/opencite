# schemas/openai_models.py
"""
OpenAI model schemas for the modular bulk prompt runner.

Defines all OpenAI models that support web search via the Responses API.
Update this file when new models are released.

Model capabilities:
- native_search: All models here support web search via Responses API
- supports_location: Location bias for search results
- supports_reasoning: Reasoning effort parameter (gpt-5.x family)
- search_context_options: Search depth control

Notes:
- o3/o4 series don't support search_context_size parameter
- gpt-5.x family supports reasoning effort and verbosity
- All models use "web_search_preview" tool type
"""

import sys
from pathlib import Path

# Add parent directory to path for imports when running standalone
if __name__ != "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from core.citation import ModelSchema


# =============================================================================
# GPT-5.2 Series (Latest Flagship)
# =============================================================================

GPT_5_2 = ModelSchema(
    id="gpt-5.2",
    name="GPT-5.2",
    native_search=True,
    supports_location=True,
    supports_reasoning=True,
    reasoning_options=["high", "medium", "low"],
    search_context_options=["low", "medium", "high"],
    additional_params={
        "verbosity_options": ["short", "medium", "long"],
        "supports_previous_response_id": True,
    },
)

GPT_5_2_PRO = ModelSchema(
    id="gpt-5.2-pro",
    name="GPT-5.2 Pro",
    native_search=True,
    supports_location=True,
    supports_reasoning=True,
    reasoning_options=["high", "medium", "low"],
    search_context_options=["low", "medium", "high"],
    additional_params={
        "verbosity_options": ["short", "medium", "long"],
        "supports_previous_response_id": True,
        "description": "Harder thinking, tougher problems, slower",
    },
)

# =============================================================================
# GPT-5.1 Series
# =============================================================================

GPT_5_1 = ModelSchema(
    id="gpt-5.1",
    name="GPT-5.1",
    native_search=True,
    supports_location=True,
    supports_reasoning=True,
    reasoning_options=["high", "medium", "low"],
    search_context_options=["low", "medium", "high"],
    additional_params={
        "verbosity_options": ["short", "medium", "long"],
        "supports_previous_response_id": True,
    },
)

# =============================================================================
# GPT-5 Series
# =============================================================================

GPT_5 = ModelSchema(
    id="gpt-5",
    name="GPT-5",
    native_search=True,
    supports_location=True,
    supports_reasoning=True,
    reasoning_options=["high", "medium", "low"],
    search_context_options=["low", "medium", "high"],
    additional_params={
        "verbosity_options": ["short", "medium", "long"],
        "supports_previous_response_id": True,
    },
)

GPT_5_MINI = ModelSchema(
    id="gpt-5-mini",
    name="GPT-5 Mini",
    native_search=True,
    supports_location=True,
    supports_reasoning=True,
    reasoning_options=["high", "medium", "low"],
    search_context_options=["low", "medium", "high"],
    additional_params={
        "verbosity_options": ["short", "medium", "long"],
        "supports_previous_response_id": True,
        "description": "Cost-optimized, faster",
    },
)

# =============================================================================
# o3 Reasoning Series
# Note: o3 series does NOT support search_context_size parameter
# =============================================================================

O3 = ModelSchema(
    id="o3",
    name="o3 (Reasoning)",
    native_search=True,
    supports_location=True,
    supports_reasoning=True,
    reasoning_options=["high", "medium", "low"],
    search_context_options=None,  # Not supported on o3
    additional_params={
        "supports_previous_response_id": True,
        "description": "Advanced reasoning model",
    },
)

O3_PRO = ModelSchema(
    id="o3-pro",
    name="o3 Pro (Reasoning)",
    native_search=True,
    supports_location=True,
    supports_reasoning=True,
    reasoning_options=["high", "medium", "low"],
    search_context_options=None,  # Not supported on o3-pro
    additional_params={
        "supports_previous_response_id": True,
        "description": "Advanced reasoning, maximum capability",
    },
)

# =============================================================================
# o4 Series
# Note: o4 series does NOT support search_context_size parameter
# =============================================================================

O4_MINI = ModelSchema(
    id="o4-mini",
    name="o4 Mini",
    native_search=True,
    supports_location=True,
    supports_reasoning=True,
    reasoning_options=["high", "medium", "low"],
    search_context_options=None,  # Not supported on o4-mini
    additional_params={
        "supports_previous_response_id": True,
        "description": "Fast reasoning model",
    },
)

# =============================================================================
# GPT-4.1 Series (Legacy but still supported)
# =============================================================================

GPT_4_1 = ModelSchema(
    id="gpt-4.1",
    name="GPT-4.1",
    native_search=True,
    supports_location=True,
    supports_reasoning=False,  # No reasoning parameter
    search_context_options=["low", "medium", "high"],
    additional_params={
        "supports_previous_response_id": True,
        "description": "Legacy model, still supported",
    },
)


# =============================================================================
# All OpenAI Models - Export List
# =============================================================================

OPENAI_MODELS = [
    # GPT-5.2 Series (Latest)
    GPT_5_2,
    GPT_5_2_PRO,
    # GPT-5.1 Series
    GPT_5_1,
    # GPT-5 Series
    GPT_5,
    GPT_5_MINI,
    # o3 Reasoning Series
    O3,
    O3_PRO,
    # o4 Series
    O4_MINI,
    # GPT-4.1 (Legacy)
    GPT_4_1,
]

# Default model for new agents
DEFAULT_MODEL_ID = "gpt-5.2"


def get_model_by_id(model_id: str) -> ModelSchema:
    """Get a model schema by ID.

    Args:
        model_id: The model ID to look up

    Returns:
        ModelSchema if found

    Raises:
        ValueError: If model not found
    """
    for model in OPENAI_MODELS:
        if model.id == model_id:
            return model
    raise ValueError(f"Unknown OpenAI model: {model_id}")


def get_search_models() -> list:
    """Get all models that support native web search.

    Returns:
        List of ModelSchema objects with native_search=True
    """
    return [m for m in OPENAI_MODELS if m.native_search]


def get_model_choices() -> list:
    """Get model choices for UI dropdown.

    Returns:
        List of (display_name, model_id) tuples
    """
    choices = []
    for m in OPENAI_MODELS:
        desc = m.additional_params.get("description", "") if m.additional_params else ""
        if desc:
            display = f"{m.name} - {desc}"
        else:
            display = m.name
        choices.append((display, m.id))
    return choices
