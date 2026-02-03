# schemas/google_models.py
"""
Google Gemini model definitions.

Update this file when new models are released.
Models are organized by series (Gemini 3, 2.5, 2.0) with different
thinking configuration support.

Gemini 3 models: Use `thinking_level` parameter
Gemini 2.5 models: Use `thinking_budget` parameter
Gemini 2.0 models: No thinking support
"""

from __future__ import annotations

from core.citation import ModelSchema


# =============================================================================
# Gemini 3 Series (Preview) - Uses thinkingLevel
# =============================================================================

GEMINI_3_FLASH_PREVIEW = ModelSchema(
    id="gemini-3-flash-preview",
    name="Gemini 3 Flash (Preview)",
    native_search=True,
    supports_location=False,  # No native support, use prompt workaround
    supports_reasoning=True,
    reasoning_options=["minimal", "low", "medium", "high"],
    search_context_options=None,  # Not configurable in Gemini
    additional_params={
        "thinking_param": "thinking_level",
        "default_thinking": "high",
    },
)

GEMINI_3_PRO_PREVIEW = ModelSchema(
    id="gemini-3-pro-preview",
    name="Gemini 3 Pro (Preview)",
    native_search=True,
    supports_location=False,
    supports_reasoning=True,
    reasoning_options=["low", "high"],  # No minimal/medium for Pro
    search_context_options=None,
    additional_params={
        "thinking_param": "thinking_level",
        "default_thinking": "high",
    },
)


# =============================================================================
# Gemini 2.5 Series - Uses thinkingBudget
# =============================================================================

GEMINI_2_5_PRO = ModelSchema(
    id="gemini-2.5-pro",
    name="Gemini 2.5 Pro",
    native_search=True,
    supports_location=False,
    supports_reasoning=True,
    reasoning_options=["dynamic", "low", "medium", "high"],  # Cannot disable
    search_context_options=None,
    additional_params={
        "thinking_param": "thinking_budget",
        "budget_map": {
            "dynamic": -1,
            "low": 1024,
            "medium": 8192,
            "high": 24576,
        },
        "cannot_disable": True,
    },
)

GEMINI_2_5_FLASH = ModelSchema(
    id="gemini-2.5-flash",
    name="Gemini 2.5 Flash",
    native_search=True,
    supports_location=False,
    supports_reasoning=True,
    reasoning_options=["off", "dynamic", "low", "medium", "high"],
    search_context_options=None,
    additional_params={
        "thinking_param": "thinking_budget",
        "budget_map": {
            "off": 0,
            "dynamic": -1,
            "low": 1024,
            "medium": 8192,
            "high": 24576,
        },
    },
)

GEMINI_2_5_FLASH_LITE = ModelSchema(
    id="gemini-2.5-flash-lite",
    name="Gemini 2.5 Flash Lite",
    native_search=True,
    supports_location=False,
    supports_reasoning=True,
    reasoning_options=["off", "dynamic", "low", "medium", "high"],
    search_context_options=None,
    additional_params={
        "thinking_param": "thinking_budget",
        "budget_map": {
            "off": 0,
            "dynamic": -1,
            "low": 1024,
            "medium": 8192,
            "high": 24576,
        },
    },
)


# =============================================================================
# Gemini 2.0 Series - No thinking support
# =============================================================================

GEMINI_2_0_FLASH = ModelSchema(
    id="gemini-2.0-flash",
    name="Gemini 2.0 Flash",
    native_search=True,
    supports_location=False,
    supports_reasoning=False,  # No thinking support
    reasoning_options=None,
    search_context_options=None,
)


# =============================================================================
# Export all models
# =============================================================================

GOOGLE_MODELS = [
    # Gemini 3 Series (Preview)
    GEMINI_3_FLASH_PREVIEW,
    GEMINI_3_PRO_PREVIEW,
    # Gemini 2.5 Series
    GEMINI_2_5_PRO,
    GEMINI_2_5_FLASH,
    GEMINI_2_5_FLASH_LITE,
    # Gemini 2.0 Series
    GEMINI_2_0_FLASH,
]


def get_search_models():
    """Return models that support native search."""
    return [m for m in GOOGLE_MODELS if m.native_search]


def get_model_by_id(model_id: str):
    """Get model schema by ID."""
    return next((m for m in GOOGLE_MODELS if m.id == model_id), None)
