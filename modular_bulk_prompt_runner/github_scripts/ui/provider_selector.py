# ui/provider_selector.py
"""
Provider and model selection UI components for the modular bulk prompt runner.

Provides ipywidgets-based dropdowns for selecting:
1. Provider (OpenAI, Anthropic, etc.)
2. Model (populated from provider's schema)
3. Dynamic parameter controls based on model capabilities
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import ipywidgets as widgets
from IPython.display import display, clear_output

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_cartridge import BaseCartridge
from core.citation import ModelSchema


# =============================================================================
# Cartridge Registry
# =============================================================================

# Maps provider key to (module_path, class_name)
# Cartridges are imported dynamically to avoid requiring all SDKs
CARTRIDGE_REGISTRY: Dict[str, Tuple[str, str]] = {
    "openai": ("providers.openai_cartridge", "OpenAICartridge"),
    "google": ("providers.google_cartridge", "GoogleCartridge"),
    # Future providers:
    # "anthropic": ("providers.anthropic_cartridge", "AnthropicCartridge"),
    # "perplexity": ("providers.perplexity_cartridge", "PerplexityCartridge"),
    # "xai": ("providers.xai_cartridge", "XAICartridge"),
}

# Display names for providers
PROVIDER_DISPLAY_NAMES: Dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "perplexity": "Perplexity",
    "google": "Google",
    "xai": "xAI",
}


def get_available_providers() -> List[Tuple[str, str]]:
    """Get list of available providers for dropdown.

    Returns:
        List of (display_name, provider_key) tuples
    """
    return [
        (PROVIDER_DISPLAY_NAMES.get(key, key), key)
        for key in CARTRIDGE_REGISTRY.keys()
    ]


def load_cartridge(provider_key: str) -> BaseCartridge:
    """Dynamically load and instantiate a cartridge.

    Args:
        provider_key: Provider key (e.g., "openai")

    Returns:
        Instantiated cartridge

    Raises:
        ValueError: If provider not found in registry
        ImportError: If cartridge module can't be loaded
    """
    if provider_key not in CARTRIDGE_REGISTRY:
        raise ValueError(f"Unknown provider: {provider_key}")

    module_path, class_name = CARTRIDGE_REGISTRY[provider_key]

    # Dynamic import
    import importlib
    module = importlib.import_module(module_path)
    cartridge_class = getattr(module, class_name)

    return cartridge_class()


# =============================================================================
# Provider Selector Widget
# =============================================================================

class ProviderModelSelector:
    """Widget for selecting provider and model with dynamic parameter controls.

    This widget provides:
    - Provider dropdown (OpenAI, Anthropic, etc.)
    - Model dropdown (populated from selected provider's schema)
    - Dynamic parameter controls based on model capabilities
    - Confirm button to initialize the agent

    Usage:
        selector = ProviderModelSelector(
            title="Search Agent Setup",
            filter_search_models=True,
            on_confirm=lambda cartridge, client, model: ...
        )
        selector.display()
    """

    def __init__(
        self,
        title: str = "Provider & Model Selection",
        description: str = "",
        filter_search_models: bool = False,
        default_provider: str = "openai",
        default_model: Optional[str] = None,
        on_confirm: Optional[Callable[[BaseCartridge, Any, str, Dict], None]] = None,
        api_key_getter: Optional[Callable[[str], str]] = None,
    ):
        """Initialize the selector.

        Args:
            title: Title displayed above the widget
            description: Description text
            filter_search_models: If True, only show models with native_search=True
            default_provider: Default provider key
            default_model: Default model ID (or None to use provider's default)
            on_confirm: Callback when user confirms selection
                        fn(cartridge, client, model_id, params)
            api_key_getter: Function to get API key for a provider
                           fn(api_key_name) -> api_key
        """
        self.title = title
        self.description = description
        self.filter_search_models = filter_search_models
        self.default_provider = default_provider
        self.default_model = default_model
        self.on_confirm = on_confirm
        self.api_key_getter = api_key_getter

        # Current state
        self.current_cartridge: Optional[BaseCartridge] = None
        self.current_client: Optional[Any] = None
        self.current_model_id: Optional[str] = None

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the widget UI components."""
        # Provider dropdown
        self.provider_dropdown = widgets.Dropdown(
            options=get_available_providers(),
            value=self.default_provider,
            description="Provider:",
            style={'description_width': '80px'},
            layout=widgets.Layout(width='300px'),
        )

        # Model dropdown (populated when provider selected)
        self.model_dropdown = widgets.Dropdown(
            options=[],
            description="Model:",
            style={'description_width': '80px'},
            layout=widgets.Layout(width='450px'),
        )

        # Dynamic parameter controls
        self.reasoning_dropdown = widgets.Dropdown(
            options=[],
            description="Reasoning:",
            style={'description_width': '80px'},
            layout=widgets.Layout(width='200px', display='none'),
        )

        self.search_context_dropdown = widgets.Dropdown(
            options=[("Medium", "medium"), ("Low", "low"), ("High", "high")],
            value="medium",
            description="Search Depth:",
            style={'description_width': '80px'},
            layout=widgets.Layout(width='200px', display='none'),
        )

        # Location controls
        self.location_checkbox = widgets.Checkbox(
            value=False,
            description="Enable Location Bias",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px'),
        )
        self.location_country = widgets.Text(
            value="US",
            description="Country:",
            style={'description_width': '60px'},
            layout=widgets.Layout(width='150px', display='none'),
        )
        self.location_city = widgets.Text(
            value="",
            placeholder="City",
            description="City:",
            style={'description_width': '60px'},
            layout=widgets.Layout(width='200px', display='none'),
        )
        self.location_region = widgets.Text(
            value="",
            placeholder="State/Region",
            description="Region:",
            style={'description_width': '60px'},
            layout=widgets.Layout(width='200px', display='none'),
        )

        self.location_container = widgets.VBox(
            [
                self.location_checkbox,
                widgets.HBox([self.location_country, self.location_city, self.location_region]),
            ],
            layout=widgets.Layout(display='none'),
        )

        # Confirm button
        self.confirm_button = widgets.Button(
            description="Confirm Selection",
            button_style="primary",
            icon="check",
            layout=widgets.Layout(width='160px'),
        )

        # Status output
        self.status_output = widgets.Output()

        # Wire up events
        self.provider_dropdown.observe(self._on_provider_change, names='value')
        self.model_dropdown.observe(self._on_model_change, names='value')
        self.location_checkbox.observe(self._on_location_toggle, names='value')
        self.confirm_button.on_click(self._on_confirm)

        # Initialize with default provider
        self._load_provider(self.default_provider)

    def _load_provider(self, provider_key: str):
        """Load a provider's cartridge and populate model dropdown."""
        try:
            self.current_cartridge = load_cartridge(provider_key)

            # Get models (optionally filtered)
            if self.filter_search_models:
                models = self.current_cartridge.get_models_with_search()
            else:
                models = self.current_cartridge.models

            # Build model options
            model_options = []
            for m in models:
                desc = ""
                if m.additional_params and "description" in m.additional_params:
                    desc = f" - {m.additional_params['description']}"
                model_options.append((f"{m.name}{desc}", m.id))

            self.model_dropdown.options = model_options

            # Set default model
            if self.default_model and any(m.id == self.default_model for m in models):
                self.model_dropdown.value = self.default_model
            elif model_options:
                self.model_dropdown.value = model_options[0][1]

        except Exception as exc:
            with self.status_output:
                clear_output()
                print(f"Error loading provider: {exc}")

    def _on_provider_change(self, change):
        """Handle provider dropdown change."""
        provider_key = change['new']
        self._load_provider(provider_key)

    def _on_model_change(self, change):
        """Handle model dropdown change - update parameter controls."""
        model_id = change['new']
        if not model_id or not self.current_cartridge:
            return

        model = self.current_cartridge.get_model(model_id)
        if not model:
            return

        # Show/hide reasoning options
        if model.supports_reasoning and model.reasoning_options:
            options = [(opt.capitalize(), opt) for opt in model.reasoning_options]
            self.reasoning_dropdown.options = options
            self.reasoning_dropdown.value = model.reasoning_options[1] if len(model.reasoning_options) > 1 else model.reasoning_options[0]
            self.reasoning_dropdown.layout.display = 'flex'
        else:
            self.reasoning_dropdown.layout.display = 'none'

        # Show/hide search context options
        if model.search_context_options:
            options = [(opt.capitalize(), opt) for opt in model.search_context_options]
            self.search_context_dropdown.options = options
            self.search_context_dropdown.value = "medium" if "medium" in model.search_context_options else model.search_context_options[0]
            self.search_context_dropdown.layout.display = 'flex'
        else:
            self.search_context_dropdown.layout.display = 'none'

        # Show/hide location controls
        if model.supports_location:
            self.location_container.layout.display = 'flex'
        else:
            self.location_container.layout.display = 'none'

    def _on_location_toggle(self, change):
        """Handle location checkbox toggle."""
        show = 'flex' if change['new'] else 'none'
        self.location_country.layout.display = show
        self.location_city.layout.display = show
        self.location_region.layout.display = show

    def _on_confirm(self, _):
        """Handle confirm button click."""
        with self.status_output:
            clear_output()

            if not self.current_cartridge:
                print("Error: No provider selected")
                return

            model_id = self.model_dropdown.value
            if not model_id:
                print("Error: No model selected")
                return

            # Get API key
            api_key = None
            if self.api_key_getter:
                try:
                    api_key = self.api_key_getter(self.current_cartridge.api_key_name)
                except Exception as exc:
                    print(f"Error getting API key: {exc}")
                    return
            else:
                # Try environment variable
                import os
                api_key = os.environ.get(self.current_cartridge.api_key_name)
                if not api_key:
                    print(f"Error: {self.current_cartridge.api_key_name} not found in environment")
                    return

            # Create client
            try:
                self.current_client = self.current_cartridge.create_client(api_key)
            except Exception as exc:
                print(f"Error creating client: {exc}")
                return

            self.current_model_id = model_id

            # Gather parameters
            params = {}

            # Reasoning
            if self.reasoning_dropdown.layout.display != 'none':
                params['reasoning_effort'] = self.reasoning_dropdown.value

            # Search context
            if self.search_context_dropdown.layout.display != 'none':
                params['search_context_size'] = self.search_context_dropdown.value

            # Location
            if self.location_checkbox.value:
                params['location'] = {
                    'country': self.location_country.value,
                    'city': self.location_city.value,
                    'region': self.location_region.value,
                }

            # Success message
            print(f"Selected: {self.current_cartridge.name} / {model_id}")
            print(f"Client initialized")

            # Call callback if provided
            if self.on_confirm:
                self.on_confirm(
                    self.current_cartridge,
                    self.current_client,
                    model_id,
                    params
                )

    def display(self):
        """Display the widget."""
        # Build layout
        title_html = widgets.HTML(f"<h3>{self.title}</h3>")

        desc_html = widgets.HTML(f"<p>{self.description}</p>") if self.description else None

        provider_model_row = widgets.HBox(
            [self.provider_dropdown, self.model_dropdown],
            layout=widgets.Layout(gap='10px'),
        )

        params_row = widgets.HBox(
            [self.reasoning_dropdown, self.search_context_dropdown],
            layout=widgets.Layout(gap='10px'),
        )

        children = [title_html]
        if desc_html:
            children.append(desc_html)
        children.extend([
            provider_model_row,
            params_row,
            self.location_container,
            self.confirm_button,
            self.status_output,
        ])

        container = widgets.VBox(children, layout=widgets.Layout(gap='8px'))
        display(container)

    def get_selection(self) -> Optional[Dict[str, Any]]:
        """Get current selection state.

        Returns:
            Dict with cartridge, client, model_id, and params if confirmed,
            None if not yet confirmed
        """
        if not self.current_client or not self.current_model_id:
            return None

        params = {}
        if self.reasoning_dropdown.layout.display != 'none':
            params['reasoning_effort'] = self.reasoning_dropdown.value
        if self.search_context_dropdown.layout.display != 'none':
            params['search_context_size'] = self.search_context_dropdown.value
        if self.location_checkbox.value:
            params['location'] = {
                'country': self.location_country.value,
                'city': self.location_city.value,
                'region': self.location_region.value,
            }

        return {
            'cartridge': self.current_cartridge,
            'client': self.current_client,
            'model_id': self.current_model_id,
            'params': params,
        }
