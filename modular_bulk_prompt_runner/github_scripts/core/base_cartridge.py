# core/base_cartridge.py
"""
Abstract base class for provider cartridges.

Each AI provider (OpenAI, Anthropic, Perplexity, etc.) implements this interface
to provide standardized search and chat capabilities for the bulk prompt runner.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from .citation import Citation, SearchResponse, ChatResponse, ModelSchema


class BaseCartridge(ABC):
    """Abstract base class for provider cartridges.

    Each cartridge provides two main capabilities:
    1. search() - Web search with citation extraction (for Search Agent)
    2. chat() - Conversation without web search (for Conversation Agent)

    Cartridges are responsible for:
    - Initializing the provider's API client
    - Executing search queries with web search enabled
    - Executing chat completions without web search
    - Extracting citations from provider-specific response formats
    - Converting responses to standardized formats

    Example usage:
        cartridge = OpenAICartridge()
        client = cartridge.create_client(api_key)
        response = cartridge.search(client, "gpt-5.2", "What is Python?")
        print(response.citations)
    """

    # Provider metadata - must be set by subclasses
    name: str = ""                      # e.g., "OpenAI"
    description: str = ""               # e.g., "OpenAI models via Responses API"
    api_key_name: str = ""              # e.g., "OPENAI_API_KEY"
    api_key_secret_name: str = ""       # e.g., "openai_API" (for Colab secrets)

    # Models - loaded from separate schema file by subclass
    models: List[ModelSchema] = []

    @abstractmethod
    def create_client(self, api_key: str) -> Any:
        """Initialize the provider's API client.

        Args:
            api_key: The API key for authentication

        Returns:
            The initialized client object (provider-specific type)
        """
        pass

    @abstractmethod
    def search(
        self,
        client: Any,
        model_id: str,
        query: str,
        location: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        search_context_size: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        **kwargs
    ) -> SearchResponse:
        """Execute a web search query and return response with citations.

        This method is used by the Search Agent to perform web searches
        and extract citations from the AI's response.

        Args:
            client: The initialized API client
            model_id: The model ID to use (e.g., "gpt-5.2")
            query: The search query or prompt
            location: Optional location bias {"city": "...", "country": "...", "region": "..."}
            reasoning_effort: Optional reasoning level (model-specific)
            search_context_size: Optional search depth (model-specific)
            previous_response_id: Optional ID for multi-turn conversations
            **kwargs: Additional provider-specific parameters

        Returns:
            SearchResponse with text, citations, and metadata
        """
        pass

    @abstractmethod
    def chat(
        self,
        client: Any,
        model_id: str,
        messages: List[Dict[str, str]],
        previous_response_id: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """Execute a chat completion without web search.

        This method is used by the Conversation Agent to generate
        persona-based follow-up questions for multi-turn conversations.
        No web search is performed, so no citations are returned.

        Args:
            client: The initialized API client
            model_id: The model ID to use
            messages: List of message dicts [{"role": "user", "content": "..."}]
            previous_response_id: Optional ID for multi-turn conversations
            **kwargs: Additional provider-specific parameters

        Returns:
            ChatResponse with text and metadata (no citations)
        """
        pass

    @abstractmethod
    def extract_citations(self, raw_response: Any) -> List[Citation]:
        """Extract citations from provider-specific response format.

        Each provider returns citations in a different format. This method
        normalizes them to the standard Citation format.

        Args:
            raw_response: The raw response object from the provider's API

        Returns:
            List of Citation objects extracted from the response
        """
        pass

    def get_model(self, model_id: str) -> Optional[ModelSchema]:
        """Get model schema by ID.

        Args:
            model_id: The model ID to look up

        Returns:
            ModelSchema if found, None otherwise
        """
        return next((m for m in self.models if m.id == model_id), None)

    def get_models_with_search(self) -> List[ModelSchema]:
        """Get all models that support native web search.

        Returns:
            List of ModelSchema objects where native_search=True
        """
        return [m for m in self.models if m.native_search]

    def get_supported_params(self, model_id: str) -> Dict[str, Any]:
        """Get supported parameters for a specific model.

        Used by the UI to determine which parameter controls to show.

        Args:
            model_id: The model ID to check

        Returns:
            Dict with parameter support info
        """
        model = self.get_model(model_id)
        if not model:
            return {}
        return {
            "location": model.supports_location,
            "reasoning": model.supports_reasoning,
            "reasoning_options": model.reasoning_options,
            "search_context_options": model.search_context_options,
            "additional_params": model.additional_params,
        }

    def validate_model(self, model_id: str) -> bool:
        """Check if a model ID is valid for this cartridge.

        Args:
            model_id: The model ID to validate

        Returns:
            True if model exists in this cartridge's schema
        """
        return self.get_model(model_id) is not None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', models={len(self.models)})"
