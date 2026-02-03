# providers/openai_cartridge.py
"""
OpenAI cartridge for the modular bulk prompt runner.

Implements the BaseCartridge interface for OpenAI's Responses API with
native web search support. Supports GPT-5.x series, o3/o4 reasoning models.

Key features:
- Web search via Responses API with "web_search_preview" tool
- Location bias for search results
- Reasoning effort control (gpt-5.x family)
- Multi-turn conversations via previous_response_id
- Citation extraction from response annotations
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_cartridge import BaseCartridge
from core.citation import Citation, SearchResponse, ChatResponse, ModelSchema
from schemas.openai_models import OPENAI_MODELS, DEFAULT_MODEL_ID


class OpenAICartridge(BaseCartridge):
    """OpenAI cartridge using the Responses API with native web search.

    Supports:
    - GPT-5.2, GPT-5.2-pro, GPT-5.1, GPT-5, GPT-5-mini (with reasoning)
    - o3, o3-pro, o4-mini (reasoning models, no search_context_size)
    - GPT-4.1 (legacy, no reasoning)

    Example usage:
        cartridge = OpenAICartridge()
        client = cartridge.create_client(api_key)

        # Web search with citations
        response = cartridge.search(client, "gpt-5.2", "What is Python?")
        print(response.citations)

        # Chat without web search (for follow-ups)
        chat_response = cartridge.chat(client, "gpt-5.2", [{"role": "user", "content": "Tell me more"}])
        print(chat_response.text)
    """

    name = "OpenAI"
    description = "OpenAI models via Responses API with native web search"
    api_key_name = "OPENAI_API_KEY"
    api_key_secret_name = "openai_API"

    # Models loaded from schema file
    models = OPENAI_MODELS

    # Models that don't support search_context_size
    _NO_SEARCH_CONTEXT_MODELS = {"o3", "o3-pro", "o4-mini"}

    # Models that support reasoning effort parameter
    _REASONING_MODEL_PREFIXES = ("gpt-5", "o3", "o4")

    def create_client(self, api_key: str) -> Any:
        """Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key

        Returns:
            Initialized OpenAI client
        """
        from openai import OpenAI
        return OpenAI(api_key=api_key)

    def search(
        self,
        client: Any,
        model_id: str,
        query: str,
        location: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        search_context_size: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        verbosity: str = "medium",
        force_search: bool = False,
        **kwargs
    ) -> SearchResponse:
        """Execute a web search query via OpenAI Responses API.

        Args:
            client: Initialized OpenAI client
            model_id: Model ID (e.g., "gpt-5.2")
            query: The search query or prompt
            location: Optional location bias {"city": "...", "country": "...", "region": "..."}
            reasoning_effort: Reasoning level ("high", "medium", "low")
            search_context_size: Search depth ("low", "medium", "high")
            previous_response_id: For multi-turn conversations
            verbosity: Response verbosity ("short", "medium", "long")
            force_search: Force web search tool usage
            **kwargs: Additional parameters

        Returns:
            SearchResponse with text, citations, and metadata
        """
        # Build web search tool config
        tools: List[Dict[str, Any]] = []
        web_search_config: Dict[str, Any] = {"type": "web_search_preview"}

        # search_context_size not supported on o3, o3-pro, o4-mini
        if model_id not in self._NO_SEARCH_CONTEXT_MODELS:
            if search_context_size:
                web_search_config["search_context_size"] = search_context_size
            else:
                web_search_config["search_context_size"] = "medium"

        # Add location bias if provided
        if location:
            web_search_config["user_location"] = {
                "type": "approximate",
                **location,
            }

        tools.append(web_search_config)

        # Build request parameters
        request_params: Dict[str, Any] = {
            "model": model_id,
            "tools": tools,
            "input": query,
        }

        # GPT-5 family supports reasoning and verbosity parameters
        if model_id.startswith("gpt-5"):
            if reasoning_effort:
                # Normalize "minimal" to "low" for API compatibility
                if reasoning_effort == "minimal":
                    reasoning_effort = "low"
                request_params["reasoning"] = {"effort": reasoning_effort}

            request_params["text"] = {"verbosity": verbosity}

            # Multi-turn support
            if previous_response_id:
                request_params["previous_response_id"] = previous_response_id

        # o3/o4 models with reasoning
        elif model_id.startswith(("o3", "o4")):
            if reasoning_effort:
                request_params["reasoning"] = {"effort": reasoning_effort}
            if previous_response_id:
                request_params["previous_response_id"] = previous_response_id

        # Force web search if requested
        if force_search:
            request_params["tool_choice"] = {"type": "web_search_preview"}

        # Execute request
        try:
            response = client.responses.create(**request_params)
        except Exception as exc:
            # Return error response
            return SearchResponse(
                text=f"Error: {str(exc)}",
                citations=[],
                raw_response={"error": str(exc)},
                model=model_id,
                provider=self.name,
                response_id=None,
            )

        # Extract response ID for multi-turn
        response_id = getattr(response, "id", None)

        # Extract text and citations
        text = self._extract_text(response)
        citations = self.extract_citations(response)

        return SearchResponse(
            text=text,
            citations=citations,
            raw_response=response,
            model=model_id,
            provider=self.name,
            response_id=response_id,
        )

    def chat(
        self,
        client: Any,
        model_id: str,
        messages: List[Dict[str, str]],
        previous_response_id: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """Execute a chat completion without web search.

        Used for generating persona-based follow-up questions in multi-turn
        conversations. No web search is performed.

        Args:
            client: Initialized OpenAI client
            model_id: Model ID (e.g., "gpt-5.2")
            messages: List of message dicts [{"role": "user", "content": "..."}]
            previous_response_id: For multi-turn conversations
            **kwargs: Additional parameters

        Returns:
            ChatResponse with text and metadata (no citations)
        """
        # Build request parameters - NO web search tools
        request_params: Dict[str, Any] = {
            "model": model_id,
            "input": messages if isinstance(messages, str) else messages[-1].get("content", ""),
        }

        # Multi-turn support
        if previous_response_id:
            request_params["previous_response_id"] = previous_response_id

        # Execute request
        try:
            response = client.responses.create(**request_params)
        except Exception as exc:
            return ChatResponse(
                text=f"Error: {str(exc)}",
                raw_response={"error": str(exc)},
                model=model_id,
                provider=self.name,
                response_id=None,
            )

        # Extract response ID for multi-turn
        response_id = getattr(response, "id", None)

        # Extract text
        text = self._extract_text(response)

        return ChatResponse(
            text=text,
            raw_response=response,
            model=model_id,
            provider=self.name,
            response_id=response_id,
        )

    def extract_citations(self, raw_response: Any) -> List[Citation]:
        """Extract citations from OpenAI Responses API format.

        Navigates: response.output[].content[].annotations[]
        Looking for annotations with type="url_citation"

        Args:
            raw_response: Raw response from OpenAI Responses API

        Returns:
            List of Citation objects
        """
        citations: List[Citation] = []
        position = 1

        try:
            if hasattr(raw_response, "output"):
                for output in raw_response.output:
                    # Handle object-style response
                    if getattr(output, "type", None) == "message" and hasattr(output, "content"):
                        for item in output.content:
                            if getattr(item, "type", None) == "output_text":
                                annotations = getattr(item, "annotations", [])
                                for ann in annotations:
                                    if getattr(ann, "type", None) == "url_citation":
                                        citations.append(Citation(
                                            url=getattr(ann, "url", ""),
                                            title=getattr(ann, "title", "Untitled"),
                                            position=position,
                                        ))
                                        position += 1

                    # Handle dict-style response
                    elif isinstance(output, dict) and output.get("type") == "message":
                        for item in output.get("content", []):
                            if item.get("type") == "output_text":
                                for ann in item.get("annotations", []):
                                    if ann.get("type") == "url_citation":
                                        citations.append(Citation(
                                            url=ann.get("url", ""),
                                            title=ann.get("title", "Untitled"),
                                            position=position,
                                        ))
                                        position += 1
        except Exception:
            pass  # Return empty list on extraction errors

        return citations

    def _extract_text(self, response: Any) -> str:
        """Extract text content from OpenAI Responses API response.

        Args:
            response: Raw response from API

        Returns:
            Extracted text or error message
        """
        try:
            # Check for direct output_text attribute
            if hasattr(response, "output_text"):
                return response.output_text

            # Navigate response.output[].content[]
            if hasattr(response, "output"):
                for output in response.output:
                    # Object-style response
                    if getattr(output, "type", None) == "message" and hasattr(output, "content"):
                        for item in output.content:
                            if getattr(item, "type", None) == "output_text":
                                return getattr(item, "text", "")

                    # Dict-style response
                    elif isinstance(output, dict) and output.get("type") == "message":
                        for item in output.get("content", []):
                            if item.get("type") == "output_text":
                                return item.get("text", "")

            return "No text response found"
        except Exception as exc:
            return f"Error extracting response: {exc}"

    def get_search_actions(self, response: Any) -> List[Dict[str, Any]]:
        """Extract web search actions from response (for debugging).

        Args:
            response: Raw response from API

        Returns:
            List of search action details
        """
        actions: List[Dict[str, Any]] = []
        try:
            if hasattr(response, "output"):
                for output in response.output:
                    if getattr(output, "type", None) == "web_search_call":
                        actions.append({
                            "id": getattr(output, "id", ""),
                            "status": getattr(output, "status", ""),
                            "action": getattr(output, "action", {}),
                        })
                    elif isinstance(output, dict) and output.get("type") == "web_search_call":
                        actions.append({
                            "id": output.get("id", ""),
                            "status": output.get("status", ""),
                            "action": output.get("action", {}),
                        })
        except Exception:
            pass
        return actions


# =============================================================================
# Helper Functions
# =============================================================================

def format_location(
    country: str = "US",
    city: str = "New York",
    region: str = "New York"
) -> Dict[str, str]:
    """Create a location dict compatible with OpenAI web search requests.

    Args:
        country: Country code (e.g., "US", "UK")
        city: City name
        region: Region/state name

    Returns:
        Location dict for search() method
    """
    return {
        "country": country,
        "city": city,
        "region": region,
    }
