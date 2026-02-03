# providers/google_cartridge.py
"""
Google Gemini provider cartridge.

Uses the generateContent API with Google Search grounding.
Supports thinking configuration for Gemini 3 (thinkingLevel) and
Gemini 2.5 (thinkingBudget) models.

SDK: google-genai>=1.0.0
Docs: https://ai.google.dev/gemini-api/docs/google-search
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from core.base_cartridge import BaseCartridge
from core.citation import Citation, SearchResponse, ChatResponse, ModelSchema
from schemas.google_models import GOOGLE_MODELS


class GoogleCartridge(BaseCartridge):
    """Google Gemini cartridge with Google Search grounding support.

    Features:
    - Google Search grounding for real-time web information
    - Thinking configuration (thinkingLevel for Gemini 3, thinkingBudget for 2.5)
    - Citation extraction from grounding_metadata
    - Chat support for multi-turn conversations

    Note: Location is not natively supported. A prompt workaround is used
    to append location context to queries.
    """

    name = "Google"
    description = "Google Gemini models with Google Search grounding"
    api_key_name = "GOOGLE_API_KEY"
    api_key_secret_name = "Google_Gemini_API"
    models = GOOGLE_MODELS

    def create_client(self, api_key: str) -> Any:
        """Initialize the Google GenAI client.

        Args:
            api_key: Google API key

        Returns:
            Initialized genai.Client instance
        """
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Run: pip install google-genai>=1.0.0"
            )

        # Set API key in environment (required by SDK)
        os.environ["GOOGLE_API_KEY"] = api_key
        return genai.Client()

    def search(
        self,
        client: Any,
        model_id: str,
        query: str,
        location: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        search_context_size: Optional[str] = None,  # Not supported by Gemini
        previous_response_id: Optional[str] = None,  # Not used (stateless API)
        **kwargs
    ) -> SearchResponse:
        """Execute search with Google Search grounding.

        Args:
            client: Initialized genai.Client
            model_id: Gemini model ID (e.g., "gemini-2.5-flash")
            query: Search query
            location: Optional location dict with city, country, region
            reasoning_effort: Thinking level/budget setting
            search_context_size: Not supported (ignored)
            previous_response_id: Not used in generateContent API

        Returns:
            SearchResponse with text, citations, and raw response
        """
        from google.genai import types

        # Build config with search tool
        config_params = {
            "tools": [{"google_search": {}}]
        }

        # Add thinking config if model supports it
        thinking_config = self._build_thinking_config(model_id, reasoning_effort)
        if thinking_config:
            config_params["thinking_config"] = thinking_config

        config = types.GenerateContentConfig(**config_params)

        # Location workaround (no native support in Gemini)
        query_text = query
        if location:
            loc_parts = []
            if location.get('city'):
                loc_parts.append(location['city'])
            if location.get('region'):
                loc_parts.append(location['region'])
            if location.get('country'):
                loc_parts.append(location['country'])
            if loc_parts:
                loc_str = ", ".join(loc_parts)
                query_text = f"{query}\n\n(Provide information relevant to someone in {loc_str})"

        # Execute request
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=query_text,
                config=config,
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")

        # Extract text
        text = ""
        if response.text:
            text = response.text
        elif response.candidates and response.candidates[0].content:
            parts = response.candidates[0].content.parts
            if parts:
                text = parts[0].text if hasattr(parts[0], 'text') else ""

        # Extract citations and return
        return SearchResponse(
            text=text,
            citations=self.extract_citations(response),
            raw_response=response,
            model=model_id,
            provider=self.name,
        )

    def chat(
        self,
        client: Any,
        model_id: str,
        messages: List[Dict[str, str]],
        previous_response_id: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """Execute chat completion without search grounding.

        Used for conversation agent to generate follow-up questions.
        No web search is performed.

        Args:
            client: Initialized genai.Client
            model_id: Gemini model ID
            messages: List of message dicts with 'role' and 'content'
            previous_response_id: Not used in generateContent API

        Returns:
            ChatResponse with text and raw response
        """
        from google.genai import types

        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Gemini uses "model" instead of "assistant"
            if role == "assistant":
                role = "model"

            contents.append({
                "role": role,
                "parts": [{"text": content}]
            })

        # Execute request (no tools = no search)
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")

        # Extract text
        text = ""
        if response.text:
            text = response.text
        elif response.candidates and response.candidates[0].content:
            parts = response.candidates[0].content.parts
            if parts:
                text = parts[0].text if hasattr(parts[0], 'text') else ""

        return ChatResponse(
            text=text,
            raw_response=response,
            model=model_id,
            provider=self.name,
        )

    def extract_citations(self, raw_response: Any) -> List[Citation]:
        """Extract citations from Gemini grounding metadata.

        Citations are extracted from:
        response.candidates[0].grounding_metadata.grounding_chunks[].web

        Args:
            raw_response: Raw Gemini API response

        Returns:
            List of Citation objects
        """
        citations = []

        # Check for valid response structure
        if not hasattr(raw_response, 'candidates') or not raw_response.candidates:
            return citations

        candidate = raw_response.candidates[0]
        grounding_metadata = getattr(candidate, 'grounding_metadata', None)

        if not grounding_metadata:
            return citations

        # Extract from grounding_chunks
        chunks = getattr(grounding_metadata, 'grounding_chunks', None) or []

        for idx, chunk in enumerate(chunks):
            web = getattr(chunk, 'web', None)
            if web:
                url = getattr(web, 'uri', '') or ''
                title = getattr(web, 'title', '') or 'Untitled'

                citations.append(Citation(
                    url=url,
                    title=title,
                    position=idx + 1,
                ))

        return citations

    def _build_thinking_config(
        self,
        model_id: str,
        reasoning_effort: Optional[str]
    ) -> Optional[Any]:
        """Build thinking config based on model type.

        Gemini 3 models use thinking_level (minimal, low, medium, high).
        Gemini 2.5 models use thinking_budget (token count or -1 for dynamic).

        Args:
            model_id: Model ID to get schema for
            reasoning_effort: User-selected reasoning level

        Returns:
            ThinkingConfig object or None
        """
        from google.genai import types

        model = self.get_model(model_id)
        if not model or not model.supports_reasoning:
            return None

        if not reasoning_effort:
            return None  # Use model default

        additional = model.additional_params or {}
        thinking_param = additional.get("thinking_param")

        if thinking_param == "thinking_level":
            # Gemini 3: Use thinking_level directly
            # Validate against allowed options
            if model.reasoning_options and reasoning_effort in model.reasoning_options:
                return types.ThinkingConfig(thinking_level=reasoning_effort)

        elif thinking_param == "thinking_budget":
            # Gemini 2.5: Map to budget value
            budget_map = additional.get("budget_map", {})
            budget = budget_map.get(reasoning_effort)

            if budget is not None:
                return types.ThinkingConfig(thinking_budget=budget)

        return None

    def get_search_queries(self, raw_response: Any) -> List[str]:
        """Extract search queries used by the model.

        Useful for debugging and understanding model reasoning.

        Args:
            raw_response: Raw Gemini API response

        Returns:
            List of search query strings
        """
        if not hasattr(raw_response, 'candidates') or not raw_response.candidates:
            return []

        candidate = raw_response.candidates[0]
        grounding_metadata = getattr(candidate, 'grounding_metadata', None)

        if not grounding_metadata:
            return []

        queries = getattr(grounding_metadata, 'web_search_queries', None) or []
        return list(queries)

    def get_grounding_supports(self, raw_response: Any) -> List[Dict[str, Any]]:
        """Extract grounding supports (text-to-source mappings).

        Each support links a text segment to source indices, useful
        for building inline citations.

        Args:
            raw_response: Raw Gemini API response

        Returns:
            List of support dicts with segment info and chunk indices
        """
        if not hasattr(raw_response, 'candidates') or not raw_response.candidates:
            return []

        candidate = raw_response.candidates[0]
        grounding_metadata = getattr(candidate, 'grounding_metadata', None)

        if not grounding_metadata:
            return []

        supports = getattr(grounding_metadata, 'grounding_supports', None) or []
        result = []

        for support in supports:
            segment = getattr(support, 'segment', None)
            if segment:
                result.append({
                    'start_index': getattr(segment, 'start_index', 0),
                    'end_index': getattr(segment, 'end_index', 0),
                    'text': getattr(segment, 'text', ''),
                    'chunk_indices': list(getattr(support, 'grounding_chunk_indices', []) or []),
                })

        return result
