# providers/google_cartridge.py
"""
Google Gemini provider cartridge.

Uses the generateContent API with Google Search grounding.
Supports thinking configuration for Gemini 3 (thinkingLevel) and
Gemini 2.5 (thinkingBudget) models.

Includes URL resolution for converting Vertex redirect URLs to actual
destination URLs with progressive retry backoff.

SDK: google-genai>=1.0.0
Docs: https://ai.google.dev/gemini-api/docs/google-search
"""

from __future__ import annotations

import os
import random
import time
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple

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
    api_key_secret_name = "GEMINI_API_KEY"
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
        resolve_redirects: bool = False,
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
            resolve_redirects: If True, resolve Vertex redirect URLs to actual destinations

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

        # Extract citations
        citations = self.extract_citations(response)

        # Optionally resolve redirect URLs to actual destinations
        if resolve_redirects and citations:
            citations = self.resolve_citations(citations)

        return SearchResponse(
            text=text,
            citations=citations,
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

        For Google responses:
        - uri contains a Vertex redirect URL
        - title contains just the domain (e.g., "techradar.com")

        Args:
            raw_response: Raw Gemini API response

        Returns:
            List of Citation objects with Google-specific fields populated
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
                redirect_url = getattr(web, 'uri', '') or ''
                google_domain = getattr(web, 'title', '') or 'Unknown'

                citations.append(Citation(
                    url=redirect_url,  # Initially set to redirect URL
                    title=google_domain,  # Google only provides domain as title
                    position=idx + 1,
                    domain=google_domain,  # Use Google's domain initially
                    redirect_url=redirect_url,  # Store original redirect URL
                    google_domain=google_domain,  # Store Google's domain as fallback
                    resolution_status='pending',  # Will be updated if resolution runs
                ))

        return citations

    def resolve_citations(
        self,
        citations: List[Citation],
        max_attempts: int = 4,
        base_delay_range: Tuple[float, float] = (3.0, 5.0),
        increment_range: Tuple[float, float] = (3.0, 5.0),
        request_timeout: float = 5.0,
    ) -> List[Citation]:
        """Resolve redirect URLs in citations to actual destination URLs.

        Uses progressive backoff with randomized delays between attempts.
        Falls back to Google's domain if resolution fails after max attempts.

        Args:
            citations: List of Citation objects with redirect URLs
            max_attempts: Maximum retry attempts per URL (default: 4)
            base_delay_range: (min, max) seconds for base delay
            increment_range: (min, max) seconds to add per retry
            request_timeout: Timeout for each HEAD request in seconds

        Returns:
            List of Citation objects with resolved URLs (or fallbacks)
        """
        resolved_citations = []

        for i, citation in enumerate(citations):
            # Skip if no redirect URL to resolve
            if not citation.redirect_url:
                resolved_citations.append(citation)
                continue

            # Attempt to resolve the redirect URL
            resolved_url, status, note = self._resolve_single_url(
                citation.redirect_url,
                max_attempts=max_attempts,
                base_delay_range=base_delay_range,
                increment_range=increment_range,
                request_timeout=request_timeout,
            )

            if resolved_url:
                # Success: extract domain from resolved URL
                domain = self._extract_domain(resolved_url)
                citation.url = resolved_url
                citation.domain = domain
                citation.title = domain  # Update title to match OpenAI behavior
                citation.resolution_status = status
            else:
                # Failure: fall back to Google's domain
                citation.url = None  # No actual URL available
                citation.domain = citation.google_domain  # Use fallback
                citation.resolution_status = status
                citation.resolution_note = note or "webpage data not available"

            resolved_citations.append(citation)

            # Delay between citations (not after the last one)
            if i < len(citations) - 1:
                delay = random.uniform(*base_delay_range)
                time.sleep(delay)

        return resolved_citations

    def _resolve_single_url(
        self,
        redirect_url: str,
        max_attempts: int,
        base_delay_range: Tuple[float, float],
        increment_range: Tuple[float, float],
        request_timeout: float,
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """Resolve a single redirect URL with progressive retry backoff.

        Args:
            redirect_url: The Vertex redirect URL to resolve
            max_attempts: Maximum number of attempts
            base_delay_range: (min, max) for base delay
            increment_range: (min, max) for delay increment
            request_timeout: Timeout per request

        Returns:
            Tuple of (resolved_url or None, status, note)
        """
        try:
            import requests
        except ImportError:
            return None, "error", "requests library not installed"

        base_delay = random.uniform(*base_delay_range)
        increment = random.uniform(*increment_range)

        last_error = None

        for attempt in range(max_attempts):
            try:
                response = requests.head(
                    redirect_url,
                    allow_redirects=True,
                    timeout=request_timeout,
                )

                # Check if we got redirected to a real URL (not still on vertex)
                final_url = response.url
                is_resolved = not final_url.startswith("https://vertexaisearch.cloud.google.com/")

                # Success - we have a resolved URL (regardless of status code)
                # Sites may return 403 for HEAD requests but we still got the real URL
                if is_resolved:
                    status = "success" if response.status_code == 200 else f"resolved_http_{response.status_code}"
                    return final_url, status, None

                # 404 on the redirect URL itself means expired
                if response.status_code == 404:
                    return None, "expired", "URL returned 404"

                # Still on vertex URL - retry
                last_error = f"HTTP {response.status_code}"

            except requests.exceptions.Timeout:
                last_error = "timeout"
            except requests.exceptions.ConnectionError:
                last_error = "connection error"
            except requests.exceptions.TooManyRedirects:
                last_error = "too many redirects"
            except requests.exceptions.RequestException as e:
                last_error = str(e)

            # If not the last attempt, wait with progressive backoff
            if attempt < max_attempts - 1:
                # Progressive delay: base + (attempt * increment) with randomization
                delay = base_delay + (attempt * increment)
                # Add slight randomization to the calculated delay
                delay = delay + random.uniform(-0.5, 0.5)
                delay = max(0.5, delay)  # Ensure minimum delay
                time.sleep(delay)

        # All attempts failed
        return None, "failed", last_error

    def _extract_domain(self, url: str) -> str:
        """Extract domain from a URL.

        Args:
            url: Full URL

        Returns:
            Domain string (e.g., "techradar.com")
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain or "unknown"
        except Exception:
            return "unknown"

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
