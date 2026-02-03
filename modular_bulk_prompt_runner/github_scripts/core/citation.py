# core/citation.py
"""
Standardized data formats for the modular bulk prompt runner.

This module defines the common data structures used across all provider cartridges
to ensure consistent citation tracking regardless of which AI provider is used.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Citation:
    """Standardized citation format across all providers.

    Represents a single URL citation extracted from an AI response.
    All provider cartridges must convert their native citation format
    to this standardized structure.
    """
    url: str
    title: str
    snippet: Optional[str] = None
    position: Optional[int] = None  # Position/rank in the response (1-indexed)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON output."""
        return {
            'url': self.url,
            'title': self.title,
            'snippet': self.snippet,
            'position': self.position,
        }


@dataclass
class SearchResponse:
    """Standardized search response format across all providers.

    Returned by cartridge.search() method. Contains the response text,
    extracted citations, and metadata about the request.
    """
    text: str                       # Full response text
    citations: List[Citation]       # Extracted citations
    raw_response: Any               # Original provider response (for debugging)
    model: str                      # Model ID used
    provider: str                   # Provider name
    response_id: Optional[str] = None  # Provider's response ID (for multi-turn)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/output."""
        return {
            'text': self.text,
            'citations': [c.to_dict() for c in self.citations],
            'model': self.model,
            'provider': self.provider,
            'response_id': self.response_id,
            'citation_count': len(self.citations),
        }


@dataclass
class ChatResponse:
    """Standardized chat response format (for conversation agent).

    Returned by cartridge.chat() method. Used for multi-turn conversations
    where we need to generate follow-up questions without web search.
    """
    text: str                       # Response text
    raw_response: Any               # Original provider response
    model: str                      # Model ID used
    provider: str                   # Provider name
    response_id: Optional[str] = None  # Provider's response ID (for multi-turn)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/output."""
        return {
            'text': self.text,
            'model': self.model,
            'provider': self.provider,
            'response_id': self.response_id,
        }


@dataclass
class ModelSchema:
    """Schema defining a model's capabilities.

    Each provider's schema file defines a list of ModelSchema objects
    describing what features each model supports. The UI uses this
    information to show/hide parameter controls dynamically.
    """
    id: str                                     # Model ID (e.g., "gpt-5.2")
    name: str                                   # Display name (e.g., "GPT-5.2")
    native_search: bool                         # Supports web search with citations
    supports_location: bool                     # Supports location bias for search
    supports_reasoning: bool                    # Has reasoning/thinking mode
    reasoning_options: Optional[List[str]] = None       # e.g., ["high", "medium", "low"]
    search_context_options: Optional[List[str]] = None  # e.g., ["low", "medium", "high"]
    additional_params: Optional[Dict[str, Any]] = None  # Provider-specific params

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'native_search': self.native_search,
            'supports_location': self.supports_location,
            'supports_reasoning': self.supports_reasoning,
            'reasoning_options': self.reasoning_options,
            'search_context_options': self.search_context_options,
            'additional_params': self.additional_params,
        }
