# core/__init__.py
"""
Core infrastructure for the modular bulk prompt runner.

This module provides the foundational classes and data structures
used by all provider cartridges and UI components.
"""

from .citation import (
    Citation,
    SearchResponse,
    ChatResponse,
    ModelSchema,
)

from .base_cartridge import BaseCartridge
from .report_helper import ReportHelper
from .batch_runner import BatchRunner

__all__ = [
    'Citation',
    'SearchResponse',
    'ChatResponse',
    'ModelSchema',
    'BaseCartridge',
    'ReportHelper',
    'BatchRunner',
]
