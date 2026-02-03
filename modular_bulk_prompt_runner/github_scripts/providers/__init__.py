# providers/__init__.py
"""
Provider cartridges for the modular bulk prompt runner.

Each provider (OpenAI, Anthropic, etc.) has its own cartridge module
that implements the BaseCartridge interface.
"""

# Cartridges are imported dynamically by the provider selector
# to avoid requiring all provider SDKs to be installed

__all__ = []
