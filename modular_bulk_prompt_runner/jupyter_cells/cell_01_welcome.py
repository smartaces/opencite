# cell_01_welcome.py
"""
Welcome cell for the Modular Bulk Prompt Runner.

This notebook enables batch processing of prompts through multiple AI providers
with web search capabilities. Track citations, analyze responses, and compare
results across different models.

Supported Providers:
- OpenAI (GPT-5.x series with native web search)
- More providers coming soon (Anthropic, Perplexity, Google, xAI)

Run cells in order from top to bottom.
"""

print("""
================================================================================
                     MODULAR BULK PROMPT RUNNER
================================================================================

This notebook lets you:

  1. Configure a workspace to store your data
  2. Choose your AI provider and model (OpenAI, etc.)
  3. Upload prompts from a CSV file
  4. Run batch searches with optional personas and multi-turn conversations
  5. Analyze citation patterns and generate reports

--------------------------------------------------------------------------------
CELL ORDER:
--------------------------------------------------------------------------------

  Cell 01: Welcome (this cell)
  Cell 02: Workspace Setup - Configure storage location
  Cell 03: Download Scripts - Get the latest modular scripts
  Cell 04a: Search Agent Setup - Select provider/model for web search
  Cell 04b: Conversation Agent Setup (Optional) - For multi-turn follow-ups
  Cell 05: Upload Prompts - Load your CSV file
  Cell 06: Run Batch - Execute searches
  Cell 07: Browse Results - View output files
  Cell 08: Reports - Generate domain/page/prompt analysis

--------------------------------------------------------------------------------
REQUIREMENTS:
--------------------------------------------------------------------------------

  - API key for your chosen provider (e.g., OPENAI_API_KEY)
  - A CSV file with prompts (columns: prompt, persona, runs, turns)

================================================================================
                     Run the next cell to set up your workspace
================================================================================
""")
