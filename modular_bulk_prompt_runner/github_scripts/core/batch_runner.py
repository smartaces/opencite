# core/batch_runner.py
"""
Provider-agnostic batch execution engine for the modular bulk prompt runner.

Executes batches of prompts through any provider cartridge, supporting:
- Single-turn and multi-turn conversations
- Parallel execution for faster processing
- Citation extraction and logging
- Persona-based follow-up generation
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

from .base_cartridge import BaseCartridge
from .citation import Citation, SearchResponse, ChatResponse


class BatchRunner:
    """Provider-agnostic batch execution engine.

    Executes prompts through a search agent cartridge, with optional
    multi-turn conversations using a conversation agent cartridge.

    Example:
        runner = BatchRunner(
            search_cartridge=openai_cartridge,
            search_client=search_client,
            search_model="gpt-5.2",
            conversation_cartridge=openai_cartridge,
            conversation_client=conv_client,
            conversation_model="gpt-5-mini",
        )
        results = runner.run_batch(prompts_df, params)
    """

    def __init__(
        self,
        search_cartridge: BaseCartridge,
        search_client: Any,
        search_model: str,
        conversation_cartridge: Optional[BaseCartridge] = None,
        conversation_client: Optional[Any] = None,
        conversation_model: Optional[str] = None,
        report_helper_class: Optional[type] = None,
        paths: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the batch runner.

        Args:
            search_cartridge: Cartridge for web search queries
            search_client: Initialized client for search
            search_model: Model ID for search
            conversation_cartridge: Cartridge for follow-up generation (optional)
            conversation_client: Initialized client for conversation (optional)
            conversation_model: Model ID for conversation (optional)
            report_helper_class: ReportHelper class for logging
            paths: Workspace paths dict
        """
        self.search_cartridge = search_cartridge
        self.search_client = search_client
        self.search_model = search_model

        self.conversation_cartridge = conversation_cartridge
        self.conversation_client = conversation_client
        self.conversation_model = conversation_model

        self.report_helper_class = report_helper_class
        self.paths = paths or {}

        # Execution state
        self._stop_requested = False

    def run_single_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        """Execute a single search query.

        Args:
            query: The search query
            params: Optional parameters (location, reasoning_effort, etc.)

        Returns:
            SearchResponse with text and citations
        """
        params = params or {}
        return self.search_cartridge.search(
            client=self.search_client,
            model_id=self.search_model,
            query=query,
            **params
        )

    def generate_followup(
        self,
        persona: str,
        conversation_history: List[Dict[str, str]],
        topic: str,
    ) -> str:
        """Generate a persona-based follow-up question.

        Args:
            persona: The persona description
            conversation_history: List of previous messages
            topic: The original topic/query

        Returns:
            Generated follow-up question
        """
        if not self.conversation_cartridge or not self.conversation_client:
            raise RuntimeError("Conversation agent not configured")

        prompt = self._build_followup_prompt(persona, conversation_history, topic)

        response = self.conversation_cartridge.chat(
            client=self.conversation_client,
            model_id=self.conversation_model,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.text.strip()

    def _build_followup_prompt(
        self,
        persona: str,
        history: List[Dict[str, str]],
        topic: str,
    ) -> str:
        """Build the prompt for generating follow-up questions."""
        history_text = self._format_history(history[-4:])  # Last 4 messages

        return f"""You are role-playing as the user described as: {persona}

You are conversing with an AI assistant about: {topic}

Read the assistant's latest reply and respond naturally as this user would -
acknowledging what you just learned, sharing reactions or preferences,
or asking a follow-up question that moves the conversation forward.

Stay in character and respond with a single message.
Keep your queries succinct - don't be verbose.

Previous conversation:
{history_text}

Generate only the follow-up question, nothing else."""

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for prompts."""
        lines = []
        for msg in history:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')[:500]  # Truncate long messages
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def execute_single_run(
        self,
        topic: str,
        persona_profile: str,
        turns: int,
        params: Dict[str, Any],
        run_label: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Execute a single run (all turns for one run).

        Args:
            topic: The query/topic
            persona_profile: Persona description for follow-ups
            turns: Number of turns in the conversation
            params: Search parameters (location, reasoning, etc.)
            run_label: Label for this run
            on_progress: Optional callback for progress updates

        Returns:
            Dict with results including citations and paths
        """
        detail_lines = []
        total_citations = 0
        all_citations: List[Citation] = []

        # Create reporter if available
        reporter = None
        if self.report_helper_class and self.paths:
            reporter = self.report_helper_class("multi_turn_batch", self.paths, run_label=run_label)
            execution_id = reporter.execution_id
        else:
            execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        detail_lines.append(f"\n{'='*70}")
        detail_lines.append(f"▶ Run — ID: {execution_id}")
        detail_lines.append(f"  Topic: {topic}")
        detail_lines.append(f"  Provider: {self.search_cartridge.name}")
        detail_lines.append(f"  Model: {self.search_model}")
        if persona_profile:
            detail_lines.append(f"  Persona: {persona_profile}")
        detail_lines.append(f"{'='*70}")

        persona_history: List[Dict[str, str]] = []
        advisor_history: List[Dict[str, str]] = []

        for turn in range(1, turns + 1):
            if self._stop_requested:
                detail_lines.append("\n[STOPPED by user]")
                break

            turn_id = f"{execution_id}_turn_{turn}"

            # Generate query for this turn
            if turn == 1:
                persona_msg = topic
                detail_lines.append(f"\n── Turn {turn}/{turns} ──")
                detail_lines.append(f"[Initial Query]")
                detail_lines.append(f"{persona_msg}")
            else:
                if not self.conversation_cartridge:
                    detail_lines.append(f"\n[Skipping turn {turn} - no conversation agent]")
                    continue

                if on_progress:
                    on_progress(f"Turn {turn}/{turns}: Generating follow-up...")

                persona_msg = self.generate_followup(
                    persona_profile,
                    persona_history + advisor_history,
                    topic
                )
                detail_lines.append(f"\n── Turn {turn}/{turns} ──")
                detail_lines.append(f"[Persona Follow-up]")
                detail_lines.append(f"{persona_msg}")

            persona_history.append({"role": "user", "content": persona_msg})

            # Log persona message
            if reporter:
                location = params.get('location', {})
                reporter.add_detail_row(
                    unit_id=turn_id,
                    turn_or_run=turn,
                    role="persona",
                    provider=self.conversation_cartridge.name if self.conversation_cartridge else "user",
                    model=self.conversation_model or "user",
                    query_or_topic=topic,
                    message_text=persona_msg,
                    citation_rank=None,
                    citation_title=None,
                    citation_url=None,
                    domain=None,
                    context=params.get('search_context_size'),
                    reasoning=params.get('reasoning_effort'),
                    location_country=location.get('country'),
                    location_city=location.get('city'),
                    location_region=location.get('region'),
                    response_file=None,
                    persona_profile=persona_profile,
                )

            # Execute search
            if on_progress:
                on_progress(f"Turn {turn}/{turns}: Searching...")

            detail_lines.append(f"\n[Querying {self.search_cartridge.name}...]")

            try:
                response = self.search_cartridge.search(
                    client=self.search_client,
                    model_id=self.search_model,
                    query=persona_msg,
                    **params
                )
            except Exception as exc:
                detail_lines.append(f"[ERROR] Search failed: {exc}")
                return {
                    "success": False,
                    "detail_lines": detail_lines,
                    "error": str(exc),
                    "total_citations": total_citations,
                }

            # Save raw response
            raw_path = None
            if reporter:
                raw_path = reporter.save_raw_response(f"turn_{turn}", response.raw_response)

            advisor_text = response.text
            advisor_history.append({"role": "assistant", "content": advisor_text})

            # Log response
            detail_lines.append(f"\n[AI Response]")
            detail_lines.append(advisor_text)

            # Process citations
            citations = response.citations
            turn_citations = len(citations)
            total_citations += turn_citations
            all_citations.extend(citations)

            if citations:
                detail_lines.append(f"\n[Citations: {turn_citations}]")
                for cite in citations:
                    detail_lines.append(f"  {cite.position}. {cite.title}")
                    detail_lines.append(f"     {cite.url}")

                    # Log each citation
                    if reporter:
                        parsed = urlparse(cite.url) if cite.url else None
                        domain = parsed.netloc.replace("www.", "") if parsed and parsed.netloc else None
                        location = params.get('location', {})
                        reporter.add_detail_row(
                            unit_id=turn_id,
                            turn_or_run=turn,
                            role="AI System",
                            provider=self.search_cartridge.name,
                            model=self.search_model,
                            query_or_topic=topic,
                            message_text=advisor_text,
                            citation_rank=cite.position,
                            citation_title=cite.title,
                            citation_url=cite.url,
                            domain=domain,
                            context=params.get('search_context_size'),
                            reasoning=params.get('reasoning_effort'),
                            location_country=location.get('country'),
                            location_city=location.get('city'),
                            location_region=location.get('region'),
                            response_file=str(raw_path) if raw_path else None,
                            persona_profile=persona_profile,
                        )
            else:
                detail_lines.append(f"\n[No citations]")
                if reporter:
                    location = params.get('location', {})
                    reporter.add_detail_row(
                        unit_id=turn_id,
                        turn_or_run=turn,
                        role="AI System",
                        provider=self.search_cartridge.name,
                        model=self.search_model,
                        query_or_topic=topic,
                        message_text=advisor_text,
                        citation_rank=None,
                        citation_title=None,
                        citation_url=None,
                        domain=None,
                        context=params.get('search_context_size'),
                        reasoning=params.get('reasoning_effort'),
                        location_country=location.get('country'),
                        location_city=location.get('city'),
                        location_region=location.get('region'),
                        response_file=str(raw_path) if raw_path else None,
                        persona_profile=persona_profile,
                    )

        # Write CSVs
        detail_path = None
        summary_path = None
        if reporter:
            detail_path = reporter.write_detail_csv()

            # Build summary
            df = pd.DataFrame(reporter._detail_rows)
            advisor_citations = df[df["role"] == "AI System"].dropna(subset=["citation_url"])
            domain_counts = advisor_citations["domain"].value_counts().head(5) if len(advisor_citations) > 0 else {}

            location = params.get('location', {})
            summary_row = {
                "provider": self.search_cartridge.name,
                "model": self.search_model,
                "topic": topic,
                "turns": turns,
                "persona_profile": persona_profile,
                "location_country": location.get('country'),
                "location_city": location.get('city'),
                "location_region": location.get('region'),
                "total_citations": len(advisor_citations),
                "unique_citation_urls": advisor_citations["citation_url"].nunique() if len(advisor_citations) > 0 else 0,
                "unique_domains": advisor_citations["domain"].nunique() if len(advisor_citations) > 0 else 0,
            }

            for idx, (domain, count) in enumerate(domain_counts.items(), start=1):
                summary_row[f"top_domain_{idx}"] = domain
                summary_row[f"top_domain_{idx}_count"] = count

            summary_path = reporter.write_summary_csv(summary_row)

            detail_lines.append(f"\n{'─'*70}")
            detail_lines.append(f"💾 Detail CSV: {detail_path.name}")
            detail_lines.append(f"💾 Summary CSV: {summary_path.name}")

        return {
            "success": True,
            "detail_lines": detail_lines,
            "detail_path": detail_path,
            "summary_path": summary_path,
            "total_citations": total_citations,
            "citations": all_citations,
        }

    def run_batch(
        self,
        prompts_df: pd.DataFrame,
        params: Dict[str, Any],
        run_label: str = "",
        parallel_workers: int = 1,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Execute a batch of prompts.

        Args:
            prompts_df: DataFrame with columns: prompt, persona, runs, turns
            params: Search parameters
            run_label: Label for this batch
            parallel_workers: Number of parallel workers (1 = sequential)
            on_progress: Callback fn(message, current, total)

        Returns:
            Dict with batch results
        """
        self._stop_requested = False
        results = []
        total_runs = 0

        # Calculate total runs
        for _, row in prompts_df.iterrows():
            runs = int(row.get('runs', 1)) if pd.notna(row.get('runs')) else 1
            total_runs += max(1, runs)

        current_run = 0

        for idx, row in prompts_df.iterrows():
            if self._stop_requested:
                break

            topic = str(row.get('prompt', ''))
            persona = str(row.get('persona', '')) if pd.notna(row.get('persona')) else ''
            runs = int(row.get('runs', 1)) if pd.notna(row.get('runs')) else 1
            turns = int(row.get('turns', 1)) if pd.notna(row.get('turns')) else 1

            runs = max(1, runs)
            turns = max(1, turns)

            for run_idx in range(1, runs + 1):
                if self._stop_requested:
                    break

                current_run += 1
                if on_progress:
                    on_progress(f"Prompt {idx+1}, Run {run_idx}/{runs}", current_run, total_runs)

                def progress_callback(msg):
                    if on_progress:
                        on_progress(f"Prompt {idx+1}, Run {run_idx}: {msg}", current_run, total_runs)

                result = self.execute_single_run(
                    topic=topic,
                    persona_profile=persona,
                    turns=turns,
                    params=params,
                    run_label=f"{run_label}_p{idx+1}_r{run_idx}" if run_label else f"p{idx+1}_r{run_idx}",
                    on_progress=progress_callback,
                )
                results.append(result)

        return {
            "results": results,
            "total_runs": len(results),
            "successful": sum(1 for r in results if r.get('success')),
            "failed": sum(1 for r in results if not r.get('success')),
            "total_citations": sum(r.get('total_citations', 0) for r in results),
        }

    def stop(self):
        """Request stop of current batch execution."""
        self._stop_requested = True
