# core/report_helper.py
"""
Report helper for the modular bulk prompt runner.

Handles logging and CSV output for batch executions.
Adapted from bulk_loader with added `provider` column for multi-provider support.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


class ReportHelper:
    """Consistent logging/serialization for notebook cells.

    Creates unique execution IDs and manages detail/summary CSV output.
    Includes `provider` column for multi-provider tracking.
    """

    # Detail CSV columns - includes provider for multi-provider support
    DETAIL_COLUMNS: Iterable[str] = [
        "row_timestamp",
        "scenario",
        "execution_id",
        "run_label",
        "unit_id",
        "turn_or_run",
        "role",
        "provider",      # NEW: Provider name (OpenAI, Anthropic, etc.)
        "model",
        "persona_profile",
        "persona_model",
        "query_or_topic",
        "message_text",
        "citation_rank",
        "citation_title",
        "citation_url",
        "domain",
        "context",
        "reasoning",
        "location_country",
        "location_city",
        "location_region",
        "response_file",
    ]

    def __init__(
        self,
        scenario: str,
        paths: Dict[str, Any],
        run_label: Optional[str] = None
    ):
        """Initialize the report helper.

        Args:
            scenario: Name of the scenario/batch type
            paths: Workspace paths dict with 'csv_output' key
            run_label: Optional label for this run
        """
        self.scenario = scenario
        self.paths = paths
        self.output_dir = Path(paths["csv_output"]).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.execution_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        cleaned_label = (run_label or "").strip()
        self.run_label: Optional[str] = cleaned_label or f"{self.scenario}_{timestamp}"
        self.start_ts = datetime.now().isoformat()
        self._detail_rows: list[Dict[str, Any]] = []

    def save_raw_response(self, label: str, response: Any) -> Path:
        """Persist raw response JSON and return the file path.

        Args:
            label: Label for the response file
            response: Response object to save

        Returns:
            Path to the saved JSON file
        """
        raw_path = self.raw_dir / f"{self.scenario}_{self.execution_id}_{label}.json"
        try:
            # Try model_dump for Pydantic models
            payload = response.model_dump()
        except AttributeError:
            try:
                # Try to_dict for dataclasses with that method
                payload = response.to_dict()
            except AttributeError:
                # Fall back to the raw response
                payload = response

        raw_path.write_text(json.dumps(payload, indent=2, default=str))
        return raw_path

    def add_detail_row(self, **row: Any) -> None:
        """Append a single detail row (with default metadata) to the log.

        Args:
            **row: Column values for the row
        """
        row.setdefault("row_timestamp", datetime.now().isoformat())
        row.setdefault("scenario", self.scenario)
        row.setdefault("execution_id", self.execution_id)
        row.setdefault("run_label", self.run_label)
        for column in self.DETAIL_COLUMNS:
            row.setdefault(column, None)
        self._detail_rows.append(row)

    def write_detail_csv(self) -> Path:
        """Flush accumulated detail rows to CSV and return the path.

        Returns:
            Path to the written CSV file
        """
        detail_path = self.output_dir / f"{self.scenario}_detail_{self.execution_id}.csv"
        df = pd.DataFrame(self._detail_rows)
        # Reorder columns if possible
        cols = [c for c in self.DETAIL_COLUMNS if c in df.columns]
        rest = [c for c in df.columns if c not in cols]
        df = df[cols + rest]
        df.to_csv(detail_path, index=False)
        return detail_path

    def write_summary_csv(self, summary_row: Dict[str, Any]) -> Path:
        """Write a one-row summary CSV.

        Args:
            summary_row: Summary data dict

        Returns:
            Path to the written CSV file
        """
        summary_row.setdefault("scenario", self.scenario)
        summary_row.setdefault("execution_id", self.execution_id)
        summary_row.setdefault("run_label", self.run_label)
        summary_row.setdefault("timestamp", datetime.now().isoformat())
        summary_path = self.output_dir / f"{self.scenario}_summary_{self.execution_id}.csv"
        pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
        return summary_path

    def get_detail_dataframe(self) -> pd.DataFrame:
        """Get current detail rows as a DataFrame.

        Returns:
            DataFrame with all accumulated detail rows
        """
        return pd.DataFrame(self._detail_rows)
