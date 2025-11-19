from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


class ReportHelper:
    """Consistent logging/serialization for notebook cells."""

    DETAIL_COLUMNS: Iterable[str] = [
        "row_timestamp",
        "scenario",
        "execution_id",
        "unit_id",
        "turn_or_run",
        "role",
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

    def __init__(self, scenario: str, paths: Dict[str, Any]):
        self.scenario = scenario
        self.paths = paths
        self.output_dir = Path(paths["csv_output"]).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.execution_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        self.start_ts = datetime.now().isoformat()
        self._detail_rows: list[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def save_raw_response(self, label: str, response: Any) -> Path:
        """Persist raw response JSON and return the file path."""
        raw_path = self.raw_dir / f"{self.scenario}_{self.execution_id}_{label}.json"
        try:
            payload = response.model_dump()  # type: ignore[attr-defined]
        except AttributeError:
            payload = response
        raw_path.write_text(json.dumps(payload, indent=2, default=str))
        return raw_path

    # ------------------------------------------------------------------
    def add_detail_row(self, **row: Any) -> None:
        """Append a single detail row (with default metadata) to the log."""
        row.setdefault("row_timestamp", datetime.now().isoformat())
        row.setdefault("scenario", self.scenario)
        row.setdefault("execution_id", self.execution_id)
        for column in self.DETAIL_COLUMNS:
            row.setdefault(column, None)
        self._detail_rows.append(row)

    # ------------------------------------------------------------------
    def write_detail_csv(self) -> Path:
        """Flush accumulated detail rows to CSV and return the path."""
        detail_path = self.output_dir / f"{self.scenario}_detail_{self.execution_id}.csv"
        df = pd.DataFrame(self._detail_rows)
        # Reorder columns if possible
        cols = [c for c in self.DETAIL_COLUMNS if c in df.columns]
        rest = [c for c in df.columns if c not in cols]
        df = df[cols + rest]
        df.to_csv(detail_path, index=False)
        return detail_path

    # ------------------------------------------------------------------
    def write_summary_csv(self, summary_row: Dict[str, Any]) -> Path:
        """Write a one-row summary CSV."""
        summary_row.setdefault("scenario", self.scenario)
        summary_row.setdefault("execution_id", self.execution_id)
        summary_row.setdefault("timestamp", datetime.now().isoformat())
        summary_path = self.output_dir / f"{self.scenario}_summary_{self.execution_id}.csv"
        pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
        return summary_path
