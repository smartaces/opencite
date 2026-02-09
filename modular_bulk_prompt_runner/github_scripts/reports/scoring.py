"""Shared scoring functions for citation intelligence reports."""
from __future__ import annotations

import pandas as pd


def compute_rank_quality_score(ranks: pd.Series) -> float:
    """Score 0-100 based on citation rank positions (1-10 get points)."""
    if ranks.empty:
        return 0.0
    capped = ranks.clip(lower=1, upper=50)
    points = capped.apply(lambda r: max(0, 11 - min(int(r), 10)))
    return float(points.sum() / (10 * len(points)) * 100)


def compute_topical_authority_score(
    unique_prompt_pct: float, avg_rank_unique: float | None
) -> float:
    """Average of prompt coverage % and rank quality component."""
    if avg_rank_unique is None or pd.isna(avg_rank_unique):
        rank_component = 50.0
    else:
        rank_component = max(0.0, (10 - min(avg_rank_unique, 10)) / 9 * 100)
    return (unique_prompt_pct + rank_component) / 2


def label_source_character(rate: float) -> str:
    """Classify based on prompt repetition rate."""
    if rate >= 2.0:
        return "Niche Specialist"
    if rate >= 1.2:
        return "Focused Authority"
    return "General Authority"


def compute_predictability_score(outputs_pct: float, top3_pct: float) -> float:
    """Average of output presence and top-3 ranking."""
    return (outputs_pct + top3_pct) / 2


def compute_overall_impact_score(
    predictability: float,
    authority: float,
    velocity: float,
    share: float,
) -> float:
    """Weighted composite: 0.4 predictability + 0.3 authority + 0.2 velocity + 0.1 share."""
    return (
        (predictability * 0.4)
        + (authority * 0.3)
        + (velocity * 0.2)
        + (share * 0.1)
    )
