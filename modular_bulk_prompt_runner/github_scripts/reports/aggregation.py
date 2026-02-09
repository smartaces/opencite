"""Generic group-level metrics engine and shared aggregation context.

Replaces the ~80% duplicated aggregation logic from bulk_loader
cells 09/10/11 with a single reusable computation path.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .scoring import (
    compute_overall_impact_score,
    compute_predictability_score,
    compute_rank_quality_score,
    compute_topical_authority_score,
    label_source_character,
)


# ------------------------------------------------------------------
# Shared context (computed once after Tier 1 filtering)
# ------------------------------------------------------------------

def compute_aggregation_context(
    enriched_df: pd.DataFrame,
    ai_role: str = "AI System",
) -> Dict[str, Any]:
    """Compute shared totals needed by all report builders.

    Parameters
    ----------
    enriched_df : DataFrame
        Already-enriched DataFrame (has ``run_id``, ``prompt_run_id``,
        ``clean_url``, ``domain``, numeric ``citation_rank``, UTC
        ``row_timestamp``).
    ai_role : str
        The role label for AI responses (default ``"AI System"``).

    Returns
    -------
    dict
        Keys: ``citations``, ``total_outputs``, ``total_prompt_runs``,
        ``total_unique_prompts``, ``prompt_run_counts``, ``domain_totals``,
        ``now_ts``, ``window_start``.
    """
    # Filter to AI role rows with valid citations
    if "role" in enriched_df.columns:
        advisor = enriched_df[enriched_df["role"] == ai_role].copy()
    else:
        advisor = enriched_df.copy()

    citations = advisor.dropna(subset=["citation_url"]).copy()
    citations = citations[citations["clean_url"] != ""]
    citations = citations[citations["domain"] != ""]

    if citations.empty:
        now_ts = pd.Timestamp.utcnow().tz_localize("UTC")
        return {
            "citations": citations,
            "total_outputs": 0,
            "total_prompt_runs": 0,
            "total_unique_prompts": 0,
            "prompt_run_counts": {},
            "domain_totals": {},
            "now_ts": now_ts,
            "window_start": now_ts - pd.Timedelta(days=7),
        }

    total_outputs = max(citations["run_id"].nunique(), 1)
    total_prompt_runs = max(citations["prompt_run_id"].nunique(), 1)
    total_unique_prompts = max(citations["query_or_topic"].nunique(), 1)

    prompt_run_counts = (
        citations.groupby("query_or_topic")["prompt_run_id"].nunique().to_dict()
    )
    domain_totals = citations.groupby("domain")["clean_url"].count().to_dict()

    now_ts = pd.Timestamp.utcnow()
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    window_start = now_ts - pd.Timedelta(days=7)

    return {
        "citations": citations,
        "total_outputs": total_outputs,
        "total_prompt_runs": total_prompt_runs,
        "total_unique_prompts": total_unique_prompts,
        "prompt_run_counts": prompt_run_counts,
        "domain_totals": domain_totals,
        "now_ts": now_ts,
        "window_start": window_start,
    }


# ------------------------------------------------------------------
# Per-group generic metrics
# ------------------------------------------------------------------

def compute_group_metrics(
    group: pd.DataFrame,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute the common ~25 metrics for a citation group.

    ``group`` is a slice of the ``context["citations"]`` DataFrame
    (e.g. one domain's rows, or one URL's rows). ``context`` contains
    the shared totals from :func:`compute_aggregation_context`.

    Returns a flat dict of metric values (not yet assigned to column names).
    """
    total_outputs = context["total_outputs"]
    total_prompt_runs = context["total_prompt_runs"]
    total_unique_prompts = context["total_unique_prompts"]
    prompt_run_counts = context["prompt_run_counts"]
    domain_totals = context["domain_totals"]
    now_ts = context["now_ts"]
    window_start = context["window_start"]

    total_citations = int(group.shape[0])

    # Domain share — use the group's primary domain for lookup
    primary_domain = group["domain"].iloc[0] if not group.empty else ""
    domain_total = max(domain_totals.get(primary_domain, total_citations), 1)
    share_pct = (total_citations / domain_total) * 100

    # Outputs
    outputs_citing = group["run_id"].nunique()
    outputs_pct = (outputs_citing / total_outputs) * 100 if total_outputs else 0.0
    avg_cites_per_output = total_citations / max(outputs_citing, 1)

    # Prompt runs
    prompt_runs_citing = group["prompt_run_id"].nunique()
    prompt_runs_pct = (
        (prompt_runs_citing / total_prompt_runs) * 100 if total_prompt_runs else 0.0
    )

    # Unique prompts
    unique_prompts = group["query_or_topic"].nunique()
    unique_prompts_pct = (
        (unique_prompts / total_unique_prompts) * 100 if total_unique_prompts else 0.0
    )

    # Repetition & character
    prompt_repetition_rate = prompt_runs_citing / max(unique_prompts, 1)
    source_character = label_source_character(prompt_repetition_rate)

    # Ranks
    ranks = group["citation_rank"].dropna()
    overall_avg_rank = float(ranks.mean()) if not ranks.empty else None
    top3_pct = float((ranks <= 3).mean() * 100) if not ranks.empty else 0.0

    repeated_mask = group["query_or_topic"].map(
        lambda q: prompt_run_counts.get(q, 0) > 1
    )
    repeated_ranks = group.loc[repeated_mask, "citation_rank"].dropna()
    avg_rank_repeated = float(repeated_ranks.mean()) if not repeated_ranks.empty else None

    unique_mask = group["query_or_topic"].map(
        lambda q: prompt_run_counts.get(q, 0) <= 1
    )
    unique_ranks = group.loc[unique_mask, "citation_rank"].dropna()
    avg_rank_unique = float(unique_ranks.mean()) if not unique_ranks.empty else None

    rank_quality = compute_rank_quality_score(ranks)

    # Timestamps
    timestamps = group["row_timestamp"].dropna()
    first_seen = timestamps.min() if not timestamps.empty else None
    last_seen = timestamps.max() if not timestamps.empty else None
    days_since_last = (now_ts - last_seen).days if pd.notna(last_seen) else None
    recent_count = int((timestamps >= window_start).sum()) if not timestamps.empty else 0
    recent_velocity = (recent_count / total_citations) * 100 if total_citations else 0.0

    # Composite scores
    predictability = compute_predictability_score(outputs_pct, top3_pct)
    topical_authority = compute_topical_authority_score(unique_prompts_pct, avg_rank_unique)
    overall_impact = compute_overall_impact_score(
        predictability, topical_authority, recent_velocity, share_pct
    )

    # Metadata strings
    run_labels = sorted(
        {str(l).strip() for l in group.get("run_label", pd.Series(dtype=str)).dropna() if str(l).strip()}
    )
    execution_ids = sorted(
        {str(e).strip() for e in group.get("execution_id", pd.Series(dtype=str)).dropna() if str(e).strip()}
    )
    run_dates_set: set[str] = set()
    for ts in group.get("row_timestamp", pd.Series(dtype="datetime64[ns]")).dropna():
        try:
            if getattr(ts, "tzinfo", None) is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            run_dates_set.add(ts.strftime("%Y-%m-%d"))
        except Exception:
            continue
    run_dates = sorted(run_dates_set)

    return {
        "total_citations": total_citations,
        "share_pct": share_pct,
        "outputs_citing": outputs_citing,
        "outputs_pct": outputs_pct,
        "avg_cites_per_output": avg_cites_per_output,
        "prompt_runs_citing": prompt_runs_citing,
        "prompt_runs_pct": prompt_runs_pct,
        "unique_prompts": unique_prompts,
        "unique_prompts_pct": unique_prompts_pct,
        "prompt_repetition_rate": prompt_repetition_rate,
        "source_character": source_character,
        "overall_avg_rank": overall_avg_rank,
        "avg_rank_repeated": avg_rank_repeated,
        "avg_rank_unique": avg_rank_unique,
        "top3_pct": top3_pct,
        "rank_quality": rank_quality,
        "first_seen": first_seen.isoformat() if pd.notna(first_seen) else None,
        "last_seen": last_seen.isoformat() if pd.notna(last_seen) else None,
        "days_since_last": days_since_last,
        "recent_velocity": recent_velocity,
        "predictability": predictability,
        "topical_authority": topical_authority,
        "overall_impact": overall_impact,
        "run_labels": ", ".join(run_labels),
        "execution_ids": ", ".join(execution_ids),
        "run_dates": ", ".join(run_dates),
    }
