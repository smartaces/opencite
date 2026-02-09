"""27-column domain intelligence report builder."""
from __future__ import annotations

import pandas as pd

from .aggregation import compute_aggregation_context, compute_group_metrics


REPORT_COLUMNS = [
    "Domain",
    "Unique Pages Cited",
    "Total Page Citations",
    "Domain Citation Share %",
    "Total Outputs Citing Page",
    "% of Total Outputs Citing Page",
    "Avg Citations per Output",
    "Total Prompt Runs Citing Page",
    "Unique Prompts Citing Page",
    "% of Unique Prompts Citing Page",
    "% of Total Prompt Runs Citing Page",
    "Prompt Repetition Rate",
    "Source Character",
    "Overall Average Rank",
    "Avg Rank on Repeated Prompts",
    "Avg Rank on Unique Prompts",
    "% of Citations in Top 3",
    "Rank Quality Score",
    "First Seen Timestamp",
    "Last Seen Timestamp",
    "Days Since Last Seen",
    "Recent Citation Velocity",
    "Predictability Score",
    "Topical Authority Score",
    "Overall Impact Score",
    "Run Labels",
    "Execution IDs",
    "Run Dates",
]

DEFAULT_DISPLAY_COLUMNS = [
    "Domain",
    "Unique Pages Cited",
    "Total Page Citations",
    "Total Outputs Citing Page",
    "Avg Citations per Output",
    "Total Prompt Runs Citing Page",
    "Unique Prompts Citing Page",
    "% of Unique Prompts Citing Page",
    "% of Total Prompt Runs Citing Page",
    "Avg Rank on Repeated Prompts",
    "Avg Rank on Unique Prompts",
    "% of Citations in Top 3",
    "Rank Quality Score",
    "Topical Authority Score",
    "Overall Impact Score",
]

SORT_OPTIONS = [
    ("Domain (A-Z)", "Domain"),
] + [(col, col) for col in REPORT_COLUMNS if col != "Domain"]


def build_domain_report(
    enriched_df: pd.DataFrame,
    ai_role: str = "AI System",
) -> pd.DataFrame:
    """Build 27-column domain intelligence report.

    Parameters
    ----------
    enriched_df : DataFrame
        Output of ``enrich_master_df()``.
    ai_role : str
        Role label for AI responses.
    """
    ctx = compute_aggregation_context(enriched_df, ai_role=ai_role)
    citations = ctx["citations"]

    if citations.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    records: list[dict] = []
    for domain, group in citations.groupby("domain"):
        m = compute_group_metrics(group, ctx)
        records.append({
            "Domain": domain,
            "Unique Pages Cited": group["clean_url"].nunique(),
            "Total Page Citations": m["total_citations"],
            "Domain Citation Share %": m["share_pct"],
            "Total Outputs Citing Page": m["outputs_citing"],
            "% of Total Outputs Citing Page": m["outputs_pct"],
            "Avg Citations per Output": m["avg_cites_per_output"],
            "Total Prompt Runs Citing Page": m["prompt_runs_citing"],
            "Unique Prompts Citing Page": m["unique_prompts"],
            "% of Unique Prompts Citing Page": m["unique_prompts_pct"],
            "% of Total Prompt Runs Citing Page": m["prompt_runs_pct"],
            "Prompt Repetition Rate": m["prompt_repetition_rate"],
            "Source Character": m["source_character"],
            "Overall Average Rank": m["overall_avg_rank"],
            "Avg Rank on Repeated Prompts": m["avg_rank_repeated"],
            "Avg Rank on Unique Prompts": m["avg_rank_unique"],
            "% of Citations in Top 3": m["top3_pct"],
            "Rank Quality Score": m["rank_quality"],
            "First Seen Timestamp": m["first_seen"],
            "Last Seen Timestamp": m["last_seen"],
            "Days Since Last Seen": m["days_since_last"],
            "Recent Citation Velocity": m["recent_velocity"],
            "Predictability Score": m["predictability"],
            "Topical Authority Score": m["topical_authority"],
            "Overall Impact Score": m["overall_impact"],
            "Run Labels": m["run_labels"],
            "Execution IDs": m["execution_ids"],
            "Run Dates": m["run_dates"],
        })

    return pd.DataFrame(records, columns=REPORT_COLUMNS)
