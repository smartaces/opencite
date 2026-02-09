"""29-column prompt-level insights report builder with nested drill-down."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .scoring import compute_topical_authority_score, compute_predictability_score
from .url_utils import derive_title_from_url


REPORT_COLUMNS = [
    "Prompt Text",
    "Scenario",
    "Persona",
    "Model",
    "Total Prompt Runs",
    "Unique Prompt Runs",
    "% of Total Prompt Runs",
    "Total Outputs",
    "% of Outputs with Citations",
    "Prompt Last Seen",
    "Domain",
    "Page Title",
    "Unique Pages Cited",
    "Full URL",
    "Total Domain Citations",
    "Total Page Citations",
    "Avg Domain Rank",
    "Avg Page Rank",
    "% of Prompt Runs Citing Domain",
    "% of Prompt Runs Citing Page",
    "% of Outputs Citing Domain",
    "% of Outputs Citing Page",
    "Avg Citations per Output (Domain)",
    "Avg Citations per Output (Page)",
    "Recent Domain Velocity",
    "First Seen Timestamp",
    "Predictability Score",
    "Page Last Seen Timestamp",
    "Topical Authority Score",
    "Days Since Last Seen",
    "Run Labels",
    "Execution IDs",
    "Run Dates",
]

DEFAULT_DISPLAY_COLUMNS = [
    "Prompt Text",
    "Domain",
    "Page Title",
    "Model",
    "Total Domain Citations",
    "Total Page Citations",
    "Avg Domain Rank",
    "Avg Page Rank",
    "Avg Citations per Output (Domain)",
    "Avg Citations per Output (Page)",
]

SORT_OPTIONS = [
    ("Prompt Text (A-Z)", "Prompt Text"),
    ("Domain (A-Z)", "Domain"),
    ("Average Domain Rank", "Avg Domain Rank"),
    ("Average Page Rank", "Avg Page Rank"),
    ("Domain Citations", "Total Domain Citations"),
    ("Page Citations", "Total Page Citations"),
    ("Predictability Score", "Predictability Score"),
    ("Topical Authority Score", "Topical Authority Score"),
    ("Recent Domain Velocity", "Recent Domain Velocity"),
    ("% Prompt Runs (Domain)", "% of Prompt Runs Citing Domain"),
    ("% Prompt Runs (Page)", "% of Prompt Runs Citing Page"),
]


def _first_non_null(series: pd.Series) -> object:
    if series is None or series.empty:
        return None
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else None


def _collect_metadata(rows: pd.DataFrame) -> Dict[str, Any]:
    """Extract run labels, execution IDs, and run dates from a group."""
    run_labels = sorted(
        {str(l).strip() for l in rows.get("run_label", pd.Series(dtype=str)).dropna() if str(l).strip()}
    )
    execution_ids = sorted(
        {str(e).strip() for e in rows.get("execution_id", pd.Series(dtype=str)).dropna() if str(e).strip()}
    )
    run_dates_set: set[str] = set()
    for ts in rows.get("row_timestamp", pd.Series(dtype="datetime64[ns]")).dropna():
        try:
            if getattr(ts, "tzinfo", None) is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            run_dates_set.add(ts.strftime("%Y-%m-%d"))
        except Exception:
            continue
    return {
        "run_labels": ", ".join(run_labels),
        "execution_ids": ", ".join(execution_ids),
        "run_dates": ", ".join(sorted(run_dates_set)),
    }


def build_prompt_insights(
    enriched_df: pd.DataFrame,
    ai_role: str = "AI System",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build prompt-level insights with nested prompt -> domain -> page grouping.

    Parameters
    ----------
    enriched_df : DataFrame
        Output of ``enrich_master_df()``.
    ai_role : str
        Role label for AI responses.

    Returns
    -------
    tuple[DataFrame, DataFrame]
        ``(domain_df, page_df)`` — domain-level and page-level records.
    """
    empty = pd.DataFrame(columns=REPORT_COLUMNS)

    if enriched_df.empty:
        return empty, empty

    if "role" in enriched_df.columns:
        working = enriched_df[enriched_df["role"] == ai_role].copy()
    else:
        working = enriched_df.copy()

    if working.empty:
        return empty, empty

    # Prepare page titles on citation rows
    citations = working.dropna(subset=["citation_url"]).copy()
    citations = citations[citations["clean_url"] != ""]
    citations = citations[citations["domain"] != ""]
    if citations.empty:
        return empty, empty

    citations["page_title"] = citations.apply(
        lambda row: row["citation_title"]
        if isinstance(row.get("citation_title"), str) and str(row["citation_title"]).strip()
        else derive_title_from_url(row["clean_url"]),
        axis=1,
    )

    now_ts = pd.Timestamp.utcnow()
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    window_start = now_ts - pd.Timedelta(days=7)
    total_prompt_runs_all = max(working["prompt_run_id"].nunique(), 1)

    domain_records: list[dict] = []
    page_records: list[dict] = []

    for prompt, prompt_rows in working.groupby("query_or_topic"):
        prompt_total_runs = prompt_rows["prompt_run_id"].nunique()
        prompt_total_outputs = prompt_rows["run_id"].nunique()
        prompt_citations = citations[citations["query_or_topic"] == prompt]
        if prompt_citations.empty:
            continue

        prompt_citing_runs = prompt_citations["prompt_run_id"].nunique()
        outputs_with_cites = prompt_citations["run_id"].nunique()
        prompt_last_seen = prompt_citations["row_timestamp"].max()

        scenario = _first_non_null(prompt_rows["scenario"]) if "scenario" in prompt_rows.columns else None
        persona = _first_non_null(prompt_rows["persona_profile"]) if "persona_profile" in prompt_rows.columns else None
        model = _first_non_null(prompt_rows["model"]) if "model" in prompt_rows.columns else None
        meta = _collect_metadata(prompt_rows)

        prompt_base = {
            "Prompt Text": prompt,
            "Scenario": scenario,
            "Persona": persona,
            "Model": model,
            "Total Prompt Runs": prompt_total_runs,
            "Unique Prompt Runs": prompt_citing_runs,
            "% of Total Prompt Runs": (prompt_total_runs / total_prompt_runs_all) * 100,
            "Total Outputs": prompt_total_outputs,
            "% of Outputs with Citations": (outputs_with_cites / max(prompt_total_outputs, 1)) * 100,
            "Prompt Last Seen": prompt_last_seen.isoformat() if pd.notna(prompt_last_seen) else None,
            "Run Labels": meta["run_labels"],
            "Execution IDs": meta["execution_ids"],
            "Run Dates": meta["run_dates"],
        }

        # --- Domain level ---
        for domain, domain_rows in prompt_citations.groupby("domain"):
            domain_unique_pages = domain_rows["clean_url"].nunique()
            domain_total_cites = int(domain_rows.shape[0])
            domain_ranks = domain_rows["citation_rank"].dropna()
            domain_avg_rank = float(domain_ranks.mean()) if not domain_ranks.empty else None
            domain_prompt_runs = domain_rows["prompt_run_id"].nunique()
            domain_outputs = domain_rows["run_id"].nunique()

            pct_prompt_runs_domain = (domain_prompt_runs / max(prompt_total_runs, 1)) * 100
            pct_outputs_domain = (domain_outputs / max(prompt_total_outputs, 1)) * 100
            avg_cites_domain = domain_total_cites / max(domain_outputs, 1)

            domain_top3_pct = (
                float((domain_ranks <= 3).mean() * 100)
                if not domain_ranks.empty
                else 0.0
            )

            timestamps = domain_rows["row_timestamp"].dropna()
            recent_count = int((timestamps >= window_start).sum()) if not timestamps.empty else 0
            recent_velocity = (recent_count / domain_total_cites) * 100 if domain_total_cites else 0.0
            predictability = compute_predictability_score(pct_outputs_domain, domain_top3_pct)
            topical_authority = compute_topical_authority_score(pct_prompt_runs_domain, domain_avg_rank)
            domain_first_seen = timestamps.min() if not timestamps.empty else None
            domain_last_seen = timestamps.max() if not timestamps.empty else None
            domain_days_since = (now_ts - domain_last_seen).days if pd.notna(domain_last_seen) else None

            domain_base = {
                **prompt_base,
                "Domain": domain,
                "Unique Pages Cited": domain_unique_pages,
                "Total Domain Citations": domain_total_cites,
                "Avg Domain Rank": domain_avg_rank,
                "% of Prompt Runs Citing Domain": pct_prompt_runs_domain,
                "% of Outputs Citing Domain": pct_outputs_domain,
                "Avg Citations per Output (Domain)": avg_cites_domain,
                "Recent Domain Velocity": recent_velocity,
                "Predictability Score": predictability,
                "Topical Authority Score": topical_authority,
                "First Seen Timestamp": domain_first_seen.isoformat() if pd.notna(domain_first_seen) else None,
                "Page Last Seen Timestamp": domain_last_seen.isoformat() if pd.notna(domain_last_seen) else None,
                "Days Since Last Seen": domain_days_since,
            }

            # Domain-only record (no page details)
            domain_records.append({
                **domain_base,
                "Page Title": None,
                "Full URL": None,
                "Total Page Citations": None,
                "Avg Page Rank": None,
                "% of Prompt Runs Citing Page": None,
                "% of Outputs Citing Page": None,
                "Avg Citations per Output (Page)": None,
            })

            # --- Page level within domain ---
            for page_url, page_rows in domain_rows.groupby("clean_url"):
                page_total_cites = int(page_rows.shape[0])
                page_ranks = page_rows["citation_rank"].dropna()
                page_avg_rank = float(page_ranks.mean()) if not page_ranks.empty else None
                page_prompt_runs = page_rows["prompt_run_id"].nunique()
                page_outputs = page_rows["run_id"].nunique()
                pct_prompt_runs_page = (page_prompt_runs / max(prompt_total_runs, 1)) * 100
                pct_outputs_page = (page_outputs / max(prompt_total_outputs, 1)) * 100
                avg_cites_page = page_total_cites / max(page_outputs, 1)
                page_first_seen = page_rows["row_timestamp"].min()
                page_last_seen = page_rows["row_timestamp"].max()
                days_since_last = (now_ts - page_last_seen).days if pd.notna(page_last_seen) else None
                page_title = page_rows["page_title"].dropna().iloc[0] if page_rows["page_title"].notna().any() else derive_title_from_url(page_url)

                page_records.append({
                    **domain_base,
                    "Page Title": page_title,
                    "Full URL": page_url,
                    "Total Page Citations": page_total_cites,
                    "Avg Page Rank": page_avg_rank,
                    "% of Prompt Runs Citing Page": pct_prompt_runs_page,
                    "% of Outputs Citing Page": pct_outputs_page,
                    "Avg Citations per Output (Page)": avg_cites_page,
                    "First Seen Timestamp": page_first_seen.isoformat() if pd.notna(page_first_seen) else None,
                    "Page Last Seen Timestamp": page_last_seen.isoformat() if pd.notna(page_last_seen) else None,
                    "Days Since Last Seen": days_since_last,
                })

    if not page_records and not domain_records:
        return empty, empty

    domain_df = pd.DataFrame(domain_records, columns=REPORT_COLUMNS) if domain_records else empty.copy()
    page_df = pd.DataFrame(page_records, columns=REPORT_COLUMNS) if page_records else empty.copy()
    return domain_df, page_df
