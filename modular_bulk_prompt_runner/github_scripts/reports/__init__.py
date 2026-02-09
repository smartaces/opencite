# reports/__init__.py
"""
Report generators for the modular bulk prompt runner.

Provides domain-level, page-level, and prompt-level citation analysis reports
with shared scoring, aggregation, and two-tier filtering.
"""

from .scoring import (
    compute_overall_impact_score,
    compute_predictability_score,
    compute_rank_quality_score,
    compute_topical_authority_score,
    label_source_character,
)
from .url_utils import (
    clean_url,
    derive_prompt_run_id,
    derive_run_id,
    derive_title_from_url,
    extract_domain,
)
from .formatting import (
    compute_summary,
    format_domain_link,
    format_numeric,
    format_timestamp_short,
    format_url_link,
    truncate_prompt_text,
)
from .dataset import (
    enrich_master_df,
    export_dataframe,
    normalize_role_labels,
    refresh_master_df,
)
from .filters import (
    apply_data_filters,
    apply_view_filters,
    create_clear_filters_button,
    create_data_filter_panel,
    create_view_filter_panel,
    reset_filter_widgets,
    update_domain_options,
)
from .aggregation import (
    compute_aggregation_context,
    compute_group_metrics,
)
from .domain_report import build_domain_report
from .page_report import build_page_report
from .prompt_report import build_prompt_insights

__all__ = [
    # scoring
    "compute_rank_quality_score",
    "compute_topical_authority_score",
    "label_source_character",
    "compute_predictability_score",
    "compute_overall_impact_score",
    # url_utils
    "clean_url",
    "extract_domain",
    "derive_title_from_url",
    "derive_run_id",
    "derive_prompt_run_id",
    # formatting
    "format_timestamp_short",
    "format_numeric",
    "format_domain_link",
    "format_url_link",
    "truncate_prompt_text",
    "compute_summary",
    # dataset
    "refresh_master_df",
    "enrich_master_df",
    "normalize_role_labels",
    "export_dataframe",
    # filters
    "create_data_filter_panel",
    "apply_data_filters",
    "create_view_filter_panel",
    "apply_view_filters",
    "update_domain_options",
    "reset_filter_widgets",
    "create_clear_filters_button",
    # aggregation
    "compute_aggregation_context",
    "compute_group_metrics",
    # reports
    "build_domain_report",
    "build_page_report",
    "build_prompt_insights",
]
