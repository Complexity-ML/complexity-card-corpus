from __future__ import annotations

import math
from typing import Any

from ..release_targets import TARGET_POST_TRAINING_ROWS

PLANNED_DISTINCT_SURFACES_PER_SOURCE_CARD = 8


MIN_MASKED_SKELETON_UNIQUENESS = 0.08


MAX_MASKED_SKELETON_SHARE = 0.01


MAX_MASKED_EIGHT_TOKEN_COVERAGE = 0.04


MAX_FAMILY_MASKED_TEMPLATE_SHARE = 0.05


def required_distinct_surfaces_per_source_card(
    source_cards: int,
    target_rows: int = TARGET_POST_TRAINING_ROWS,
) -> int:
    """Return the minimum pre-deduplication surface budget for a row target."""

    if source_cards < 1:
        raise ValueError("source_cards must be positive")
    if target_rows < 1:
        raise ValueError("target_rows must be positive")
    return math.ceil(target_rows / source_cards)


def post_training_capacity_report(
    *,
    source_cards: int,
    configured_variants_per_source_card: int,
    audit: dict[str, Any],
    target_rows: int = TARGET_POST_TRAINING_ROWS,
) -> dict[str, Any]:
    """Describe scale capacity without claiming that unbuilt rows exist.

    Exact uniqueness alone is not a scale signal. The report also evaluates
    masked response structures, repeated eight-token spans, and family-local
    template concentration after semantic variables have been removed.
    """

    if configured_variants_per_source_card < 1:
        raise ValueError("configured_variants_per_source_card must be positive")
    required_surfaces = required_distinct_surfaces_per_source_card(
        source_cards,
        target_rows,
    )
    masked = audit["masked_response_diversity"]
    repetition = audit["response_repetition_gate"]
    family_metrics = audit["family_metrics"]
    hotspots = sorted(
        family
        for family, metrics in family_metrics.items()
        if metrics["maximum_masked_template_share"]
        >= MAX_FAMILY_MASKED_TEMPLATE_SHARE
    )
    quality_gates = {
        "exact_final_response_uniqueness": (
            audit["exact_final_response_uniqueness_ratio"] == 1.0
        ),
        "masked_skeleton_uniqueness": (
            masked["exact_skeleton_uniqueness_ratio"]
            >= MIN_MASKED_SKELETON_UNIQUENESS
        ),
        "masked_skeleton_concentration": (
            masked["maximum_skeleton_share"] < MAX_MASKED_SKELETON_SHARE
        ),
        "masked_eight_token_coverage": (
            repetition["maximum_masked_eight_token_message_coverage"]
            < MAX_MASKED_EIGHT_TOKEN_COVERAGE
        ),
        "family_template_concentration": not hotspots,
    }
    generated_rows = int(audit["rows"])
    configured_ceiling = source_cards * configured_variants_per_source_card
    planned_ceiling = source_cards * PLANNED_DISTINCT_SURFACES_PER_SOURCE_CARD
    return {
        "target_rows": target_rows,
        "generated_rows": generated_rows,
        "source_cards": source_cards,
        "retained_source_cards": int(audit["source_scenarios"]),
        "required_distinct_surfaces_per_source_card": required_surfaces,
        "configured_variants_per_source_card": configured_variants_per_source_card,
        "configured_pre_deduplication_ceiling": configured_ceiling,
        "configured_variant_shortfall": max(
            0,
            required_surfaces - configured_variants_per_source_card,
        ),
        "planned_distinct_surfaces_per_source_card": (
            PLANNED_DISTINCT_SURFACES_PER_SOURCE_CARD
        ),
        "planned_pre_deduplication_ceiling": planned_ceiling,
        "planned_capacity_exceeds_target": planned_ceiling >= target_rows,
        "current_configuration_can_reach_target": configured_ceiling >= target_rows,
        "static_surface_hotspots": hotspots,
        "quality_thresholds": {
            "minimum_masked_skeleton_uniqueness": MIN_MASKED_SKELETON_UNIQUENESS,
            "maximum_masked_skeleton_share": MAX_MASKED_SKELETON_SHARE,
            "maximum_masked_eight_token_coverage": (
                MAX_MASKED_EIGHT_TOKEN_COVERAGE
            ),
            "maximum_family_masked_template_share": (
                MAX_FAMILY_MASKED_TEMPLATE_SHARE
            ),
        },
        "quality_gates": quality_gates,
        "surface_quality_ready": all(quality_gates.values()),
        "target_generated": generated_rows >= target_rows,
        "release_target_ready": (
            generated_rows >= target_rows and all(quality_gates.values())
        ),
        "claim_scope": (
            "capacity contract only; planned or theoretical rows are not "
            "reported as generated examples"
        ),
    }
