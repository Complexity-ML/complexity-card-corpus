from __future__ import annotations


def meeting_summary_cards(
    contrast_ratio: int,
    *,
    default_decision: str,
    default_open_point: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return compatible decision and open-point variants for meeting summaries."""

    decisions = (
        default_decision,
        f"use the revised settings-page wording once it meets the {contrast_ratio}:1 contrast target",
        f"move forward with the settings-page copy revision subject to a {contrast_ratio}:1 contrast result",
        f"accept the settings-page wording change after confirming its {contrast_ratio}:1 contrast ratio",
    )
    open_points = (
        default_open_point,
        f"when the copy cleared at {contrast_ratio}:1 will be released and in which rollout sequence",
        f"the release timing and deployment order after the {contrast_ratio}:1 accessibility check",
        f"which rollout stage will receive the {contrast_ratio}:1-compliant copy first and on what date",
    )
    return decisions, open_points
