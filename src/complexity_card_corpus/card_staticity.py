from __future__ import annotations

import math
from collections import Counter
from typing import Mapping, Sequence


def audit_card_staticity(
    hands: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    """Measure repeated card choices without treating row IDs as diversity.

    The audit works on semantic card values only. It deliberately ignores
    scenario IDs, hashes, numbers, and rendered prose, so a new identifier
    cannot make an otherwise static hand look diverse.
    """

    if not hands:
        raise ValueError("card staticity audit requires at least one hand")
    axes = tuple(hands[0])
    if not axes or any(tuple(hand) != axes for hand in hands):
        raise ValueError("all card hands must expose the same ordered axes")
    signatures = [tuple(hand[axis] for axis in axes) for hand in hands]
    signature_counts = Counter(signatures)
    axis_metrics: dict[str, dict[str, object]] = {}
    for axis in axes:
        counts = Counter(hand[axis] for hand in hands)
        probabilities = [count / len(hands) for count in counts.values()]
        entropy = -sum(value * math.log(value) for value in probabilities)
        normalized_entropy = entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
        dominant_value, dominant_count = counts.most_common(1)[0]
        axis_metrics[axis] = {
            "unique_values": len(counts),
            "dominant_value": dominant_value,
            "maximum_share": round(dominant_count / len(hands), 6),
            "normalized_entropy": round(normalized_entropy, 6),
        }
    return {
        "hands": len(hands),
        "unique_hands": len(signature_counts),
        "exact_hand_uniqueness_ratio": round(
            len(signature_counts) / len(hands),
            6,
        ),
        "maximum_hand_share": round(
            max(signature_counts.values()) / len(hands),
            6,
        ),
        "static_axes": sorted(
            axis for axis, metrics in axis_metrics.items() if metrics["unique_values"] == 1
        ),
        "axes": axis_metrics,
    }
