from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable


_BANDS = (
    ("direct_1_25", 1, 25),
    ("concise_26_80", 26, 80),
    ("detailed_81_200", 81, 200),
    ("extended_201_512", 201, 512),
)


def _band(words: int) -> str:
    for name, low, high in _BANDS:
        if low <= words <= high:
            return name
    return "empty" if words == 0 else "over_512"


def audit_v2_lengths(
    rows: Iterable[dict[str, Any]],
    *,
    require_global_bands: bool = True,
) -> dict[str, Any]:
    final_bands = Counter()
    task_bands: dict[str, Counter[str]] = defaultdict(Counter)
    thinking_lengths: list[int] = []
    thinking_outside_contract = 0
    total = 0
    for row in rows:
        if row.get("split", "train") != "train":
            continue
        total += 1
        task = str(row.get("task", "unknown"))
        final_words = len(str(row.get("final_response", "")).split())
        band = _band(final_words)
        final_bands[band] += 1
        task_bands[task][band] += 1
        thinking = str(row.get("reasoning_trace", "")).strip()
        if thinking:
            words = len(thinking.split())
            thinking_lengths.append(words)
            thinking_outside_contract += not 8 <= words <= 120
    band_shares = {
        name: final_bands[name] / max(1, total)
        for name, _low, _high in _BANDS
    }
    violations = []
    if final_bands["empty"]:
        violations.append("empty final responses")
    if final_bands["over_512"]:
        violations.append("final responses exceed 512 words")
    if thinking_outside_contract:
        violations.append("thinking traces fall outside 8-120 words")
    if require_global_bands and any(share < 0.05 for share in band_shares.values()):
        violations.append("one or more final response length bands are below 5%")
    return {
        "format": "complexity-card-corpus-v2-length-audit-v1",
        "passed": not violations,
        "violations": violations,
        "rows": total,
        "final_bands": dict(sorted(final_bands.items())),
        "final_band_shares": {
            name: round(share, 6) for name, share in band_shares.items()
        },
        "thinking_examples": len(thinking_lengths),
        "thinking_outside_contract": thinking_outside_contract,
        "thinking_min_words": min(thinking_lengths) if thinking_lengths else 0,
        "thinking_max_words": max(thinking_lengths) if thinking_lengths else 0,
        "tasks": {
            task: dict(sorted(counts.items()))
            for task, counts in sorted(task_bands.items())
        },
        "thresholds": {
            "minimum_global_share_per_final_band": 0.05,
            "maximum_final_words": 512,
            "thinking_word_range": [8, 120],
        },
    }


__all__ = ("audit_v2_lengths",)
