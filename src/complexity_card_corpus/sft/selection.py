from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any


_STRUCTURE_SLOT = re.compile(
    r"\b(?:[A-Z0-9]{5,}|[A-Z]+\d+[A-Z0-9]*|(?i:day)\s+\d+|\d{1,2}:\d{2}|\$\d+(?:\.\d+)?|\d+(?:\.\d+)?)\b",
)


def _normalized_structure(text: str) -> str:
    """Normalize volatile slots while retaining syntax and answer shape."""

    normalized = _STRUCTURE_SLOT.sub("<slot>", text)
    normalized = re.sub(r"[\"'“”][^\"'“”]{1,80}[\"'“”]", "<quoted>", normalized)
    normalized = re.sub(r"(?m)^\s*\d+[.)]\s*", "<item> ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _deduplicate_structural_rows(
    rows: list[dict[str, Any]],
    *,
    target_key: str = "_projected_target",
    max_per_structure: int = 1,
    per_task_limits: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bound deterministic examples per task and normalized answer shape."""

    if max_per_structure < 1:
        raise ValueError("max_per_structure must be positive")

    kept: list[dict[str, Any]] = []
    per_task_limits = per_task_limits or {}
    if any(limit < 1 for limit in per_task_limits.values()):
        raise ValueError("per-task structure limits must be positive")
    counts: Counter[tuple[str, str]] = Counter()
    retained: Counter[tuple[str, str]] = Counter()
    for row in sorted(rows, key=lambda item: item["example_id"]):
        signature = _normalized_structure(row[target_key])
        key = (row["task"], signature)
        limit = per_task_limits.get(row["task"], max_per_structure)
        counts[key] += 1
        if retained[key] >= limit:
            continue
        retained[key] += 1
        copy = dict(row)
        copy["_structure_signature"] = signature
        kept.append(copy)
    return kept, {
        "input_examples": len(rows),
        "kept_examples": len(kept),
        "dropped_structural_duplicates": len(rows) - len(kept),
        "distinct_task_structures": len(counts),
        "maximum_retained_per_structure": max(
            (max_per_structure, *per_task_limits.values())
        ),
        "default_maximum_retained_per_structure": max_per_structure,
        "per_task_structure_limits": dict(sorted(per_task_limits.items())),
        "maximum_examples_per_structure_before_dedup": max(counts.values(), default=0),
    }


def _deduplicate_exact_responses(
    rows: list[dict[str, Any]],
    *,
    target_key: str = "_projected_target",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove exact assistant-response duplicates deterministically."""

    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: item["example_id"]):
        target = re.sub(r"\s+", " ", row[target_key]).strip()
        if target in seen:
            continue
        seen.add(target)
        kept.append(row)
    return kept, {
        "input_examples": len(rows),
        "kept_examples": len(kept),
        "dropped_exact_response_duplicates": len(rows) - len(kept),
        "exact_response_uniqueness_ratio": 1.0,
    }


def _balance_task_families(
    rows: list[dict[str, Any]],
    *,
    max_examples_per_family: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cap dominant families without duplicating minority examples."""

    if max_examples_per_family < 1:
        raise ValueError("max_examples_per_family must be positive")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["task"]].append(row)
    kept: list[dict[str, Any]] = []
    before = {task: len(items) for task, items in sorted(buckets.items())}
    for task, items in sorted(buckets.items()):
        ranked = sorted(
            items,
            key=lambda item: hashlib.sha256(
                f"family-balance:{task}:{item['example_id']}".encode()
            ).digest(),
        )
        kept.extend(ranked[:max_examples_per_family])
    kept.sort(key=lambda item: item["example_id"])
    after = dict(sorted(Counter(row["task"] for row in kept).items()))
    total = len(kept)
    shares = {
        task: round(count / total, 6) if total else 0.0 for task, count in after.items()
    }
    return kept, {
        "input_examples": len(rows),
        "kept_examples": total,
        "dropped_for_family_balance": len(rows) - total,
        "maximum_examples_per_family": max_examples_per_family,
        "before": before,
        "after": after,
        "shares": shares,
    }
