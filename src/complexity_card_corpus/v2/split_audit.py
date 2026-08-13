from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any, Iterable


_SPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*")
_WORD = re.compile(r"[a-z0-9']+", re.I)


def _exact_signature(row: dict[str, Any]) -> str:
    material = "\n".join(
        _SPACE.sub(" ", str(row.get(field, "")).strip().casefold())
        for field in ("prompt", "final_response")
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _structural_signature(row: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    fields = []
    for field in ("prompt", "final_response"):
        normalized = _NUMBER.sub("<n>", str(row.get(field, "")).casefold())
        words = _WORD.findall(normalized)
        fields.append(tuple(words))
    return tuple(fields)


def _cross_split_collisions(
    rows: list[dict[str, Any]],
    signature,
) -> tuple[int, list[dict[str, str]]]:
    seen: dict[Any, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        seen[signature(row)][str(row.get("split", "train"))].append(
            str(row.get("example_id", "unknown"))
        )
    collision_rows = 0
    examples = []
    for split_rows in seen.values():
        if len(split_rows) < 2:
            continue
        collision_rows += sum(len(ids) for ids in split_rows.values())
        if len(examples) < 20:
            left, right = sorted(split_rows)[:2]
            examples.append(
                {
                    "left_split": left,
                    "left_id": split_rows[left][0],
                    "right_split": right,
                    "right_id": split_rows[right][0],
                }
            )
    return collision_rows, examples


def audit_v2_splits(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    split_counts: dict[str, int] = defaultdict(int)
    for row in materialized:
        split_counts[str(row.get("split", "train"))] += 1
    exact_count, exact_examples = _cross_split_collisions(
        materialized, _exact_signature
    )
    structural_count, structural_examples = _cross_split_collisions(
        materialized, _structural_signature
    )
    violations = []
    if len(split_counts) < 2:
        violations.append("at least two populated splits are required")
    if exact_count:
        violations.append("exact conversations leak across splits")
    if structural_count:
        violations.append("normalized template structures leak across splits")
    return {
        "format": "complexity-card-corpus-v2-split-audit-v1",
        "passed": not violations,
        "violations": violations,
        "rows": len(materialized),
        "split_counts": dict(sorted(split_counts.items())),
        "exact_cross_split_collision_rows": exact_count,
        "exact_examples": exact_examples,
        "structural_cross_split_collision_rows": structural_count,
        "structural_examples": structural_examples,
    }


__all__ = ("audit_v2_splits",)
