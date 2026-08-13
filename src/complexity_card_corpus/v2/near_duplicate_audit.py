from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any, Iterable


_WORD = re.compile(r"[a-z0-9']+", re.I)


def _shingles(text: str) -> frozenset[tuple[str, ...]]:
    normalized = re.sub(r"\d+(?:[.,:/]\d+)*", "<n>", text.casefold())
    words = _WORD.findall(normalized)
    if len(words) < 3:
        return frozenset({tuple(words)}) if words else frozenset()
    return frozenset(zip(words, words[1:], words[2:]))


def _simhash(shingles: frozenset[tuple[str, ...]]) -> int:
    scores = [0] * 32
    for shingle in shingles:
        value = int.from_bytes(
            hashlib.blake2s(" ".join(shingle).encode(), digest_size=4).digest(),
            "big",
        )
        for bit in range(32):
            scores[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, score in enumerate(scores):
        if score >= 0:
            result |= 1 << bit
    return result


def _jaccard(
    left: frozenset[tuple[str, ...]],
    right: frozenset[tuple[str, ...]],
) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def _field_audit(
    rows: list[dict[str, Any]],
    field: str,
    *,
    threshold: float,
) -> dict[str, Any]:
    signatures: dict[frozenset[tuple[str, ...]], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        signatures[_shingles(str(row.get(field, "")))].append(index)
    collided: set[int] = set()
    examples: list[dict[str, Any]] = []
    unique_signatures = list(signatures)
    for signature, indexes in signatures.items():
        if len(indexes) > 1:
            collided.update(indexes)
            if len(examples) < 20:
                examples.append(
                    {
                        "left": str(rows[indexes[0]].get("example_id", indexes[0])),
                        "right": str(rows[indexes[1]].get("example_id", indexes[1])),
                        "similarity": 1.0,
                        "kind": "normalized_structure",
                    }
                )
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    hashes = [_simhash(signature) for signature in unique_signatures]
    compared: set[tuple[int, int]] = set()
    for index, value in enumerate(hashes):
        candidates: set[int] = set()
        for band in range(4):
            key = (band, (value >> (band * 8)) & 0xFF)
            candidates.update(buckets[key])
            buckets[key].append(index)
        for other in candidates:
            pair = (other, index)
            if pair in compared:
                continue
            compared.add(pair)
            similarity = _jaccard(
                unique_signatures[other], unique_signatures[index]
            )
            if similarity < threshold:
                continue
            left_indexes = signatures[unique_signatures[other]]
            right_indexes = signatures[unique_signatures[index]]
            collided.update(left_indexes)
            collided.update(right_indexes)
            if len(examples) < 20:
                examples.append(
                    {
                        "left": str(rows[left_indexes[0]].get("example_id", left_indexes[0])),
                        "right": str(rows[right_indexes[0]].get("example_id", right_indexes[0])),
                        "similarity": round(similarity, 6),
                        "kind": "near_duplicate",
                    }
                )
    return {
        "sampled_rows": len(rows),
        "unique_normalized_structures": len(signatures),
        "collision_rows": len(collided),
        "collision_share": round(len(collided) / max(1, len(rows)), 6),
        "examples": examples,
    }


def audit_v2_near_duplicates(
    rows: Iterable[dict[str, Any]],
    *,
    maximum_examples_per_task: int = 2_000,
    similarity_threshold: float = 0.82,
    maximum_collision_share: float = 0.10,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split", "train") == "train":
            grouped[str(row.get("task", "unknown"))].append(row)
    tasks = {}
    failing_tasks = []
    for task, task_rows in sorted(grouped.items()):
        sampled = sorted(
            task_rows,
            key=lambda row: hashlib.sha256(
                str(row.get("example_id", "")).encode()
            ).digest(),
        )[:maximum_examples_per_task]
        prompt = _field_audit(sampled, "prompt", threshold=similarity_threshold)
        final = _field_audit(
            sampled, "final_response", threshold=similarity_threshold
        )
        failures = []
        if prompt["collision_share"] > maximum_collision_share:
            failures.append("prompt_near_duplicates")
        if final["collision_share"] > maximum_collision_share:
            failures.append("final_near_duplicates")
        if failures:
            failing_tasks.append(task)
        tasks[task] = {
            "rows": len(task_rows),
            "failures": failures,
            "prompt": prompt,
            "final": final,
        }
    return {
        "format": "complexity-card-corpus-v2-near-duplicate-audit-v1",
        "passed": not failing_tasks,
        "failing_tasks": failing_tasks,
        "tasks": tasks,
        "thresholds": {
            "similarity": similarity_threshold,
            "maximum_collision_share": maximum_collision_share,
            "maximum_examples_per_task": maximum_examples_per_task,
        },
    }


__all__ = ("audit_v2_near_duplicates",)
