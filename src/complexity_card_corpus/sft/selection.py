from __future__ import annotations

import hashlib
import math
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
    """Bound examples per task, semantic domain, and normalized answer shape.

    The same response form can legitimately teach different subject matter in
    two domains.  Collapsing those rows before domain balancing silently
    removes authored coverage, so structural duplicates are bounded only
    inside their own semantic domain.
    """

    if max_per_structure < 1:
        raise ValueError("max_per_structure must be positive")

    kept: list[dict[str, Any]] = []
    per_task_limits = per_task_limits or {}
    if any(limit < 1 for limit in per_task_limits.values()):
        raise ValueError("per-task structure limits must be positive")
    counts: Counter[tuple[str, str, str]] = Counter()
    retained: Counter[tuple[str, str, str]] = Counter()
    for row in sorted(rows, key=lambda item: item["example_id"]):
        signature = _normalized_structure(row[target_key])
        key = (row["task"], row.get("domain", "__unspecified__"), signature)
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
        "structural_deduplication_unit": "task+domain+response_structure",
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


def _deduplicate_exact_prompts(
    rows: list[dict[str, Any]],
    *,
    prompt_key: str = "_projected_prompt",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep one target for each exact model-facing prompt.

    Several individually valid completions for the same prompt create an
    avoidable one-to-many supervision conflict in a small SFT corpus. Rich
    response variation belongs behind distinct user contexts instead.
    """

    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: item["example_id"]):
        prompt = re.sub(r"\s+", " ", row[prompt_key]).strip()
        if prompt in seen:
            continue
        seen.add(prompt)
        kept.append(row)
    return kept, {
        "input_examples": len(rows),
        "kept_examples": len(kept),
        "dropped_exact_prompt_duplicates": len(rows) - len(kept),
        "exact_prompt_uniqueness_ratio": 1.0,
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


def _balance_task_domains(
    rows: list[dict[str, Any]],
    *,
    maximum_share: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cap semantic-domain skew inside each family without upsampling.

    A five-percent ceiling is mathematically possible only when a family has
    at least twenty realized domains.  Smaller families use their unavoidable
    ``1 / domain_count`` floor and are reported explicitly for later tank
    hydration rather than hidden behind duplicated examples.
    """

    if not 0 < maximum_share <= 1:
        raise ValueError("maximum_share must be in (0, 1]")
    tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tasks[row["task"]].append(row)

    kept: list[dict[str, Any]] = []
    task_audit: dict[str, Any] = {}
    for task, task_rows in sorted(tasks.items()):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in task_rows:
            buckets[row["domain"]].append(row)
        counts = {domain: len(items) for domain, items in buckets.items()}
        effective_share = max(maximum_share, 1 / len(buckets))
        target = len(task_rows)
        while target:
            cap = max(1, math.floor(effective_share * target))
            available = sum(min(count, cap) for count in counts.values())
            if available >= target:
                break
            target = available
        cap = max(1, math.floor(effective_share * target)) if target else 0
        selected: list[dict[str, Any]] = []
        for domain, domain_rows in sorted(buckets.items()):
            ranked = sorted(
                domain_rows,
                key=lambda item: hashlib.sha256(
                    f"domain-balance:{task}:{domain}:{item['example_id']}".encode()
                ).digest(),
            )
            selected.extend(ranked[:cap])
        if len(selected) > target:
            selected = sorted(
                selected,
                key=lambda item: hashlib.sha256(
                    f"domain-target:{task}:{item['example_id']}".encode()
                ).digest(),
            )[:target]
        kept.extend(selected)
        after = Counter(row["domain"] for row in selected)
        task_audit[task] = {
            "input_examples": len(task_rows),
            "kept_examples": len(selected),
            "distinct_domains": len(buckets),
            "requested_maximum_share": maximum_share,
            "effective_maximum_share": round(effective_share, 6),
            "requires_tank_hydration": len(buckets) < math.ceil(1 / maximum_share),
            "maximum_domain_share_before": round(
                max(counts.values(), default=0) / len(task_rows), 6
            ),
            "maximum_domain_share_after": round(
                max(after.values(), default=0) / len(selected) if selected else 0.0,
                6,
            ),
        }

    kept.sort(key=lambda item: item["example_id"])
    return kept, {
        "input_examples": len(rows),
        "kept_examples": len(kept),
        "dropped_overrepresented_domains": len(rows) - len(kept),
        "requested_maximum_share": maximum_share,
        "tasks_requiring_tank_hydration": sorted(
            task for task, item in task_audit.items() if item["requires_tank_hydration"]
        ),
        "tasks": task_audit,
    }


def _balance_response_card_hands(
    rows: list[dict[str, Any]],
    *,
    maximum_share: float = 0.12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prevent one invisible response-card hand from dominating a family.

    The target size is solved independently for each task. Rows are discarded
    only when a hand is overrepresented relative to the other hands actually
    present; underrepresented hands are never duplicated.
    """

    if not 0 < maximum_share <= 1:
        raise ValueError("maximum_share must be in (0, 1]")
    tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tasks[row["task"]].append(row)

    kept: list[dict[str, Any]] = []
    task_audit: dict[str, Any] = {}
    for task, task_rows in sorted(tasks.items()):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in task_rows:
            cards = row["_conditioning_cards"]
            buckets[cards.response_structure_signature].append(row)
        counts = {hand: len(items) for hand, items in buckets.items()}
        effective_share = max(maximum_share, 1 / len(buckets))
        target = len(task_rows)
        while target:
            cap = max(1, math.floor(effective_share * target))
            available = sum(min(count, cap) for count in counts.values())
            if available >= target:
                break
            target = available
        cap = max(1, math.floor(effective_share * target)) if target else 0
        selected: list[dict[str, Any]] = []
        for hand, hand_rows in sorted(buckets.items()):
            ranked = sorted(
                hand_rows,
                key=lambda item: hashlib.sha256(
                    f"response-hand:{task}:{hand}:{item['example_id']}".encode()
                ).digest(),
            )
            selected.extend(ranked[:cap])
        if len(selected) > target:
            selected = sorted(
                selected,
                key=lambda item: hashlib.sha256(
                    f"response-target:{task}:{item['example_id']}".encode()
                ).digest(),
            )[:target]
        kept.extend(selected)
        after = Counter(
            row["_conditioning_cards"].response_structure_signature for row in selected
        )
        task_audit[task] = {
            "input_examples": len(task_rows),
            "kept_examples": len(selected),
            "distinct_hands": len(buckets),
            "maximum_hand_count_before": max(counts.values(), default=0),
            "maximum_hand_share_before": round(
                max(counts.values(), default=0) / len(task_rows) if task_rows else 0.0,
                6,
            ),
            "maximum_hand_count_after": max(after.values(), default=0),
            "maximum_hand_share_after": round(
                max(after.values(), default=0) / len(selected) if selected else 0.0,
                6,
            ),
        }
    kept.sort(key=lambda item: item["example_id"])
    return kept, {
        "input_examples": len(rows),
        "kept_examples": len(kept),
        "dropped_overrepresented_response_hands": len(rows) - len(kept),
        "requested_maximum_share": maximum_share,
        "tasks": task_audit,
    }
