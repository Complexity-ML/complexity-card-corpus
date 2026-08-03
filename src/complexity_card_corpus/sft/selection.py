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
    max_per_structure: int | None = None,
    per_task_limits: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bound examples per task, semantic domain, and normalized answer shape.

    The same response form can legitimately teach different subject matter in
    two domains.  Collapsing those rows before domain balancing silently
    removes authored coverage, so structural duplicates are bounded only
    inside their own semantic domain.
    """

    if max_per_structure is not None and max_per_structure < 1:
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
        if limit is not None and retained[key] >= limit:
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
        "maximum_retained_per_structure": (
            max(
                value
                for value in (max_per_structure, *per_task_limits.values())
                if value is not None
            )
            if max_per_structure is not None or per_task_limits
            else None
        ),
        "default_maximum_retained_per_structure": max_per_structure,
        "per_task_structure_limits": dict(sorted(per_task_limits.items())),
        "policy": (
            "manual_opt_in_cap"
            if max_per_structure is not None or per_task_limits
            else "preserve_all_non_exact_structures"
        ),
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
    max_examples_per_family: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cap dominant families without duplicating minority examples."""

    if max_examples_per_family is not None and max_examples_per_family < 1:
        raise ValueError("max_examples_per_family must be positive")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["task"]].append(row)
    kept: list[dict[str, Any]] = []
    before = {task: len(items) for task, items in sorted(buckets.items())}
    if max_examples_per_family is None:
        kept = sorted(rows, key=lambda item: item["example_id"])
        total = len(kept)
        shares = {
            task: round(count / total, 6) if total else 0.0
            for task, count in before.items()
        }
        return kept, {
            "input_examples": total,
            "kept_examples": total,
            "dropped_for_family_balance": 0,
            "maximum_examples_per_family": None,
            "selection_strategy": "preserve_all_audit_ratios",
            "before": before,
            "after": before,
            "shares": shares,
        }
    for task, items in sorted(buckets.items()):
        # A global hash sample can accidentally skew a previously balanced
        # family toward a few semantic domains.  That later makes perfectly
        # valid domain-specific wording look like boilerplate.  Deal the cap
        # round-robin across domains instead, with deterministic hash ordering
        # inside every domain bucket.
        domain_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            domain_buckets[str(item.get("domain", "__unspecified__"))].append(item)
        ranked_domains = {
            domain: sorted(
                domain_items,
                key=lambda item: hashlib.sha256(
                    f"family-balance:{task}:{domain}:{item['example_id']}".encode()
                ).digest(),
            )
            for domain, domain_items in sorted(domain_buckets.items())
        }
        selected: list[dict[str, Any]] = []
        depth = 0
        while len(selected) < min(max_examples_per_family, len(items)):
            added = False
            for domain in sorted(ranked_domains):
                domain_items = ranked_domains[domain]
                if depth >= len(domain_items):
                    continue
                selected.append(domain_items[depth])
                added = True
                if len(selected) == min(max_examples_per_family, len(items)):
                    break
            if not added:
                break
            depth += 1
        kept.extend(selected)
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
        "selection_strategy": "deterministic_domain_round_robin",
        "before": before,
        "after": after,
        "shares": shares,
    }


def _balance_task_domains(
    rows: list[dict[str, Any]],
    *,
    maximum_share: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cap semantic-domain skew inside each family without upsampling.

    A five-percent ceiling is mathematically possible only when a family has
    at least twenty realized domains.  Smaller families use their unavoidable
    ``1 / domain_count`` floor and are reported explicitly for later tank
    hydration rather than hidden behind duplicated examples.
    """

    if maximum_share is not None and not 0 < maximum_share <= 1:
        raise ValueError("maximum_share must be in (0, 1]")
    tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tasks[row["task"]].append(row)

    if maximum_share is None:
        task_audit = {}
        for task, task_rows in sorted(tasks.items()):
            counts = Counter(row["domain"] for row in task_rows)
            task_audit[task] = {
                "input_examples": len(task_rows),
                "kept_examples": len(task_rows),
                "distinct_domains": len(counts),
                "requested_maximum_share": None,
                "effective_maximum_share": None,
                "requires_tank_hydration": False,
                "maximum_domain_share_before": round(
                    max(counts.values(), default=0) / max(1, len(task_rows)), 6
                ),
                "maximum_domain_share_after": round(
                    max(counts.values(), default=0) / max(1, len(task_rows)), 6
                ),
            }
        return sorted(rows, key=lambda item: item["example_id"]), {
            "input_examples": len(rows),
            "kept_examples": len(rows),
            "dropped_overrepresented_domains": 0,
            "requested_maximum_share": None,
            "policy": "preserve_all_audit_ratios",
            "tasks_requiring_tank_hydration": [],
            "tasks": task_audit,
        }

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
    maximum_share: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prevent one invisible response-card hand from dominating a family.

    The target size is solved independently for each task. Rows are discarded
    only when a hand is overrepresented relative to the other hands actually
    present; underrepresented hands are never duplicated.
    """

    if maximum_share is not None and not 0 < maximum_share <= 1:
        raise ValueError("maximum_share must be in (0, 1]")
    tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tasks[row["task"]].append(row)

    if maximum_share is None:
        task_audit = {}
        for task, task_rows in sorted(tasks.items()):
            counts = Counter(
                row["_conditioning_cards"].response_structure_signature
                for row in task_rows
            )
            share = max(counts.values(), default=0) / max(1, len(task_rows))
            task_audit[task] = {
                "input_examples": len(task_rows),
                "kept_examples": len(task_rows),
                "distinct_hands": len(counts),
                "maximum_hand_count_before": max(counts.values(), default=0),
                "maximum_hand_share_before": round(share, 6),
                "maximum_hand_count_after": max(counts.values(), default=0),
                "maximum_hand_share_after": round(share, 6),
            }
        return sorted(rows, key=lambda item: item["example_id"]), {
            "input_examples": len(rows),
            "kept_examples": len(rows),
            "dropped_overrepresented_response_hands": 0,
            "requested_maximum_share": None,
            "policy": "preserve_all_audit_ratios",
            "tasks": task_audit,
        }

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
