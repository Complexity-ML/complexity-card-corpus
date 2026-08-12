from __future__ import annotations

import json
import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..training_cards import RESPONSE_STRUCTURE_SIBLING_TASKS
from .language import _render_messages
from .selection import _normalized_structure


def load_heldout_evaluation(path: Path) -> list[dict[str, Any]]:
    """Load source-separated held-out exchanges into the common schema."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples", [])
    if not examples:
        raise ValueError("held-out evaluation contains no examples")
    rows: list[dict[str, Any]] = []
    for item in examples:
        messages = [
            {"role": "user", "content": item["prompt"].strip()},
            {"role": "assistant", "content": item["response"].strip()},
        ]
        rows.append(
            {
                "example_id": f"heldout:{item['id']}",
                "task": item["task"],
                "mode": "instruct",
                "difficulty": item.get("difficulty", "medium"),
                "dataset_id": payload["dataset_id"],
                "domain": item["domain"],
                "language": "en",
                "split": "validation",
                "messages": messages,
                "prompt": messages[0]["content"],
                "response": messages[1]["content"],
                "rendered_text": _render_messages(messages),
                "source_keys": [item["id"]],
                "evidence": item.get("evidence", []),
                "answer_json": json.dumps(
                    {
                        "evaluation_source": item.get(
                            "evaluation_source", "separately_authored"
                        ),
                        "use_verbatim_target": True,
                    },
                    sort_keys=True,
                ),
                "source": payload["source"],
                "source_urls": [],
                "license": payload["license"],
                "version": payload["version"],
            }
        )
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("held-out evaluation contains duplicate ids")
    return rows


_FORBIDDEN_SFT_TARGET_PHRASES = (
    "hand ",
    "next step:",
    "owner:",
    "timing:",
    "core idea:",
    "example:",
    "check:",
    "decision:",
    "action:",
    "open point:",
    "open item:",
    "weakness:",
    "revision:",
    "immediate action:",
    "boundary:",
    "sequence:",
    "fallback trigger:",
    "revised text:",
    "assigned action:",
    "remaining work:",
    "blocker:",
    "a concrete example is this:",
    "consider this example:",
    "keep this limit in mind:",
    "each description states",
    "remain feasible under the stated limits",
    "the response should",
    "the final review should",
    "treat the task as complete",
    "if that cannot be established",
    "return to a smaller causal model",
    # Generic answer-development language is not a user-facing answer.  These
    # phrases used to be spread across many lexical variants, allowing every
    # individual n-gram to remain below five percent while preserving the same
    # repetitive discourse act.
    "the supported takeaway",
    "the response can therefore",
    "the answer is complete on this basis",
    "that gives a concrete completion criterion",
    "the evidence and conclusion remain aligned",
    "the closing test is satisfied",
    "the supplied material keeps",
    "the supplied numbers give",
)


_GENERALIST_POST_TRAINING_TASKS = {
    "practical_action",
    "explanation_learning",
    "troubleshooting",
    "writing_transformation",
    "planning_comparison",
    "conversation_empathy",
    "safety_uncertainty",
    "grounded_qa",
    "summarization_synthesis",
    "extraction_classification",
    "reasoning_verification",
    "critique_revision",
    "brainstorming_creativity",
    "context_clarification",
    "casual_conversation",
}

MAXIMUM_SFT_OPENING_SHARE = 0.05
SFT_OPENING_WORDS = 5
MINIMUM_OPENING_AUDIT_EXAMPLES = 100
_OPENING_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
_OPENING_AUDIT_EXEMPT_TASKS = {"extraction_classification"}

MAXIMUM_SFT_REPETITION_SHARE = 0.05
SFT_REPETITION_WINDOWS = (3, 5, 8)
MINIMUM_REPETITION_AUDIT_EXAMPLES = 100
_SENTENCE_BOUNDARY = re.compile(r"(?:[.!?]+|\n+)")
_ROLE_LABEL = re.compile(r"(?im)^\s*(?:user|assistant):\s*")
_REASONING_PROTOCOL_TAG = re.compile(r"</?(?:think|final)>", flags=re.IGNORECASE)
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")
_STRUCTURED_RESPONSE_TASKS = {"extraction_classification"}
_INCOMPLETE_TARGET_ENDING = re.compile(
    r"(?:[:\-—]|\b(?:and|or|because|with|without|to|the|a|an|is|are|was|were))\s*$",
    flags=re.IGNORECASE,
)
_EMPTY_RESPONSE_HEADING = re.compile(
    r"\b(?:comparison result|criteria review|outcome review|practical result|"
    r"fit with the brief|selection|recommendation):\s*$",
    flags=re.IGNORECASE,
)


def _dimension_is_audited(task: str, dimension: str) -> bool:
    if (
        dimension.startswith("response_card_sibling_")
        and task not in RESPONSE_STRUCTURE_SIBLING_TASKS
    ):
        return False
    return not (
        task in _STRUCTURED_RESPONSE_TASKS
        and dimension.startswith("response_")
        and dimension != "response_exact"
        and not dimension.startswith("response_card_")
    )


def _normalized_opening(text: str, *, words: int = SFT_OPENING_WORDS) -> str:
    """Return a slot-normalized lexical signature for an answer opening."""

    if words < 1:
        raise ValueError("opening signature must contain at least one word")
    text = _REASONING_PROTOCOL_TAG.sub(" ", text)
    without_list_marker = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", text)
    tokens = _OPENING_TOKEN.findall(_normalized_structure(without_list_marker))
    return " ".join(tokens[:words])


def audit_sft_opening_diversity(
    rows: list[dict[str, Any]],
    *,
    target_key: str = "_projected_target",
    maximum_share: float = MAXIMUM_SFT_OPENING_SHARE,
    minimum_examples: int = MINIMUM_OPENING_AUDIT_EXAMPLES,
) -> dict[str, Any]:
    """Measure the largest five-word opening inside each SFT family."""

    if not 0 < maximum_share <= 1:
        raise ValueError("maximum opening share must be in (0, 1]")
    if minimum_examples < 1:
        raise ValueError("minimum opening audit examples must be positive")

    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_task[row["task"]][_normalized_opening(row[target_key])] += 1

    tasks: dict[str, Any] = {}
    violations: list[dict[str, Any]] = []
    for task, counts in sorted(by_task.items()):
        total = sum(counts.values())
        opening, count = counts.most_common(1)[0]
        share = count / total
        exempt = task in _OPENING_AUDIT_EXEMPT_TASKS
        audited = total >= minimum_examples and not exempt
        passed = not audited or share <= maximum_share
        item = {
            "examples": total,
            "distinct_openings": len(counts),
            "most_common_opening": opening,
            "most_common_opening_count": count,
            "maximum_opening_share": round(share, 6),
            "exempt": exempt,
            "audited": audited,
            "passed": passed,
        }
        tasks[task] = item
        if not passed:
            violations.append({"task": task, **item})

    return {
        "opening_words": SFT_OPENING_WORDS,
        "maximum_allowed_share": maximum_share,
        "minimum_examples": minimum_examples,
        "passed": not violations,
        "violations": violations,
        "tasks": tasks,
    }


def assert_sft_opening_diversity(
    rows: list[dict[str, Any]],
    *,
    target_key: str = "_projected_target",
    maximum_share: float = MAXIMUM_SFT_OPENING_SHARE,
    minimum_examples: int = MINIMUM_OPENING_AUDIT_EXAMPLES,
) -> dict[str, Any]:
    """Fail SFT preparation when one family exceeds the opening ceiling."""

    audit = audit_sft_opening_diversity(
        rows,
        target_key=target_key,
        maximum_share=maximum_share,
        minimum_examples=minimum_examples,
    )
    if audit["violations"]:
        details = "; ".join(
            f"{item['task']}={item['maximum_opening_share']:.2%} "
            f"({item['most_common_opening']!r})"
            for item in audit["violations"]
        )
        raise ValueError(
            "SFT opening repetition exceeds the "
            f"{maximum_share:.0%} per-family ceiling: " + details
        )
    return audit


def _normalized_lexical_tokens(text: str) -> tuple[str, ...]:
    """Return comparable lexical tokens without chat labels or volatile slots."""

    text = _REASONING_PROTOCOL_TAG.sub(" ", _ROLE_LABEL.sub("", text))
    text = _LIST_MARKER.sub("", text)
    return tuple(_OPENING_TOKEN.findall(_normalized_structure(text)))


def _text_repetition_signatures(text: str, *, side: str) -> dict[str, set[str]]:
    """Extract exact, structural, edge, sentence and internal-span signatures.

    Every value is a set so a phrase repeated twice inside one example counts as
    one affected example. Shares therefore remain percentages of examples, not
    percentages of all n-grams emitted by a long answer.
    """

    protocol_free = _REASONING_PROTOCOL_TAG.sub(" ", _ROLE_LABEL.sub("", text))
    compact = re.sub(r"\s+", " ", protocol_free).strip().lower()
    structure = _normalized_structure(protocol_free)
    tokens = _normalized_lexical_tokens(text)
    signatures: dict[str, set[str]] = {
        f"{side}_exact": {compact} if compact else set(),
        f"{side}_structure": {structure} if structure else set(),
    }
    for size in SFT_REPETITION_WINDOWS:
        signatures[f"{side}_opening_{size}"] = (
            {" ".join(tokens[:size])} if len(tokens) >= size else set()
        )
        signatures[f"{side}_closing_{size}"] = (
            {" ".join(tokens[-size:])} if len(tokens) >= size else set()
        )

    span_size = max(SFT_REPETITION_WINDOWS)
    signatures[f"{side}_span_{span_size}"] = {
        " ".join(tokens[index : index + span_size])
        for index in range(max(0, len(tokens) - span_size + 1))
    }
    sentences: set[str] = set()
    for sentence in _SENTENCE_BOUNDARY.split(protocol_free):
        sentence_tokens = _normalized_lexical_tokens(sentence)
        if len(sentence_tokens) >= 4:
            sentences.add(" ".join(sentence_tokens))
    signatures[f"{side}_sentence"] = sentences
    return signatures


def _row_repetition_signatures(
    row: dict[str, Any],
    *,
    prompt_key: str = "_projected_prompt",
    target_key: str = "_projected_target",
) -> dict[str, set[str]]:
    """Return every audited repetition signature carried by one row."""

    signatures: dict[str, set[str]] = {}
    if prompt_key in row:
        signatures.update(_text_repetition_signatures(row[prompt_key], side="prompt"))
    if target_key in row:
        signatures.update(
            _text_repetition_signatures(row[target_key], side="response")
        )
    cards = row.get("_conditioning_cards")
    if cards is not None:
        signatures["response_card_hand"] = {cards.response_structure_signature}
        signatures.update(
            {
                f"response_card_sibling_{dimension}": {signature}
                for dimension, signature in (
                    cards.response_structure_sibling_signatures.items()
                )
            }
        )
    return signatures


def filter_sft_repetition_quality(
    rows: list[dict[str, Any]],
    *,
    prompt_key: str = "_projected_prompt",
    target_key: str = "_projected_target",
    maximum_share: float = MAXIMUM_SFT_REPETITION_SHARE,
    minimum_examples: int = MINIMUM_REPETITION_AUDIT_EXAMPLES,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop only compositions responsible for a repetition ceiling breach.

    Selection is deterministic and operates independently inside each task.
    The corpus is never made to pass by weakening the ceiling: rows carrying
    the largest number of currently overrepresented signatures are removed,
    then the complete audit is recomputed against the smaller denominator.
    """

    if not 0 < maximum_share <= 1:
        raise ValueError("maximum repetition share must be in (0, 1]")
    if minimum_examples < 1:
        raise ValueError("minimum repetition audit examples must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)

    if workers > 1 and len(by_task) > 1:
        worker_count = min(workers, len(by_task), os.cpu_count() or 1)
        arguments = [
            (task_rows, prompt_key, target_key, maximum_share, minimum_examples)
            for _task, task_rows in sorted(by_task.items())
        ]

        def collect(executor: ProcessPoolExecutor | ThreadPoolExecutor):
            return list(executor.map(_filter_sft_repetition_task, arguments))

        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                results = collect(executor)
        except (NotImplementedError, OSError, PermissionError):
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = collect(executor)

        kept = sorted(
            [row for task_rows, _selection in results for row in task_rows],
            key=lambda row: row["example_id"],
        )
        selections = [selection for _task_rows, selection in results]
        return kept, {
            "method": "deterministic_overrepresented_signature_pruning",
            "execution": "parallel_by_task",
            "workers": worker_count,
            "maximum_allowed_share": maximum_share,
            "input_examples": len(rows),
            "kept_examples": len(kept),
            "dropped_examples": len(rows) - len(kept),
            "tasks": {
                task: stats
                for selection in selections
                for task, stats in selection["tasks"].items()
            },
            "input_audit": _combine_repetition_audits(
                [selection["input_audit"] for selection in selections]
            ),
            "final_audit": _combine_repetition_audits(
                [selection["final_audit"] for selection in selections]
            ),
        }

    input_audit = audit_sft_repetition_quality(
        rows,
        prompt_key=prompt_key,
        target_key=target_key,
        maximum_share=maximum_share,
        minimum_examples=minimum_examples,
    )

    kept: list[dict[str, Any]] = []
    task_audit: dict[str, Any] = {}
    for task, task_rows in sorted(by_task.items()):
        active = {
            row["example_id"]: row
            for row in sorted(task_rows, key=lambda item: item["example_id"])
        }
        signature_by_id = {
            example_id: _row_repetition_signatures(
                row, prompt_key=prompt_key, target_key=target_key
            )
            for example_id, row in active.items()
        }
        counters: dict[str, Counter[str]] = defaultdict(Counter)
        reverse: dict[tuple[str, str], set[str]] = defaultdict(set)
        domains = Counter(str(row.get("domain", "")) for row in active.values())
        domain_ceiling_enabled = len(domains) >= math.ceil(1 / maximum_share)
        hands = Counter(
            row["_conditioning_cards"].response_structure_signature
            for row in active.values()
            if row.get("_conditioning_cards") is not None
        )
        hand_ceiling_enabled = len(hands) >= math.ceil(1 / maximum_share)
        for example_id, dimensions in signature_by_id.items():
            for dimension, values in dimensions.items():
                if not _dimension_is_audited(task, dimension):
                    continue
                counters[dimension].update(values)
                for value in values:
                    reverse[(dimension, value)].add(example_id)
        rounds = 0
        while len(active) >= minimum_examples:
            rounds += 1
            ceiling = int(maximum_share * len(active))
            violations = {
                (dimension, value): count
                for dimension, counts in counters.items()
                for value, count in counts.items()
                if count > ceiling
            }
            if not violations:
                break

            scores: Counter[str] = Counter()
            for signature, count in violations.items():
                excess = count - ceiling
                for example_id in reverse[signature]:
                    if example_id in active:
                        scores[example_id] += excess
            maximum_excess = max(count - ceiling for count in violations.values())
            ranked = sorted(
                scores,
                key=lambda example_id: (
                    -(
                        max(
                            0,
                            domains[str(active[example_id].get("domain", ""))]
                            - ceiling,
                        )
                        if domain_ceiling_enabled
                        else 0
                    ),
                    -(
                        max(
                            0,
                            hands[
                                active[
                                    example_id
                                ]["_conditioning_cards"].response_structure_signature
                            ]
                            - ceiling,
                        )
                        if hand_ceiling_enabled
                        and active[example_id].get("_conditioning_cards") is not None
                        else 0
                    ),
                    -scores[example_id],
                    hashlib.sha256(
                        f"repetition-filter:{task}:{example_id}".encode()
                    ).digest(),
                ),
            )
            removable = min(maximum_excess, len(active) - minimum_examples)
            if removable <= 0:
                break
            removal_ids: list[str] = []
            if domain_ceiling_enabled:
                next_size = len(active) - removable
                domain_floor = next_size // len(domains)
                domain_capacity = {
                    domain: max(0, count - domain_floor)
                    for domain, count in domains.items()
                }
                for example_id in ranked:
                    domain = str(active[example_id].get("domain", ""))
                    if domain_capacity[domain] <= 0:
                        continue
                    removal_ids.append(example_id)
                    domain_capacity[domain] -= 1
                    if len(removal_ids) == removable:
                        break
            if len(removal_ids) < removable:
                selected_ids = set(removal_ids)
                removal_ids.extend(
                    example_id
                    for example_id in ranked
                    if example_id not in selected_ids
                )
                removal_ids = removal_ids[:removable]

            for example_id in removal_ids:
                row = active.pop(example_id, None)
                if row is None:
                    continue
                domains[str(row.get("domain", ""))] -= 1
                cards = row.get("_conditioning_cards")
                if cards is not None:
                    hands[cards.response_structure_signature] -= 1
                for dimension, values in signature_by_id[example_id].items():
                    if not _dimension_is_audited(task, dimension):
                        continue
                    counters[dimension].subtract(values)

        family_rows = list(active.values())
        kept.extend(family_rows)
        task_audit[task] = {
            "input_examples": len(task_rows),
            "kept_examples": len(family_rows),
            "dropped_examples": len(task_rows) - len(family_rows),
            "selection_rounds": rounds,
            "fell_below_audit_minimum": len(family_rows) < minimum_examples,
            "domain_ceiling_preserved": domain_ceiling_enabled,
            "response_card_hand_ceiling_preserved": hand_ceiling_enabled,
        }

    kept.sort(key=lambda row: row["example_id"])
    final_audit = audit_sft_repetition_quality(
        kept,
        prompt_key=prompt_key,
        target_key=target_key,
        maximum_share=maximum_share,
        minimum_examples=minimum_examples,
    )
    return kept, {
        "method": "deterministic_overrepresented_signature_pruning",
        "execution": "serial_by_task",
        "workers": 1,
        "maximum_allowed_share": maximum_share,
        "input_examples": len(rows),
        "kept_examples": len(kept),
        "dropped_examples": len(rows) - len(kept),
        "tasks": task_audit,
        "input_audit": input_audit,
        "final_audit": final_audit,
    }


def _filter_sft_repetition_task(
    arguments: tuple[list[dict[str, Any]], str, str, float, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, prompt_key, target_key, maximum_share, minimum_examples = arguments
    return filter_sft_repetition_quality(
        rows,
        prompt_key=prompt_key,
        target_key=target_key,
        maximum_share=maximum_share,
        minimum_examples=minimum_examples,
        workers=1,
    )


def _combine_repetition_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    if not audits:
        return {
            "maximum_allowed_share": MAXIMUM_SFT_REPETITION_SHARE,
            "minimum_examples": MINIMUM_REPETITION_AUDIT_EXAMPLES,
            "maximum_examples_per_task": None,
            "opening_and_closing_windows": list(SFT_REPETITION_WINDOWS),
            "internal_span_words": max(SFT_REPETITION_WINDOWS),
            "passed": True,
            "supervised_passed": True,
            "violations": [],
            "supervised_violations": [],
            "tasks": {},
        }
    first = audits[0]
    violations = [item for audit in audits for item in audit["violations"]]
    supervised_violations = [
        item
        for audit in audits
        for item in audit.get("supervised_violations", [])
    ]
    return {
        "maximum_allowed_share": first["maximum_allowed_share"],
        "minimum_examples": first["minimum_examples"],
        "maximum_examples_per_task": first["maximum_examples_per_task"],
        "opening_and_closing_windows": first["opening_and_closing_windows"],
        "internal_span_words": first["internal_span_words"],
        "passed": not violations,
        "supervised_passed": not supervised_violations,
        "violations": violations,
        "supervised_violations": supervised_violations,
        "tasks": {
            task: stats
            for audit in audits
            for task, stats in audit["tasks"].items()
        },
    }


def audit_sft_repetition_quality(
    rows: list[dict[str, Any]],
    *,
    prompt_key: str = "_projected_prompt",
    target_key: str = "_projected_target",
    maximum_share: float = MAXIMUM_SFT_REPETITION_SHARE,
    minimum_examples: int = MINIMUM_REPETITION_AUDIT_EXAMPLES,
    maximum_examples_per_task: int | None = None,
) -> dict[str, Any]:
    """Audit every material form of SFT repetition inside each family.

    The audit covers prompt and response duplicates, normalized structures,
    3/5/8-word openings and endings, repeated full sentences, repeated internal
    8-word spans, invisible response-card hands, and every one-card-away
    response-hand neighbourhood. Structured JSON responses are exempt only
    from prose-shape checks; their exact responses and card structures remain
    audited.
    """

    if not 0 < maximum_share <= 1:
        raise ValueError("maximum repetition share must be in (0, 1]")
    if minimum_examples < 1:
        raise ValueError("minimum repetition audit examples must be positive")
    if maximum_examples_per_task is not None and maximum_examples_per_task < 1:
        raise ValueError("maximum_examples_per_task must be positive")

    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_task[row["task"]].append(row)

    tasks: dict[str, Any] = {}
    violations: list[dict[str, Any]] = []
    for task, population_rows in sorted(rows_by_task.items()):
        task_rows = population_rows
        if (
            maximum_examples_per_task is not None
            and len(task_rows) > maximum_examples_per_task
        ):
            task_rows = sorted(
                task_rows,
                key=lambda row: hashlib.sha256(
                    f"repetition-audit:{task}:{row['example_id']}".encode()
                ).digest(),
            )[:maximum_examples_per_task]
        counters: dict[str, Counter[str]] = defaultdict(Counter)
        for row in task_rows:
            if prompt_key in row:
                for dimension, signatures in _text_repetition_signatures(
                    row[prompt_key], side="prompt"
                ).items():
                    counters[dimension].update(signatures)
            if target_key in row:
                for dimension, signatures in _text_repetition_signatures(
                    row[target_key], side="response"
                ).items():
                    counters[dimension].update(signatures)
            cards = row.get("_conditioning_cards")
            if cards is not None:
                counters["response_card_hand"][cards.response_structure_signature] += 1
                for dimension, signature in (
                    cards.response_structure_sibling_signatures.items()
                ):
                    counters[f"response_card_sibling_{dimension}"][signature] += 1

        total = len(task_rows)
        task_audited = total >= minimum_examples
        dimensions: dict[str, Any] = {}
        for dimension, counts in sorted(counters.items()):
            if not counts:
                continue
            signature, count = min(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
            share = count / total
            structured_prose_exempt = not _dimension_is_audited(task, dimension)
            audited = task_audited and not structured_prose_exempt
            passed = not audited or share <= maximum_share
            item = {
                "distinct_signatures": len(counts),
                "most_common_signature": signature,
                "most_common_count": count,
                "maximum_share": round(share, 6),
                "structured_prose_exempt": structured_prose_exempt,
                "audited": audited,
                "passed": passed,
            }
            dimensions[dimension] = item
            if not passed:
                violations.append({"task": task, "dimension": dimension, **item})
        tasks[task] = {
            "population_examples": len(population_rows),
            "examples": total,
            "sampling": (
                "deterministic_sha256"
                if len(task_rows) < len(population_rows)
                else "full_population"
            ),
            "audited": task_audited,
            "passed": all(item["passed"] for item in dimensions.values()),
            "supervised_passed": all(
                item["passed"]
                for name, item in dimensions.items()
                if name.startswith("response_")
            ),
            "dimensions": dimensions,
        }

    supervised_violations = [
        item for item in violations if item["dimension"].startswith("response_")
    ]
    return {
        "maximum_allowed_share": maximum_share,
        "minimum_examples": minimum_examples,
        "maximum_examples_per_task": maximum_examples_per_task,
        "opening_and_closing_windows": list(SFT_REPETITION_WINDOWS),
        "internal_span_words": max(SFT_REPETITION_WINDOWS),
        "passed": not violations,
        "supervised_passed": not supervised_violations,
        "violations": violations,
        "supervised_violations": supervised_violations,
        "tasks": tasks,
    }


def assert_sft_repetition_quality(
    rows: list[dict[str, Any]],
    *,
    prompt_key: str = "_projected_prompt",
    target_key: str = "_projected_target",
    maximum_share: float = MAXIMUM_SFT_REPETITION_SHARE,
    minimum_examples: int = MINIMUM_REPETITION_AUDIT_EXAMPLES,
) -> dict[str, Any]:
    """Reject tokenization when any audited repetition exceeds five percent."""

    audit = audit_sft_repetition_quality(
        rows,
        prompt_key=prompt_key,
        target_key=target_key,
        maximum_share=maximum_share,
        minimum_examples=minimum_examples,
    )
    if audit["violations"]:
        details = "; ".join(
            f"{item['task']}.{item['dimension']}="
            f"{item['maximum_share']:.2%} ({item['most_common_signature']!r})"
            for item in audit["violations"]
        )
        raise ValueError(
            "SFT repetition exceeds the "
            f"{maximum_share:.0%} per-family ceiling: " + details
        )
    return audit


def _audit_sft_projection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hits: list[dict[str, str]] = []
    incomplete_targets: list[dict[str, str]] = []
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        target = row["_projected_target"].strip()
        lowered = target.lower()
        for phrase in _FORBIDDEN_SFT_TARGET_PHRASES:
            if phrase in lowered:
                hits.append(
                    {
                        "example_id": row["example_id"],
                        "task": row["task"],
                        "phrase": phrase,
                    }
                )
        if not target or _INCOMPLETE_TARGET_ENDING.search(target):
            incomplete_targets.append(
                {
                    "example_id": row["example_id"],
                    "task": row["task"],
                    "ending": target[-120:],
                }
            )
        elif _EMPTY_RESPONSE_HEADING.search(target):
            incomplete_targets.append(
                {
                    "example_id": row["example_id"],
                    "task": row["task"],
                    "ending": target[-120:],
                }
            )
        by_task[row["task"]][_normalized_structure(target)] += 1
    if hits:
        first_by_phrase = {
            hit["phrase"]: hit
            for hit in reversed(hits)
        }
        raise ValueError(
            "model-facing answer contains a control rubric: "
            f"{list(first_by_phrase.values())}; total_hits={len(hits)}"
        )
    if incomplete_targets:
        raise ValueError(
            "model-facing answer is incomplete: "
            f"{incomplete_targets[0]}"
        )
    task_stats = {
        task: {
            "examples": sum(counts.values()),
            "distinct_normalized_structures": len(counts),
            "maximum_structure_share": round(
                max(counts.values()) / sum(counts.values()), 6
            ),
        }
        for task, counts in sorted(by_task.items())
    }
    underspecified = [
        task
        for task, stats in task_stats.items()
        if task in _GENERALIST_POST_TRAINING_TASKS
        and stats["examples"] > 1
        and stats["distinct_normalized_structures"] < 2
    ]
    if underspecified:
        raise ValueError(
            f"model-facing family has only one normalized structure: {underspecified}"
        )
    return {
        "examples": len(rows),
        "exact_answer_uniqueness_ratio": round(
            len({row["_projected_target"] for row in rows}) / len(rows), 6
        ),
        "control_rubric_hits": 0,
        "incomplete_target_hits": 0,
        "tasks": task_stats,
    }
