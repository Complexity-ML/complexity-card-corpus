from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")
_STRUCTURED_RESPONSE_TASKS = {"extraction_classification"}


def _normalized_opening(text: str, *, words: int = SFT_OPENING_WORDS) -> str:
    """Return a slot-normalized lexical signature for an answer opening."""

    if words < 1:
        raise ValueError("opening signature must contain at least one word")
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

    text = _ROLE_LABEL.sub("", text)
    text = _LIST_MARKER.sub("", text)
    return tuple(_OPENING_TOKEN.findall(_normalized_structure(text)))


def _text_repetition_signatures(text: str, *, side: str) -> dict[str, set[str]]:
    """Extract exact, structural, edge, sentence and internal-span signatures.

    Every value is a set so a phrase repeated twice inside one example counts as
    one affected example. Shares therefore remain percentages of examples, not
    percentages of all n-grams emitted by a long answer.
    """

    compact = re.sub(r"\s+", " ", _ROLE_LABEL.sub("", text)).strip().lower()
    structure = _normalized_structure(_ROLE_LABEL.sub("", text))
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
    for sentence in _SENTENCE_BOUNDARY.split(_ROLE_LABEL.sub("", text)):
        sentence_tokens = _normalized_lexical_tokens(sentence)
        if len(sentence_tokens) >= 4:
            sentences.add(" ".join(sentence_tokens))
    signatures[f"{side}_sentence"] = sentences
    return signatures


def audit_sft_repetition_quality(
    rows: list[dict[str, Any]],
    *,
    prompt_key: str = "_projected_prompt",
    target_key: str = "_projected_target",
    maximum_share: float = MAXIMUM_SFT_REPETITION_SHARE,
    minimum_examples: int = MINIMUM_REPETITION_AUDIT_EXAMPLES,
) -> dict[str, Any]:
    """Audit every material form of SFT repetition inside each family.

    The audit covers prompt and response duplicates, normalized structures,
    3/5/8-word openings and endings, repeated full sentences, repeated internal
    8-word spans, and invisible response-card hands. Structured JSON responses
    are exempt only from prose-shape checks; their exact responses and card
    hands remain audited.
    """

    if not 0 < maximum_share <= 1:
        raise ValueError("maximum repetition share must be in (0, 1]")
    if minimum_examples < 1:
        raise ValueError("minimum repetition audit examples must be positive")

    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_task[row["task"]].append(row)

    tasks: dict[str, Any] = {}
    violations: list[dict[str, Any]] = []
    for task, task_rows in sorted(rows_by_task.items()):
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

        total = len(task_rows)
        task_audited = total >= minimum_examples
        dimensions: dict[str, Any] = {}
        for dimension, counts in sorted(counters.items()):
            if not counts:
                continue
            signature, count = counts.most_common(1)[0]
            share = count / total
            structured_prose_exempt = (
                task in _STRUCTURED_RESPONSE_TASKS
                and dimension.startswith("response_")
                and dimension not in {"response_exact", "response_card_hand"}
            )
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
            "examples": total,
            "audited": task_audited,
            "passed": all(item["passed"] for item in dimensions.values()),
            "dimensions": dimensions,
        }

    return {
        "maximum_allowed_share": maximum_share,
        "minimum_examples": minimum_examples,
        "opening_and_closing_windows": list(SFT_REPETITION_WINDOWS),
        "internal_span_words": max(SFT_REPETITION_WINDOWS),
        "passed": not violations,
        "violations": violations,
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
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        target = row["_projected_target"]
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
        by_task[row["task"]][_normalized_structure(target)] += 1
    if hits:
        raise ValueError(f"model-facing answer contains a control rubric: {hits[0]}")
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
        "tasks": task_stats,
    }
