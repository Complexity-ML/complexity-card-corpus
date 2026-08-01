from __future__ import annotations

import json
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
