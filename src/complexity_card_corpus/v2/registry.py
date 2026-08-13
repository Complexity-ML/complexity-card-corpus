from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .families import (
    brainstorming_creativity_capacity,
    casual_conversation_capacity,
    context_clarification_capacity,
    conversation_empathy_capacity,
    critique_revision_capacity,
    extraction_classification_capacity,
    explanation_learning_capacity,
    grounded_qa_capacity,
    planning_comparison_capacity,
    practical_action_capacity,
    reasoning_verification_capacity,
    render_brainstorming_creativity_rows,
    render_casual_conversation_rows,
    render_context_clarification_rows,
    render_conversation_empathy_rows,
    render_critique_revision_rows,
    render_extraction_classification_rows,
    render_explanation_learning_rows,
    render_grounded_qa_rows,
    render_planning_comparison_rows,
    render_practical_action_rows,
    render_reasoning_verification_rows,
    render_safety_uncertainty_rows,
    render_summarization_synthesis_rows,
    render_troubleshooting_rows,
    troubleshooting_capacity,
    writing_transformation_capacity,
    safety_uncertainty_capacity,
    summarization_synthesis_capacity,
    render_writing_transformation_rows,
)
from .gates import v2_gate_progress
from .plan import ALL_TASKS
from .split_audit import composition_split_key


FamilyRenderer = Callable[[], list[dict[str, Any]]]

FAMILY_RENDERERS: dict[str, FamilyRenderer] = {
    "brainstorming_creativity": render_brainstorming_creativity_rows,
    "casual_conversation": render_casual_conversation_rows,
    "context_clarification": render_context_clarification_rows,
    "conversation_empathy": render_conversation_empathy_rows,
    "critique_revision": render_critique_revision_rows,
    "extraction_classification": render_extraction_classification_rows,
    "explanation_learning": render_explanation_learning_rows,
    "grounded_qa": render_grounded_qa_rows,
    "planning_comparison": render_planning_comparison_rows,
    "practical_action": render_practical_action_rows,
    "reasoning_verification": render_reasoning_verification_rows,
    "safety_uncertainty": render_safety_uncertainty_rows,
    "summarization_synthesis": render_summarization_synthesis_rows,
    "troubleshooting": render_troubleshooting_rows,
    "writing_transformation": render_writing_transformation_rows,
}
FAMILY_CAPACITIES: dict[str, Callable[[], int]] = {
    "brainstorming_creativity": brainstorming_creativity_capacity,
    "casual_conversation": casual_conversation_capacity,
    "context_clarification": context_clarification_capacity,
    "conversation_empathy": conversation_empathy_capacity,
    "critique_revision": critique_revision_capacity,
    "extraction_classification": extraction_classification_capacity,
    "explanation_learning": explanation_learning_capacity,
    "grounded_qa": grounded_qa_capacity,
    "planning_comparison": planning_comparison_capacity,
    "practical_action": practical_action_capacity,
    "reasoning_verification": reasoning_verification_capacity,
    "safety_uncertainty": safety_uncertainty_capacity,
    "summarization_synthesis": summarization_synthesis_capacity,
    "troubleshooting": troubleshooting_capacity,
    "writing_transformation": writing_transformation_capacity,
}


def v2_generation_progress() -> dict[str, object]:
    registered = tuple(task for task in ALL_TASKS if task in FAMILY_RENDERERS)
    missing = tuple(task for task in ALL_TASKS if task not in FAMILY_RENDERERS)
    covered = sum(FAMILY_CAPACITIES[task]() for task in registered)
    return {
        "example_limit": None,
        "registered_capacity": covered,
        "registered_families": registered,
        "missing_families": missing,
    }


def _assign_release_split(row: dict[str, Any]) -> None:
    """Reserve deterministic evaluation rows while keeping public anchors trained."""

    try:
        case_id = str(json.loads(str(row["source_representation"]))["case_id"])
    except (KeyError, TypeError, json.JSONDecodeError):
        case_id = ""
    if case_id.startswith("anchor:"):
        row["split"] = "train"
        return
    material = json.dumps(composition_split_key(row), separators=(",", ":"))
    bucket = int.from_bytes(
        hashlib.sha256(material.encode()).digest()[:2],
        "big",
    ) % 100
    row["split"] = "validation" if bucket == 0 else "test" if bucket == 1 else "train"


def _ensure_family_heldouts(rows: list[dict[str, Any]]) -> None:
    """Guarantee both held-out roles without cutting through composition groups."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task"])].append(row)
    for task, task_rows in grouped.items():
        populated = {str(row["split"]) for row in task_rows}
        for required in ("validation", "test"):
            if required in populated:
                continue
            candidates: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
            for row in task_rows:
                if row["split"] != "train":
                    continue
                try:
                    case_id = str(
                        json.loads(str(row["source_representation"]))["case_id"]
                    )
                except (KeyError, TypeError, json.JSONDecodeError):
                    case_id = ""
                if case_id.startswith("anchor:"):
                    continue
                candidates[composition_split_key(row)].append(row)
            if not candidates:
                raise ValueError(f"{task} has no composition group for {required}")
            selected = min(
                candidates,
                key=lambda key: hashlib.sha256(
                    f"{task}:{required}:{key}".encode()
                ).digest(),
            )
            for row in candidates[selected]:
                row["split"] = required
            populated.add(required)


def render_complete_v2() -> list[dict[str, Any]]:
    """Render every independently authored family at its full valid capacity."""

    progress = v2_generation_progress()
    missing = progress["missing_families"]
    if missing:
        raise RuntimeError(
            "Card Corpus V2 is incomplete; missing family renderers: "
            + ", ".join(missing)
        )
    gate_progress = v2_gate_progress()
    if not gate_progress["complete"]:
        raise RuntimeError(
            "Card Corpus V2 release gates are incomplete: "
            + ", ".join(gate_progress["missing"])
        )
    rows = [
        row
        for task in ALL_TASKS
        for row in FAMILY_RENDERERS[task]()
    ]
    counts = {
        task: sum(row["task"] == task for row in rows)
        for task in ALL_TASKS
    }
    wrong = {
        task: (counts[task], FAMILY_CAPACITIES[task]())
        for task in ALL_TASKS
        if counts[task] != FAMILY_CAPACITIES[task]()
    }
    if wrong:
        raise ValueError(f"Card Corpus V2 family capacity mismatch: {wrong}")
    if len({str(row["example_id"]) for row in rows}) != len(rows):
        raise ValueError("Card Corpus V2 produced duplicate example IDs")
    for row in rows:
        _assign_release_split(row)
    _ensure_family_heldouts(rows)
    return rows


__all__ = (
    "FAMILY_CAPACITIES",
    "FAMILY_RENDERERS",
    "render_complete_v2",
    "v2_generation_progress",
)
