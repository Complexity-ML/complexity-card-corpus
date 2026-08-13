from __future__ import annotations

import hashlib
import json
from typing import Any

from ..decks import V2RoleSeparatedDeck
from ..integrity_audit import render_think_final


def render_v2_row(
    *,
    task: str,
    case_id: str,
    domain: str,
    difficulty: str,
    deck: V2RoleSeparatedDeck,
    facts: dict[str, Any],
    validator: dict[str, Any],
) -> dict[str, object]:
    pair = deck.deal(case_id)
    assistant = (
        render_think_final(pair.thinking, pair.answer)
        if pair.thinking
        else pair.answer
    )
    rendered = f"User: {pair.prompt}\nAssistant: {assistant}"
    return {
        "example_id": f"v2:{task}:"
        + hashlib.sha256(rendered.encode()).hexdigest()[:24],
        "task": task,
        "mode": "chat",
        "difficulty": difficulty,
        "domain": domain,
        "language": "en",
        "split": "train",
        "messages": [
            {"role": "user", "content": pair.prompt},
            {"role": "assistant", "content": assistant},
        ],
        "prompt": pair.prompt,
        "response": assistant,
        "reasoning_envelope": bool(pair.thinking),
        "reasoning_trace": pair.thinking,
        "final_response": pair.answer,
        "source_representation": json.dumps(
            {
                "case_id": case_id,
                "facts": facts,
                "prompt_subcards": pair.prompt_subcards,
                "thinking_subcards": pair.thinking_subcards,
                "answer_subcards": pair.answer_subcards,
                "variable_by": deck.variables.matrix.field_names(),
                "deck_name": deck.name,
                "variable_indices": pair.variable_indices,
                "variable_card_counts": pair.variable_card_counts,
                "dependency_graph": pair.dependency_graph,
                "validator": validator,
            },
            sort_keys=True,
        ),
        "source": "AETHORIA-AI Card Corpus V2 authored decks",
        "license": "CC BY-NC 4.0",
        "version": "2.0.0",
    }


def validate_complete_rows(
    task: str,
    rows: list[dict[str, object]],
    capacity: int,
) -> list[dict[str, object]]:
    rows.sort(key=lambda row: str(row["example_id"]))
    if len(rows) != capacity:
        raise ValueError(f"{task} rendered {len(rows)} rows; expected {capacity}")
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError(f"{task} produced duplicate example IDs")
    return rows


__all__ = ("render_v2_row", "validate_complete_rows")
