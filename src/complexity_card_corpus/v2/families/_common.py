from __future__ import annotations

import hashlib
import json
from typing import Any

from ..contracts import SemanticFrame
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
    semantic_frame: SemanticFrame | None = None,
) -> dict[str, object]:
    pair = deck.deal(case_id)
    frame = semantic_frame or SemanticFrame(
        intent=task,
        facts=facts,
        constraints=(str(validator.get("kind", "unspecified")),),
        expected_outcome=validator,
        uncertainty="bounded" if task == "safety_uncertainty" else "none",
        user_tone=pair.user_tone,
    )
    assistant = (
        render_think_final(pair.thinking, pair.answer)
        if pair.thinking
        else pair.answer
    )
    messages = [
        {"role": turn.role, "content": turn.content}
        for turn in frame.history
    ]
    messages.extend(
        (
            {"role": "user", "content": pair.prompt},
            {"role": "assistant", "content": assistant},
        )
    )
    rendered = json.dumps(messages, sort_keys=True)
    composition = {
        "intent": frame.intent,
        "domain": domain,
        "deck_name": deck.name,
        "prompt_plan": pair.prompt_plan,
        "answer_plan": pair.answer_plan,
        "thinking_plan": pair.thinking_plan,
        "prompt_functions": pair.prompt_functions,
        "answer_functions": pair.answer_functions,
        "thinking_functions": pair.thinking_functions,
        "user_tone": frame.user_tone,
        "thinking_budget": pair.thinking_budget,
        "allowed_prompt_answer_edges": pair.allowed_prompt_answer_edges,
        "allowed_answer_thinking_edges": pair.allowed_answer_thinking_edges,
    }
    return {
        "example_id": f"v2:{task}:"
        + hashlib.sha256(rendered.encode()).hexdigest()[:24],
        "task": task,
        "mode": "chat",
        "difficulty": difficulty,
        "domain": domain,
        "language": "en",
        "split": "train",
        "messages": messages,
        "prompt": pair.prompt,
        "response": assistant,
        "reasoning_envelope": bool(pair.thinking),
        "reasoning_trace": pair.thinking,
        "final_response": pair.answer,
        "source_representation": json.dumps(
            {
                "case_id": case_id,
                "facts": facts,
                "semantic_frame": frame.as_metadata(),
                "composition": composition,
                "prompt_subcards": pair.prompt_subcards,
                "thinking_subcards": pair.thinking_subcards,
                "answer_subcards": pair.answer_subcards,
                "variable_by": deck.variables.matrix.field_names(),
                "deck_name": deck.name,
                "variable_indices": pair.variable_indices,
                "variable_card_counts": pair.variable_card_counts,
                "dependency_graph": pair.dependency_graph,
                "prompt_plan": pair.prompt_plan,
                "answer_plan": pair.answer_plan,
                "thinking_plan": pair.thinking_plan,
                "prompt_functions": pair.prompt_functions,
                "answer_functions": pair.answer_functions,
                "thinking_functions": pair.thinking_functions,
                "thinking_budget": pair.thinking_budget,
                "allowed_prompt_answer_edges": pair.allowed_prompt_answer_edges,
                "allowed_answer_thinking_edges": pair.allowed_answer_thinking_edges,
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
