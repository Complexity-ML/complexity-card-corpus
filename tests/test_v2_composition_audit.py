from __future__ import annotations

import json

from complexity_card_corpus.v2 import audit_v2_composition


def _row(
    index: int,
    *,
    task: str = "casual_conversation",
    prompt_plan: str = "prompt-0",
    answer_plan: str = "answer-0",
    multi_turn: bool = True,
    thinking_budget: str = "none",
) -> dict:
    messages = [
        {"role": "user", "content": f"Remember value {index}."},
        {"role": "assistant", "content": "Okay."},
        {"role": "user", "content": "And now?"},
        {"role": "assistant", "content": f"Result {index}."},
    ]
    if not multi_turn:
        messages = messages[-2:]
    return {
        "task": task,
        "messages": messages,
        "source_representation": json.dumps(
            {
                "composition": {
                    "prompt_plan": prompt_plan,
                    "answer_plan": answer_plan,
                    "thinking_plan": f"thinking-{thinking_budget}",
                    "prompt_functions": [f"request-{prompt_plan}"],
                    "answer_functions": [f"respond-{answer_plan}"],
                    "thinking_functions": [thinking_budget],
                    "thinking_budget": thinking_budget,
                }
            }
        ),
    }


def test_v2_composition_audit_accepts_balanced_independent_plans() -> None:
    rows = [
        _row(
            index,
            prompt_plan=f"prompt-{index % 4}",
            answer_plan=f"answer-{(index // 4) % 4}",
        )
        for index in range(160)
    ]

    audit = audit_v2_composition(rows)
    task = audit["tasks"]["casual_conversation"]

    assert audit["passed"] is True
    assert task["distinct_prompt_answer_edges"] == 16
    assert task["top_prompt_answer_edge_share"] <= 0.25
    assert task["multi_turn_share"] == 1.0


def test_v2_composition_audit_rejects_lexical_variety_with_one_plan() -> None:
    rows = [
        _row(index, multi_turn=False)
        for index in range(100)
    ]
    rows[0].pop("source_representation")

    audit = audit_v2_composition(rows)
    failures = audit["tasks"]["casual_conversation"]["failures"]

    assert "composition_provenance_unavailable" in failures
    assert "insufficient_contextual_multi_turn" in failures
