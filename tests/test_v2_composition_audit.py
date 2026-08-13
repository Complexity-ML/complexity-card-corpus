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
    remembered = f"value {index}"
    messages = [
        {"role": "user", "content": f"Remember {remembered}."},
        {"role": "assistant", "content": "Okay."},
        {"role": "user", "content": "And now?"},
        {"role": "assistant", "content": f"Result {index}."},
    ]
    if not multi_turn:
        messages = messages[-2:]
    prompt_answer_edges = [
        [f"prompt-{prompt_index}", f"answer-{answer_index}"]
        for prompt_index in range(4)
        for answer_index in range(4)
    ]
    answer_thinking_edges = [
        [f"answer-{answer_index}", f"thinking-{thinking_budget}"]
        for answer_index in range(4)
    ]
    return {
        "task": task,
        "messages": messages,
        "source_representation": json.dumps(
            {
                "facts": {"remembered": remembered},
                "semantic_frame": {
                    "intent": "contextual_follow_up",
                    "facts": {"remembered": remembered},
                    "constraints": [],
                    "expected_outcome": index,
                    "uncertainty": "none",
                    "user_tone": "casual",
                    "history": messages[:-2],
                    "history_required_facts": ["remembered"] if multi_turn else [],
                },
                "composition": {
                    "intent": "contextual_follow_up",
                    "domain": "test",
                    "deck_name": "balanced-test-deck",
                    "prompt_plan": prompt_plan,
                    "answer_plan": answer_plan,
                    "thinking_plan": f"thinking-{thinking_budget}",
                    "prompt_functions": [f"request-{prompt_plan}"],
                    "answer_functions": [f"respond-{answer_plan}"],
                    "thinking_functions": [thinking_budget],
                    "user_tone": "casual",
                    "thinking_budget": thinking_budget,
                    "allowed_prompt_answer_edges": prompt_answer_edges,
                    "allowed_answer_thinking_edges": answer_thinking_edges,
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
    assert task["genuine_multi_turn_rows"] == 160
    assert task["prompt_answer_edge_coverage"] == 1.0
    assert task["answer_thinking_edge_coverage"] == 1.0


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


def test_v2_composition_audit_rejects_history_repeated_in_current_prompt() -> None:
    row = _row(7)
    row["messages"][-2]["content"] = "And now with value 7?"

    audit = audit_v2_composition([row])
    task = audit["tasks"]["casual_conversation"]

    assert task["history_contract_failure_rows"] == 1
    assert "fake_or_unproven_multi_turn" in task["failures"]


def test_v2_composition_audit_rejects_an_undeclared_plan_edge() -> None:
    row = _row(3)
    metadata = json.loads(row["source_representation"])
    metadata["composition"]["allowed_prompt_answer_edges"] = [
        ["prompt-1", "answer-1"]
    ]
    row["source_representation"] = json.dumps(metadata)

    audit = audit_v2_composition([row])
    task = audit["tasks"]["casual_conversation"]

    assert task["invalid_prompt_answer_edge_rows"] == 1
    assert "invalid_prompt_answer_compatibility" in task["failures"]
