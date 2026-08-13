from __future__ import annotations

import json

from complexity_card_corpus.v2 import audit_v2_integrity, render_think_final


def _row(*, prompt: str = "What is 3 + 4?", final: str = "The answer is 7.") -> dict:
    thinking = "Add the two integers. A reverse subtraction confirms the value."
    assistant = render_think_final(thinking, final)
    return {
        "example_id": "reasoning-1",
        "task": "reasoning_verification",
        "split": "train",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "prompt": prompt,
        "response": assistant,
        "reasoning_trace": thinking,
        "final_response": final,
        "source_representation": json.dumps(
            {
                "facts": {
                    "operation": "addition",
                    "kind": "calculate",
                    "left": "3",
                    "right": "4",
                    "result": "7",
                    "candidate": "7",
                },
                "validator": {"kind": "arithmetic"},
            }
        ),
        "source": "authored V2 test",
        "license": "CC BY-NC 4.0",
        "version": "2.0.0",
    }


def test_v2_integrity_accepts_a_consistent_reasoning_row() -> None:
    audit = audit_v2_integrity([_row()])

    assert audit["passed"] is True
    assert audit["arithmetic_error_count"] == 0
    assert audit["envelope_error_count"] == 0


def test_v2_integrity_accepts_alternating_assistant_history() -> None:
    row = _row()
    row["messages"] = [
        {"role": "user", "content": "Remember that the first value is 3."},
        {"role": "assistant", "content": "I will keep 3 in context."},
        *row["messages"],
    ]

    audit = audit_v2_integrity([row])

    assert audit["passed"] is True
    assert audit["envelope_error_count"] == 0


def test_v2_integrity_rejects_non_alternating_history() -> None:
    row = _row()
    row["messages"].insert(1, {"role": "user", "content": "Another user turn."})

    audit = audit_v2_integrity([row])

    assert audit["passed"] is False
    assert audit["envelope_error_count"] == 1


def test_v2_integrity_rejects_wrong_arithmetic_even_when_text_looks_clean() -> None:
    row = _row(final="The answer is 8.")

    audit = audit_v2_integrity([row])

    assert audit["passed"] is False
    assert audit["arithmetic_error_count"] == 1


def test_v2_integrity_rejects_malformed_think_final_envelope() -> None:
    row = _row()
    row["messages"][-1]["content"] = "<think>unfinished"
    row["response"] = "<think>unfinished"

    audit = audit_v2_integrity([row])

    assert audit["envelope_error_count"] == 1


def test_v2_integrity_rejects_conflicting_answers_for_identical_prompt() -> None:
    first = _row()
    second = _row(final="The answer is 9.")
    second["example_id"] = "reasoning-2"

    audit = audit_v2_integrity([first, second])

    assert audit["conflicting_prompt_count"] == 1


def test_v2_integrity_distinguishes_identical_followups_by_history() -> None:
    first = _row(prompt="And the total?", final="The answer is 7.")
    first["messages"] = [
        {"role": "user", "content": "Keep 3 and 4 in mind."},
        {"role": "assistant", "content": "I have both values."},
        {"role": "user", "content": "And the total?"},
        first["messages"][-1],
    ]
    second = _row(prompt="And the total?", final="The answer is 9.")
    second["example_id"] = "reasoning-2"
    second["messages"] = [
        {"role": "user", "content": "Keep 4 and 5 in mind."},
        {"role": "assistant", "content": "I have both values."},
        {"role": "user", "content": "And the total?"},
        second["messages"][-1],
    ]

    audit = audit_v2_integrity([first, second])

    assert audit["conflicting_prompt_count"] == 0


def test_v2_integrity_rejects_rendering_artifacts_and_placeholders() -> None:
    row = _row(prompt="Calculate {scenario[value]}..")

    audit = audit_v2_integrity([row])

    assert audit["placeholder_error_count"] == 1
    assert audit["rendering_error_counts"]["double_terminal_period"] == 1


def test_v2_integrity_rejects_an_unverifiable_natural_answer() -> None:
    row = _row()
    metadata = json.loads(row["source_representation"])
    metadata.pop("validator")
    row["source_representation"] = json.dumps(metadata)

    audit = audit_v2_integrity([row])

    assert audit["validator_error_count"] == 1


def test_v2_integrity_detects_known_template_grammar_failures() -> None:
    row = _row(prompt="The same a backpack is used. First, Reduce the risk.")

    audit = audit_v2_integrity([row])

    assert audit["rendering_error_counts"]["same_plus_article"] == 1
    assert audit["rendering_error_counts"]["capitalized_verb_after_connector"] == 1
