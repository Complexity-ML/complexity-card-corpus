from __future__ import annotations

import pytest

from complexity_card_corpus.v2 import RoleSeparatedVariableBy, SurfaceRole
from complexity_card_corpus.variable_by import VariableBy2D


def _reservoir() -> RoleSeparatedVariableBy:
    return RoleSeparatedVariableBy(
        VariableBy2D(
            {
                "scenario": {
                    "subject": ("four apples", "three books"),
                    "result": ("4", "3"),
                },
                "prompt": {
                    "request": (
                        "How many items are there?",
                        "Give the item count.",
                    ),
                },
                "answer": {
                    "direct": (
                        "There are {scenario[result]} items.",
                        "The count is {scenario[result]}.",
                    ),
                },
            }
        )
    )


def test_v2_prompt_and_answer_surface_reservoirs_are_disjoint() -> None:
    reservoir = _reservoir()

    reservoir.validate(SurfaceRole.PROMPT, ("{prompt[request]}",))
    reservoir.validate(SurfaceRole.ANSWER, ("{answer[direct]}",))

    assert set(reservoir.matrix.cards("prompt", "request")).isdisjoint(
        reservoir.matrix.cards("answer", "direct")
    )


def test_v2_answer_cannot_consume_a_prompt_card_directly() -> None:
    with pytest.raises(ValueError, match="cross the prompt/answer boundary"):
        _reservoir().validate(SurfaceRole.ANSWER, ("{prompt[request]}",))


def test_v2_answer_cannot_consume_a_prompt_card_through_nested_variable() -> None:
    reservoir = RoleSeparatedVariableBy(
        VariableBy2D(
            {
                "scenario": {"result": ("4",)},
                "prompt": {"request": ("What is two plus two?",)},
                "answer": {"direct": ("{prompt[request]} It is {scenario[result]}.",)},
            }
        )
    )

    with pytest.raises(ValueError, match="cross the prompt/answer boundary"):
        reservoir.validate(SurfaceRole.ANSWER, ("{answer[direct]}",))


def test_v2_release_contract_has_no_synthetic_training_example_cap() -> None:
    from complexity_card_corpus.v2 import audit_v2_behavior

    audit = audit_v2_behavior(
        [],
        thresholds={
            "minimum_direct_casual_examples": 0,
            "minimum_direct_casual_share": 0.0,
            "minimum_short_direct_casual_share": 0.0,
        },
    )

    assert audit["passed"] is False
    assert not any(
        violation.startswith("expected ") and "train examples" in violation
        for violation in audit["violations"]
    )
    assert audit["thresholds"]["required_train_examples"] is None
