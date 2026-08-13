from __future__ import annotations

import pytest

from complexity_card_corpus.v2 import (
    ALL_TASKS,
    CORE_TASKS,
    RoleSeparatedVariableBy,
    SurfaceRole,
    V2RoleSeparatedDeck,
    V2SubcardPool,
)
from complexity_card_corpus.variable_by import VariableBy2D


def _variables() -> RoleSeparatedVariableBy:
    return RoleSeparatedVariableBy(
        VariableBy2D(
            {
                "scenario": {"result": ("4",)},
                "prompt": {"request": ("Calculate the value.",)},
                "answer": {"direct": ("The result is {scenario[result]}.",)},
            }
        )
    )


def test_v2_plan_has_all_families_without_an_example_cap() -> None:
    assert ALL_TASKS == ("casual_conversation", *CORE_TASKS)
    assert len(ALL_TASKS) == 15


def test_v2_deck_rejects_prompt_wording_inside_answer_subcards() -> None:
    with pytest.raises(ValueError, match="cross the prompt/answer boundary"):
        V2RoleSeparatedDeck(
            name="leaky",
            variables=_variables(),
            prompt_pools=(
                V2SubcardPool("request", SurfaceRole.PROMPT, ("{prompt[request]}",)),
            ),
            answer_pools=(
                V2SubcardPool("answer", SurfaceRole.ANSWER, ("{prompt[request]}",)),
            ),
        )


def test_v2_deck_rejects_literal_prompt_answer_phrase_reuse() -> None:
    with pytest.raises(ValueError, match="share literal trigrams"):
        V2RoleSeparatedDeck(
            name="lexically-leaky",
            variables=_variables(),
            prompt_pools=(
                V2SubcardPool(
                    "request",
                    SurfaceRole.PROMPT,
                    ("Please calculate the requested value.",),
                ),
            ),
            answer_pools=(
                V2SubcardPool(
                    "answer",
                    SurfaceRole.ANSWER,
                    ("The requested value is {scenario[result]}.",),
                ),
            ),
        )


def test_v2_deck_checks_real_variable_reservoir_surfaces() -> None:
    variables = RoleSeparatedVariableBy(
        VariableBy2D(
            {
                "scenario": {"result": ("4",)},
                "prompt": {"request": ("Please calculate the requested value.",)},
                "answer": {"direct": ("The requested value is {scenario[result]}.",)},
            }
        )
    )

    with pytest.raises(ValueError, match="share literal trigrams"):
        V2RoleSeparatedDeck(
            name="reservoir-leak",
            variables=variables,
            prompt_pools=(
                V2SubcardPool("request", SurfaceRole.PROMPT, ("{prompt[request]}",)),
            ),
            answer_pools=(
                V2SubcardPool("answer", SurfaceRole.ANSWER, ("{answer[direct]}",)),
            ),
        )


def test_v2_deck_deals_deterministically_with_role_provenance() -> None:
    deck = V2RoleSeparatedDeck(
        name="clean",
        variables=_variables(),
        prompt_pools=(
            V2SubcardPool("request", SurfaceRole.PROMPT, ("{prompt[request]}",)),
        ),
        answer_pools=(
            V2SubcardPool("answer", SurfaceRole.ANSWER, ("{answer[direct]}",)),
        ),
    )

    first = deck.deal("case-1")
    second = deck.deal("case-1")

    assert first == second
    assert first.prompt == "Calculate the value."
    assert first.thinking == ""
    assert first.answer == "The result is 4."
    assert first.prompt_subcards == ("Calculate the value.",)
    assert first.thinking_subcards == ()
    assert first.answer_subcards == ("The result is 4.",)
    assert first.variable_indices["scenario"]["result"] == 0
    assert first.variable_card_counts["answer"]["direct"] == 1
