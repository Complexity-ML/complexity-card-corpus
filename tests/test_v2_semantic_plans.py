from __future__ import annotations

import json

from complexity_card_corpus.v2 import (
    AnswerPlan,
    ConversationTurn,
    PlanCompatibility,
    PromptPlan,
    RoleSeparatedVariableBy,
    SemanticFrame,
    SurfaceRole,
    ThinkingBudget,
    ThinkingPlan,
    V2RoleSeparatedDeck,
    V2SubcardPool,
)
from complexity_card_corpus.v2.families._common import render_v2_row
from complexity_card_corpus.variable_by import VariableBy2D


def _planned_deck() -> V2RoleSeparatedDeck:
    variables = RoleSeparatedVariableBy(
        VariableBy2D(
            {
                "scenario": {"result": ("four",)},
                "prompt": {
                    "context": ("I am checking a small calculation.",),
                    "request": ("Could you solve it now?",),
                },
                "thinking": {
                    "derive": ("Add both operands carefully.",),
                    "verify": ("Reverse the operation as a check.",),
                },
                "answer": {
                    "direct": ("The computed result is {scenario[result]}.",),
                    "check": ("A reverse check reaches the same value.",),
                },
            }
        )
    )
    return V2RoleSeparatedDeck(
        name="planned-arithmetic",
        variables=variables,
        prompt_pools=(
            V2SubcardPool("context", SurfaceRole.PROMPT, ("{prompt[context]}",)),
            V2SubcardPool("request", SurfaceRole.PROMPT, ("{prompt[request]}",)),
        ),
        thinking_pools=(
            V2SubcardPool("derive", SurfaceRole.THINKING, ("{thinking[derive]}",)),
            V2SubcardPool("verify", SurfaceRole.THINKING, ("{thinking[verify]}",)),
        ),
        answer_pools=(
            V2SubcardPool("direct", SurfaceRole.ANSWER, ("{answer[direct]}",)),
            V2SubcardPool("check", SurfaceRole.ANSWER, ("{answer[check]}",)),
        ),
        prompt_plans=(
            PromptPlan("brief", ("request",), ("request",), "brief"),
            PromptPlan(
                "contextual",
                ("context", "request"),
                ("context", "request"),
                "neutral",
            ),
        ),
        answer_plans=(
            AnswerPlan("direct", ("direct",), ("answer",)),
            AnswerPlan(
                "verified",
                ("direct", "check"),
                ("answer", "verification"),
            ),
        ),
        thinking_plans=(
            ThinkingPlan("none", (), ("direct",), ThinkingBudget.NONE),
            ThinkingPlan(
                "short",
                ("derive",),
                ("derivation",),
                ThinkingBudget.SHORT,
            ),
            ThinkingPlan(
                "verified",
                ("derive", "verify"),
                ("derivation", "verification"),
                ThinkingBudget.VERIFICATION,
            ),
        ),
        compatibility=PlanCompatibility(
            prompt_to_answers={
                "brief": ("direct", "verified"),
                "contextual": ("direct", "verified"),
            },
            answer_to_thinking={
                "direct": ("none", "short"),
                "verified": ("verified",),
            },
        ),
    )


def test_v2_selects_prompt_answer_and_thinking_plans_independently() -> None:
    deals = [_planned_deck().deal(f"case-{index}") for index in range(100)]

    assert {deal.prompt_plan for deal in deals} == {"brief", "contextual"}
    assert {deal.answer_plan for deal in deals} == {"direct", "verified"}
    assert len({(deal.prompt_plan, deal.answer_plan) for deal in deals}) == 4
    assert all(
        deal.thinking_plan == "verified"
        for deal in deals
        if deal.answer_plan == "verified"
    )
    assert all(
        deal.thinking_plan in {"none", "short"}
        for deal in deals
        if deal.answer_plan == "direct"
    )
    assert all(
        (deal.prompt_plan, deal.answer_plan) in deal.allowed_prompt_answer_edges
        for deal in deals
    )
    assert all(
        (deal.answer_plan, deal.thinking_plan) in deal.allowed_answer_thinking_edges
        for deal in deals
    )
    assert {edge for deal in deals for edge in deal.allowed_prompt_answer_edges} == {
        ("brief", "direct"),
        ("brief", "verified"),
        ("contextual", "direct"),
        ("contextual", "verified"),
    }


def test_v2_semantic_frame_adds_real_history_without_exposing_metadata() -> None:
    frame = SemanticFrame(
        intent="arithmetic_follow_up",
        facts={"result": 4, "operation": "calculation"},
        constraints=("answer directly",),
        expected_outcome=4,
        user_tone="brief",
        history=(
            ConversationTurn("user", "Can you help with a calculation?"),
            ConversationTurn("assistant", "Yes. What should I calculate?"),
        ),
        history_required_facts=("operation",),
    )

    row = render_v2_row(
        task="casual_conversation",
        case_id="multi-turn-four",
        domain="addition",
        difficulty="easy",
        deck=_planned_deck(),
        facts={"result": 4, "operation": "calculation"},
        validator={"kind": "contains", "required": ["four"]},
        semantic_frame=frame,
    )

    assert [message["role"] for message in row["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert "SemanticFrame" not in str(row["messages"])
    metadata = json.loads(str(row["source_representation"]))
    assert metadata["semantic_frame"]["intent"] == "arithmetic_follow_up"
    assert metadata["semantic_frame"]["history_required_facts"] == ["operation"]
    assert metadata["composition"]["answer_plan"] in {"direct", "verified"}
    assert metadata["composition"]["allowed_prompt_answer_edges"]
    assert metadata["composition"]["allowed_answer_thinking_edges"]
    assert metadata["thinking_budget"] in {
        ThinkingBudget.NONE,
        ThinkingBudget.SHORT,
        ThinkingBudget.VERIFICATION,
    }


def test_v2_semantic_frame_rejects_disposable_multi_turn_history() -> None:
    try:
        SemanticFrame(
            intent="follow_up",
            facts={"result": 4},
            history=(
                ConversationTurn("user", "Hello."),
                ConversationTurn("assistant", "Hello."),
            ),
        )
    except ValueError as error:
        assert "facts required from history" in str(error)
    else:
        raise AssertionError("multi-turn history without a required fact was accepted")
