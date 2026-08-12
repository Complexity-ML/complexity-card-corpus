from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from .action import _planning, _practical, _troubleshooting
from .communication import _clarification, _empathy, _explanation, _writing
from .core import TaskHand, _code, _compose_subcards
from .extraction import render_extraction
from .intent_contracts import intent_contract_for
from .knowledge import _grounded_qa, _summary
from .reasoning import _brainstorm, _critique, _reasoning
from .safety import _safety

_RENDERERS = {
    "practical_action": _practical,
    "explanation_learning": _explanation,
    "troubleshooting": _troubleshooting,
    "writing_transformation": _writing,
    "planning_comparison": _planning,
    "conversation_empathy": _empathy,
    "safety_uncertainty": _safety,
    "grounded_qa": _grounded_qa,
    "summarization_synthesis": _summary,
    "extraction_classification": render_extraction,
    "reasoning_verification": _reasoning,
    "critique_revision": _critique,
    "brainstorming_creativity": _brainstorm,
    "context_clarification": _clarification,
}


def deal_task_hand(row: dict[str, Any], variant: int) -> TaskHand:
    try:
        hand = _RENDERERS[row["family"]](row, variant)
    except KeyError as error:
        raise ValueError(f"no card renderer for {row['family']}") from error
    if row["family"] != "extraction_classification":
        code = _code(row)
        answer = _compose_subcards(
            row,
            variant,
            f"{row['family']}-surface",
            (
                (
                    f"Hand {code} —",
                    f"For hand {code}:",
                    f"Hand {code} -",
                ),
                (hand.answer,),
            ),
            pool_names=("hand_reference", "family_answer"),
            cycle_first_pool=True,
        )
        hand = replace(
            hand,
            answer=answer,
            contract=intent_contract_for(row),
        )
    validate_task_hand(row["family"], hand)
    return hand


def validate_task_hand(family: str, hand: TaskHand) -> None:
    if not hand.data.strip() or not hand.goal.strip() or not hand.answer.strip():
        raise ValueError(f"empty task card in {family}")
    checks = {
        "practical_action": lambda: all(
            x in hand.answer for x in ("Next step:", "Owner:", "Timing:")
        ),
        "explanation_learning": lambda: (
            all(x in hand.answer for x in ("Core idea:", "Example:", "Check:"))
            and "?" in hand.answer
        ),
        "troubleshooting": lambda: (all(x in hand.answer for x in ("1.", "2.", "3."))),
        "writing_transformation": lambda: (len(hand.answer.split()) >= 12),
        "planning_comparison": lambda: any(
            topology.name == "planning-answer"
            and topology.pool_names
            == ("criteria", "choice", "sequence", "fallback")
            for topology in hand.answer.deck_topologies
        ),
        "conversation_empathy": lambda: hand.answer.count("?") <= 1,
        "safety_uncertainty": lambda: any(
            topology.name == "safety-answer"
            and topology.pool_names
            == ("protective_action", "boundary", "escalation_channel")
            for topology in hand.answer.deck_topologies
        ),
        "grounded_qa": lambda: ("unknown" in hand.answer.lower()),
        "summarization_synthesis": lambda: all(
            x in hand.answer for x in ("Decision:", "Action:", "Open point:")
        ),
        "extraction_classification": lambda: isinstance(json.loads(hand.answer), dict),
        "reasoning_verification": lambda: (
            all(x in hand.answer for x in ("Equation:", "Total:", "Check:"))
            and bool(re.search(r"\d", hand.answer))
        ),
        "critique_revision": lambda: all(
            x in hand.answer for x in ("Weakness:", "Revision:")
        ),
        "brainstorming_creativity": lambda: all(
            x in hand.answer for x in ("1.", "2.", "3.", "Select")
        ),
        "context_clarification": lambda: hand.answer.count("?") == 1,
    }
    if family not in checks or not checks[family]():
        raise ValueError(f"task hand does not fulfil the {family} contract")
