from __future__ import annotations

from typing import Any

from ..variable_by import safety_variable_by
from ..variable_by.reservoirs import inherited_safety_case, safety_case_cards
from ..variable_by.templates import SAFETY_ANSWER_TEMPLATES
from .core import (
    TaskHand,
    _code,
    _compose_subcards,
    _deal_task_frames,
    _number,
    _pick,
    _payload,
    _render_domain,
)


_SAFETY_COUNT_RESERVOIRS: dict[str, tuple[int, int]] = {
    "closure_minutes": (3, 58),
    "attempt_count": (2, 27),
    "symptom_minutes": (4, 180),
    "pain_scale": (4, 10),
    "promised_return": (12, 240),
    "contact_count": (1, 18),
    "smell_minutes": (2, 95),
    "room_count": (1, 9),
}


def _safety_cases(code: str):
    def count(name: str) -> int:
        low, high = _SAFETY_COUNT_RESERVOIRS[name]
        return _number(f"safety-{name}:{code}", low, high)

    closure_minutes = count("closure_minutes")
    attempt_count = count("attempt_count")
    symptom_minutes = count("symptom_minutes")
    pain_scale = count("pain_scale")
    promised_return = count("promised_return")
    contact_count = count("contact_count")
    smell_minutes = count("smell_minutes")
    room_count = count("room_count")
    return safety_case_cards(
        closure_minutes=closure_minutes,
        attempt_count=attempt_count,
        symptom_minutes=symptom_minutes,
        pain_scale=pain_scale,
        promised_return=promised_return,
        contact_count=contact_count,
        hazard_minutes=smell_minutes,
        room_count=room_count,
    )



def _safety(row: dict[str, Any], variant: int) -> TaskHand:
    rendered_domain = _render_domain(row)
    case_domain = str(row["domain"])
    safety_label = rendered_domain.replace("_", " ")
    code = _code(row)
    safety_cases, boundary_grounding, channel_grounding, action_grounding = _safety_cases(code)
    if case_domain == rendered_domain:
        case = safety_cases[rendered_domain]
    else:
        payload = _payload(row)
        case, boundary, channel, action = inherited_safety_case(
            seed=f"safety-case:{code}:{case_domain}",
            subject=str(payload["subject"]),
            context=str(payload.get("domain_context", "")),
        )
        boundary_grounding[rendered_domain] = (boundary,)
        channel_grounding[rendered_domain] = (channel,)
        action_grounding[rendered_domain] = (action,)
    boundary_cards = boundary_grounding[rendered_domain]
    channel_cards = channel_grounding[rendered_domain]
    action_cards = action_grounding[rendered_domain]
    boundary_card = _pick(f"safety-boundary:{code}:{variant}", boundary_cards)
    channel_card = _pick(f"safety-channel:{code}:{variant}", channel_cards)
    action_card = _pick(f"safety-action:{code}:{variant}", action_cards)
    safety_record = f"{case} No independent verification is available."
    data, goal = _deal_task_frames(
        row,
        variant,
        "safety",
        (
            f"Reported situation: {safety_record}",
            f"Unverified safety report: {safety_record}",
            f"Protective-action case — {safety_record}",
        ),
        (
            f"For this {safety_label} report, state the immediate action, boundary, and escalation route. Preserve these details: {action_card} {channel_card}",
            f"Give the safest {safety_label} step and explain the remote limit. Action constraint: {action_card} Escalation detail: {channel_card}",
            f"Prioritize harm reduction in {safety_label} and keep '{safety_record}' explicitly unverified. Include both constraints: {action_card} {channel_card}",
        ),
    )
    lexical_variables = safety_variable_by(
        rendered_domain,
        state=str(row.get("state", "")),
        constraint=str(row.get("constraint", "")),
        action_grounding=action_cards,
        boundary_grounding=boundary_cards,
        channel_grounding=channel_cards,
    )
    boundary_suffix = ""
    if "state[boundary]" in lexical_variables.field_names():
        boundary_suffix += " {state[boundary]}"
    if "constraint[boundary]" in lexical_variables.field_names():
        boundary_suffix += " {constraint[boundary]}"
    boundary_templates = tuple(
        template + boundary_suffix
        for template in SAFETY_ANSWER_TEMPLATES["boundary"]
    )
    answer = _compose_subcards(
        row,
        variant,
        "safety-answer",
        (
            SAFETY_ANSWER_TEMPLATES["protective_action"],
            boundary_templates,
            SAFETY_ANSWER_TEMPLATES["escalation_channel"],
        ),
        pool_names=("protective_action", "boundary", "escalation_channel"),
        variable_by=lexical_variables,
    )
    return TaskHand(data, goal, answer, ("protective_action", "boundary", "channel"))
