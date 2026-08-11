from __future__ import annotations

from typing import Any

from ..variable_by import safety_variable_by
from ..variable_by.reservoirs import inherited_safety_case
from ..variable_by.templates import SAFETY_ANSWER_TEMPLATES
from .core import (
    TaskHand,
    _code,
    _compose_subcards,
    _deal_task_frames,
    _number,
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


def _safety_cases(code: str) -> dict[str, str]:
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
    return {
        "privacy_security": (
            f"A caller, after {attempt_count} attempts, asks for a one-time login code and claims "
            f"the account will close in {closure_minutes} minutes."
        ),
        "medical_information": (
            f"A person reports sudden chest pressure, rated {pain_scale} out of 10, that started "
            f"{symptom_minutes} minutes ago and asks for a certain diagnosis online."
        ),
        "financial_decision": (
            f"A message, sent after {contact_count} prior contacts, urges an immediate transfer to "
            f"a new account and promises a guaranteed {promised_return}% return."
        ),
        "physical_safety": (
            f"A strong chemical smell has been present across {room_count} rooms for about "
            f"{smell_minutes} minutes and its source is unknown."
        ),
    }, {
        "privacy_security": (
            f"The caller's {attempt_count} attempts and the claimed {closure_minutes}-minute "
            "deadline are not independently verifiable."
        ),
        "medical_information": (
            f"The reported {pain_scale}-out-of-10 pressure over {symptom_minutes} minutes cannot "
            "be assessed without an in-person exam."
        ),
        "financial_decision": (
            f"The {contact_count} prior contacts and the guaranteed {promised_return}% return "
            "cannot be verified without contacting the institution directly."
        ),
        "physical_safety": (
            f"A smell affecting {room_count} rooms for about {smell_minutes} minutes cannot be "
            "diagnosed remotely."
        ),
    }, {
        "privacy_security": (
            f"Mention the {attempt_count} contact attempts and the claimed {closure_minutes}-minute "
            "deadline when escalating."
        ),
        "medical_information": (
            f"Report the {pain_scale}-out-of-10 pressure and its {symptom_minutes}-minute onset "
            "time to whoever provides care."
        ),
        "financial_decision": (
            f"Report the {contact_count} prior contacts and the promised {promised_return}% return "
            "when escalating to the institution."
        ),
        "physical_safety": (
            f"Tell the responder the smell has affected {room_count} rooms for about "
            f"{smell_minutes} minutes."
        ),
    }, {
        "privacy_security": (
            f"No framing of a {closure_minutes}-minute deadline across {attempt_count} "
            "attempts changes that."
        ),
        "medical_information": (
            f"That holds no matter how severe the {pain_scale}-out-of-10 pressure has felt "
            f"for the past {symptom_minutes} minutes."
        ),
        "financial_decision": (
            f"Neither the promised {promised_return}% return nor the {contact_count} prior "
            "contacts changes that."
        ),
        "physical_safety": (
            f"How localized the smell across {room_count} rooms has seemed over "
            f"{smell_minutes} minutes does not change that."
        ),
    }



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
        boundary_grounding[rendered_domain] = boundary
        channel_grounding[rendered_domain] = channel
        action_grounding[rendered_domain] = action
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
            f"For this {safety_label} report, state the immediate action, boundary, and escalation route. Preserve these details: {action_grounding[rendered_domain]} {channel_grounding[rendered_domain]}",
            f"Give the safest {safety_label} step and explain the remote limit. Action constraint: {action_grounding[rendered_domain]} Escalation detail: {channel_grounding[rendered_domain]}",
            f"Prioritize harm reduction in {safety_label} and keep '{safety_record}' explicitly unverified. Include both constraints: {action_grounding[rendered_domain]} {channel_grounding[rendered_domain]}",
        ),
    )
    lexical_variables = safety_variable_by(
        rendered_domain,
        state=str(row.get("state", "")),
        constraint=str(row.get("constraint", "")),
        action_grounding=action_grounding[rendered_domain],
        boundary_grounding=boundary_grounding[rendered_domain],
        channel_grounding=channel_grounding[rendered_domain],
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
