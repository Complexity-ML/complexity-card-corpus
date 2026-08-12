from __future__ import annotations

from .matrix import VariableBy2D
from .reservoirs import (
    casual_reservoir,
    critique_reservoir,
    empathy_reservoir,
    reasoning_reservoir,
    reasoning_envelope_reservoir,
    safety_reservoir,
)


def casual_variable_by(
    topic: dict,
    context: dict,
    intent: dict,
    arc: dict,
    decks: dict[str, list[str]],
) -> VariableBy2D:
    """Build nested variables for one semantic casual conversation unit."""

    return VariableBy2D(casual_reservoir(topic, context, intent, arc, decks))


_AUDIENCE_SENSE_BY_BRAINSTORM_DOMAIN = {
    "names": "community_member",
    "lesson_activity": "learner",
    "event_plan": "event_attendee",
    "feature_ideas": "software_user",
    "writing_prompts": "writer",
    "low_cost_activity": "activity_participant",
    "outreach": "outreach_audience",
    "workflow": "workflow_reviewer",
}

_COMMON_NOUNS_BY_SENSE = {
    "community_member": ("residents", "community members", "local participants"),
    "learner": ("learners", "students", "class participants"),
    "event_attendee": ("attendees", "guests", "event participants"),
    "software_user": ("users", "testers", "pilot users"),
    "writer": ("writers", "authors", "workshop participants"),
    "activity_participant": (
        "participants",
        "group members",
        "activity participants",
    ),
    "outreach_audience": ("people", "residents", "community members"),
    "workflow_reviewer": ("reviewers", "team members", "operators"),
}

_LINKERS_BY_SENSE = {
    "measurement": ("track", "measure", "compare"),
    "duration": (
        "within",
        "over",
        "across",
        "during",
        "throughout",
        "for",
        "in",
        "inside",
        "within the forthcoming",
        "over the upcoming",
        "across the following",
        "during the coming",
        "throughout the allotted",
        "inside an estimated",
        "over about",
        "during nearly",
        "within roughly",
        "within a scheduled",
        "across approximately",
        "over no more than",
        "for up to",
        "in a window capped at",
        "over a period lasting",
        "during a span covering",
    ),
}

_UNITS_BY_SENSE = {
    "trial_round": ("rounds", "test cycles", "pilot rounds"),
}

def brainstorming_variable_by(
    domain: str,
    *,
    scale: int,
    days: int,
    rounds: int,
    setting: str,
    signal: str,
) -> VariableBy2D:
    """Build the linked lexical and scenario matrix for one brainstorm hand."""

    try:
        audience_sense = _AUDIENCE_SENSE_BY_BRAINSTORM_DOMAIN[domain]
    except KeyError as error:
        raise ValueError(
            f"unsupported brainstorming variable_by domain: {domain}"
        ) from error
    return VariableBy2D(
        {
            "domain": {"label": (domain.replace("_", " "),)},
            "goal": {
                "generate": (
                    "Generate three meaningfully different {domain[label]} options.",
                    "Propose three distinct {domain[label]} approaches.",
                    "Create three feasible {domain[label]} alternatives.",
                ),
                "compare": (
                    "Test each {domain[label]} option against the brief.",
                    "Compare their {domain[label]} fit with the stated limits.",
                    "Check each {domain[label]} option against the named criteria.",
                ),
                "select": (
                    "Select the strongest {domain[label]} option.",
                    "Recommend one {domain[label]} option and explain the choice.",
                    "Choose the best bounded {domain[label]} proposal to test first.",
                ),
            },
            "constraint": {
                "explain": (
                    "Explain the choice using the stated limits.",
                    "Tie the recommendation to the supplied criteria.",
                    "Give the concrete reason for the final choice.",
                ),
            },
            "audience": {
                "common_noun": _COMMON_NOUNS_BY_SENSE[audience_sense],
            },
            "linker": _LINKERS_BY_SENSE,
            "unit": _UNITS_BY_SENSE,
            "measurement": {
                "signal": ("{scenario[signal]}",),
            },
            "scenario": {
                "scale": (str(scale),),
                "days": (str(days),),
                "rounds": (str(rounds),),
                "setting": (setting,),
                "signal": (signal,),
            },
        }
    )


def safety_variable_by(
    domain: str,
    *,
    state: str,
    constraint: str,
    action_grounding: str | tuple[str, ...],
    boundary_grounding: str | tuple[str, ...],
    channel_grounding: str | tuple[str, ...],
) -> VariableBy2D:
    """Return the localized semantic reservoir for one protective response."""

    return VariableBy2D(
        safety_reservoir(
            domain,
            state=state,
            constraint=constraint,
            action_grounding=action_grounding,
            boundary_grounding=boundary_grounding,
            channel_grounding=channel_grounding,
        )
    )


def empathy_variable_by(domain: str, *, state: str) -> VariableBy2D:
    """Return the localized semantic reservoir for one empathy response."""

    return VariableBy2D(empathy_reservoir(domain, state))


def reasoning_variable_by(
    *,
    equation: str,
    total: str | tuple[str, ...],
    check: str | tuple[str, ...],
    quantity_roles: tuple[str, ...],
    domain: str,
    code: str,
    data: str,
) -> VariableBy2D:
    """Return nested language cells for one verified calculation."""

    return VariableBy2D(
        reasoning_reservoir(
            equation=equation,
            total=total,
            check=check,
            quantity_roles=quantity_roles,
            domain=domain,
            code=code,
            data=data,
        )
    )


def reasoning_envelope_variable_by(
    task: str,
    *,
    analysis: str,
    analysis_inline: str,
    verification: str,
    verification_inline: str,
    final_variants: tuple[str, ...],
) -> VariableBy2D:
    """Return the V18 nested think/final matrix for one grounded answer."""

    return VariableBy2D(
        reasoning_envelope_reservoir(
            task,
            analysis=analysis,
            analysis_inline=analysis_inline,
            verification=verification,
            verification_inline=verification_inline,
            final_variants=final_variants,
        )
    )


def critique_variable_by(
    code: str,
    *,
    weakness: str | tuple[str, ...] | None = None,
    revision: str | tuple[str, ...] | None = None,
    consequences: tuple[str, ...] = (),
) -> VariableBy2D:
    """Return nested language cells for one critique instruction."""

    return VariableBy2D(
        critique_reservoir(
            code,
            weakness=weakness,
            revision=revision,
            consequences=consequences,
        )
    )
