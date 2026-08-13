from __future__ import annotations

from ...variable_by import VariableBy2D
from ..contracts import RoleSeparatedVariableBy, SurfaceRole
from ..decks import V2RoleSeparatedDeck, V2SubcardPool, prompt_variant_plans
from ._axes import PEOPLE
from ._common import render_v2_row, validate_complete_rows


TASK = "critique_revision"
_CASES = (
    ("operations", "maintenance note", "confirm the affected equipment", "Tuesday noon", "the evening team needs an accurate handoff"),
    ("operations", "room schedule", "remove the duplicate booking", "Friday morning", "both workshop leaders need a confirmed room"),
    ("logistics", "delivery manifest", "correct the destination code", "the next dispatch", "the parcel must reach the intended depot"),
    ("logistics", "inventory count", "reconcile the missing units", "today's close", "the replenishment order uses the verified balance"),
    ("training", "onboarding guide", "replace the outdated login steps", "the next session", "new starters otherwise receive unusable instructions"),
    ("training", "workshop agenda", "add the practice interval", "Wednesday", "participants need time to apply the demonstration"),
    ("research", "methods appendix", "name the calibration procedure", "the review meeting", "another researcher must be able to repeat the measurement"),
    ("research", "survey summary", "separate responses from interpretations", "Monday afternoon", "the evidence should remain distinguishable from the analysis"),
    ("finance", "expense record", "attach the missing receipt", "month-end review", "the reimbursement needs documented support"),
    ("finance", "budget forecast", "update the supplier estimate", "Thursday", "the current total uses an expired price"),
    ("software", "deployment note", "state the rollback condition", "release approval", "operators need a clear trigger for restoring the prior version"),
    ("software", "issue report", "include the minimal reproduction steps", "triage", "the maintainer must be able to observe the failure"),
    ("community", "event notice", "clarify the accessible entrance", "publication", "visitors need the route before arriving"),
    ("community", "volunteer message", "specify the check-in location", "the morning shift", "volunteers are currently arriving at different doors"),
    ("communications", "service alert", "replace the vague outage description", "immediate release", "readers need to know which service is affected"),
    ("communications", "correction notice", "identify the original error", "today", "the public should understand what changed"),
)
_STYLES = ("formal", "concise", "friendly", "plain")
_STYLE_GUIDANCE = {
    "formal": "use professional register, an explicit request, and no conversational filler",
    "concise": "remove every nonessential phrase while retaining action, timing, and reason",
    "friendly": "use a warm greeting and a polite request without weakening the deadline",
    "plain": "use common words and state the actor, action, timing, and reason directly",
}
_PROMPTS = (
    "Name the main clarity problem, then rewrite the draft in exactly two sentences using a {scenario[style]} tone; {scenario[style_guidance]}. Draft: {scenario[draft]}",
    "Revise this message into exactly two {scenario[style]} sentences; {scenario[style_guidance]}. Preserve the facts in this draft: {scenario[draft]}",
    "The draft is wordy and indirect. Produce an exact two-sentence {scenario[style]} replacement and {scenario[style_guidance]}. Draft: {scenario[draft]}",
    "Rewrite the following as two clear sentences in a {scenario[style]} style. Specifically, {scenario[style_guidance]}. Source: {scenario[draft]}",
)
_PROMPT_FUNCTIONS = (
    ("diagnose_clarity", "request_exact_rewrite", "specify_style"),
    ("request_exact_rewrite", "specify_style", "preserve_facts"),
    ("diagnose_indirectness", "request_exact_rewrite", "specify_style"),
    ("request_clear_rewrite", "specify_style", "supply_source"),
)


def _target(
    style: str,
    person: str,
    artifact: str,
    action: str,
    deadline: str,
    reason: str,
) -> str:
    reason_cap = reason[0].upper() + reason[1:]
    if style == "formal":
        return f"{person}, please {action} in the {artifact} by {deadline}. This is required because {reason}."
    if style == "concise":
        return f"Please {action} in the {artifact} by {deadline}. {reason_cap}."
    if style == "friendly":
        return f"Hi {person}, could you {action} in the {artifact} by {deadline}? That will help because {reason}."
    if style == "plain":
        return f"{person} needs to {action} in the {artifact} by {deadline}. The reason is that {reason}."
    raise ValueError(style)


def critique_revision_capacity() -> int:
    return len(_CASES) * len(_STYLES)


def render_critique_revision_rows() -> list[dict[str, object]]:
    rows = []
    for case_index, (domain, artifact, action, deadline, reason) in enumerate(_CASES):
        person = PEOPLE[case_index % len(PEOPLE)]
        draft = (
            f"So, {person}, about the {artifact}, I was thinking maybe you could "
            f"{action} sometime before {deadline}, mostly since {reason}, if possible."
        )
        for style in _STYLES:
            style_guidance = _STYLE_GUIDANCE[style]
            target = _target(style, person, artifact, action, deadline, reason)
            target_template = target.replace("{", "{{").replace("}", "}}")
            variables = RoleSeparatedVariableBy(
                VariableBy2D(
                    {
                        "scenario": {
                            "style": (style,), "draft": (draft,),
                            "style_guidance": (style_guidance,),
                            "person": (person,), "artifact": (artifact,),
                            "action": (action,), "deadline": (deadline,),
                            "reason": (reason,),
                        },
                        "prompt": {"revision_request": _PROMPTS},
                        "answer": {"rewrite": (target_template,)},
                    }
                )
            )
            deck = V2RoleSeparatedDeck(
                name=f"{TASK}:{domain}:{artifact}:{style}", variables=variables,
                prompt_pools=(V2SubcardPool("revision_request", SurfaceRole.PROMPT, ("{prompt[revision_request]}",)),),
                answer_pools=(V2SubcardPool("rewrite", SurfaceRole.ANSWER, ("{answer[rewrite]}",)),),
                prompt_plans=prompt_variant_plans(
                    sense="revision_request",
                    pool_name="revision_request",
                    functions=_PROMPT_FUNCTIONS,
                ),
            )
            case_id = f"{domain}:{artifact}:{style}"
            rows.append(
                render_v2_row(
                    task=TASK, case_id=case_id, domain=domain, difficulty="medium",
                    deck=deck,
                    facts={"style": style, "draft": draft, "target": target},
                    validator={"kind": "exact", "expected": target},
                )
            )
    return validate_complete_rows(TASK, rows, critique_revision_capacity())


__all__ = ("critique_revision_capacity", "render_critique_revision_rows")
