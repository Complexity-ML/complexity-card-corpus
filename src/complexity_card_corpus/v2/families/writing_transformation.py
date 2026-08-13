from __future__ import annotations

from ...variable_by import VariableBy2D
from ..contracts import RoleSeparatedVariableBy, SurfaceRole
from ..decks import V2RoleSeparatedDeck, V2SubcardPool
from ._axes import PEOPLE
from ._common import render_v2_row, validate_complete_rows


TASK = "writing_transformation"
_CASES = (
    ("operations", "maintenance log", "replace the loose guard", "Tuesday noon", "the fan can be tested safely"),
    ("operations", "room schedule", "resolve the duplicate booking", "Friday morning", "both workshops have a confirmed location"),
    ("logistics", "delivery record", "correct the depot code", "the next dispatch", "the parcel follows the verified route"),
    ("logistics", "stock report", "reconcile the missing filters", "today's close", "the order uses an accurate balance"),
    ("training", "login guide", "replace the outdated steps", "the next session", "new staff can activate their accounts"),
    ("training", "workshop plan", "add a practice interval", "Wednesday", "participants can apply the demonstration"),
    ("research", "methods note", "name the calibration procedure", "the review meeting", "another researcher can repeat the measurement"),
    ("research", "survey report", "separate quotations from interpretations", "Monday afternoon", "readers can trace conclusions to evidence"),
    ("finance", "expense claim", "attach the readable receipt", "month-end review", "the reimbursement has documented support"),
    ("finance", "budget sheet", "update the supplier estimate", "Thursday", "the total reflects the current price"),
    ("software", "release note", "state the rollback condition", "approval", "operators know when to restore the prior version"),
    ("software", "issue ticket", "include the reproduction steps", "triage", "the maintainer can observe the failure"),
    ("community", "event notice", "clarify the accessible entrance", "publication", "visitors know the route before arriving"),
    ("community", "volunteer reminder", "specify the check-in location", "the morning shift", "everyone arrives at the same door"),
    ("communications", "service alert", "identify the affected service", "immediate release", "readers understand the impact"),
    ("communications", "correction notice", "name the original error", "today", "the public can see what changed"),
)
_TRANSFORMS = {
    "formal": "use professional register and an explicit request without filler",
    "concise": "retain only the actor, required action, timing, and reason",
    "friendly": "use a warm greeting and a polite but unambiguous request",
    "plain_language": "use common vocabulary and short direct sentences",
}
_PROMPTS = (
    "Transform the source into {scenario[transform]} writing; {scenario[guidance]}. Source: {scenario[source]}",
    "Rewrite this message in a {scenario[transform]} style. Specifically, {scenario[guidance]}. Text: {scenario[source]}",
    "Preserve the facts while changing the expression to {scenario[transform]}; {scenario[guidance]}. Original: {scenario[source]}",
    "Produce a clean {scenario[transform]} version of the following. The transformation must {scenario[guidance]}. {scenario[source]}",
)


def _target(transform: str, person: str, artifact: str, action: str, deadline: str, reason: str) -> str:
    reason_cap = reason[0].upper() + reason[1:]
    if transform == "formal":
        return f"{person}, please {action} in the {artifact} by {deadline}. This is necessary so that {reason}."
    if transform == "concise":
        return f"{person}: {action.capitalize()} in the {artifact} by {deadline}. {reason_cap}."
    if transform == "friendly":
        return f"Hi {person}, could you {action} in the {artifact} by {deadline}? That will ensure {reason}."
    if transform == "plain_language":
        return f"{person} must {action} in the {artifact} by {deadline}. This will mean {reason}."
    raise ValueError(transform)


def writing_transformation_capacity() -> int:
    return len(_CASES) * len(_TRANSFORMS)


def render_writing_transformation_rows() -> list[dict[str, object]]:
    rows = []
    for case_index, (domain, artifact, action, deadline, reason) in enumerate(_CASES):
        person = PEOPLE[case_index % len(PEOPLE)]
        source = (
            f"Hey {person}, just a quick thing about the {artifact}: maybe {action} "
            f"at some point before {deadline}, because otherwise we may not be sure "
            f"that {reason}, thanks."
        )
        for transform, guidance in _TRANSFORMS.items():
            target = _target(transform, person, artifact, action, deadline, reason)
            variables = RoleSeparatedVariableBy(
                VariableBy2D(
                    {
                        "scenario": {
                            "transform": (transform,), "guidance": (guidance,),
                            "source": (source,), "person": (person,),
                            "artifact": (artifact,), "action": (action,),
                            "deadline": (deadline,), "reason": (reason,),
                        },
                        "prompt": {"transform_request": _PROMPTS},
                        "answer": {"transformed_text": (target,)},
                    }
                )
            )
            deck = V2RoleSeparatedDeck(
                name=f"{TASK}:{domain}:{artifact}:{transform}", variables=variables,
                prompt_pools=(V2SubcardPool("transform_request", SurfaceRole.PROMPT, ("{prompt[transform_request]}",)),),
                answer_pools=(V2SubcardPool("transformed_text", SurfaceRole.ANSWER, ("{answer[transformed_text]}",)),),
            )
            rows.append(
                render_v2_row(
                    task=TASK, case_id=f"{domain}:{artifact}:{transform}", domain=domain,
                    difficulty="easy", deck=deck,
                    facts={"transform": transform, "source": source, "target": target},
                    validator={"kind": "exact", "expected": target},
                )
            )
    return validate_complete_rows(TASK, rows, writing_transformation_capacity())


__all__ = ("render_writing_transformation_rows", "writing_transformation_capacity")
