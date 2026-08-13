from __future__ import annotations

from ...variable_by import VariableBy2D
from ..contracts import RoleSeparatedVariableBy, SurfaceRole
from ..decks import V2RoleSeparatedDeck, V2SubcardPool
from ._axes import SITES
from ._common import render_v2_row, validate_complete_rows
from .planning_comparison import _CONSTRAINTS


TASK = "practical_action"
_TASKS = (
    ("operations", "prepare a room for a public workshop", "confirm the capacity and accessible route", "place the required furniture and test the presentation equipment", "record the room as ready in the shared schedule"),
    ("operations", "handoff an unfinished maintenance job", "photograph the current state and isolate any hazard", "write the completed and remaining actions in the job record", "name the next owner and obtain their acknowledgment"),
    ("logistics", "receive a time-sensitive delivery", "verify the sender, seal, and expected contents", "record the arrival time before moving the package to its specified storage", "notify the documented recipient through the official channel"),
    ("logistics", "move supplies between two work areas", "count and label the items before movement", "use the approved route and preserve temperature or security requirements", "recount at destination and resolve any discrepancy"),
    ("training", "run a short onboarding session", "check participant access and the learning objective", "demonstrate the core task and let each participant practice it", "capture unanswered questions and provide the verified reference"),
    ("training", "update a procedural guide", "confirm the current process with its owner", "revise only the affected steps and mark the effective date", "ask a new user to test the instructions before publication"),
    ("research", "start a field observation", "confirm consent, location, and calibrated equipment", "record conditions before collecting the first measurement", "store the observation with its timestamp and method"),
    ("research", "correct a dataset entry", "preserve the original value and identify the authoritative source", "apply the correction with a reason in the audit log", "rerun the affected validation and notify the dataset owner"),
    ("community", "open a volunteer check-in desk", "prepare the roster and a private way to resolve missing names", "brief volunteers on roles and emergency contacts", "track arrivals without displaying personal details publicly"),
    ("community", "respond to a venue accessibility concern", "ask what barrier the visitor encountered", "provide an immediate accessible route or equivalent service", "document the barrier and assign a permanent correction"),
    ("finance", "process a small reimbursement", "match the claim to the policy, receipt, and approval", "record the eligible amount under the correct cost code", "send the claimant a clear decision and retain the evidence"),
    ("finance", "resolve a disputed invoice line", "compare the invoice with the order and delivery record", "ask the supplier to correct the unsupported line", "approve payment only after the revised total matches the records"),
    ("software", "release a configuration change", "capture the current configuration and define a rollback trigger", "apply the change to the smallest safe scope and monitor it", "expand only after the checks pass, otherwise restore the snapshot"),
    ("software", "handle a failed scheduled job", "preserve the error log and identify the last successful run", "fix the confirmed cause and rerun one bounded batch", "verify outputs before releasing the remaining backlog"),
    ("communications", "publish an urgent service update", "verify the affected service, timeframe, and contact", "write the impact and available alternative in plain language", "publish through the active channels and remove the notice when resolved"),
    ("communications", "correct a public factual error", "capture the original statement and verify the correction", "publish the corrected fact with a transparent note", "update linked copies and monitor for continued circulation"),
)
_PROMPTS = (
    "Turn this goal into a concrete three-step action plan: {scenario[brief]}",
    "Give the next practical sequence under the stated condition: {scenario[brief]}",
    "What should be done first, next, and finally? {scenario[brief]}",
    "Provide an executable plan without invented deadlines or resources: {scenario[brief]}",
    "Convert this bounded task into ordered actions: {scenario[brief]}",
    "State a safe, checkable workflow for this request: {scenario[brief]}",
)
_ANSWERS = (
    "For the {scenario[domain]} goal to {scenario[goal]} at {scenario[site]}: first, {scenario[first]}. Next, {scenario[second]}. Finally, {scenario[third]}. Throughout, account for the condition that {scenario[constraint]}.",
    "In this {scenario[domain]} task at {scenario[site]}, begin by having the team {scenario[first]}. Then {scenario[second]}. Close the task by ensuring they {scenario[third]}, while {scenario[constraint]}.",
    "The practical {scenario[domain]} sequence for {scenario[goal]} is: 1) {scenario[first]}; 2) {scenario[second]}; 3) {scenario[third]}. Apply it at {scenario[site]} with the limitation that {scenario[constraint]}.",
    "Start the {scenario[domain]} work to {scenario[goal]} at {scenario[site]} by doing this: {scenario[first]}. After that, {scenario[second]}. Complete the handoff when you {scenario[third]}; remember that {scenario[constraint]}.",
    "Use three {scenario[domain]} checkpoints at {scenario[site]} for {scenario[goal]}. Preparation: {scenario[first]}. Execution: {scenario[second]}. Completion: {scenario[third]}. The plan must still respect that {scenario[constraint]}.",
    "For this {scenario[domain]} task to {scenario[goal]}, {scenario[first]}; once confirmed, {scenario[second]}; then {scenario[third]}. That order keeps the work compatible with the fact that {scenario[constraint]}.",
)
_LOCAL_DETAILS = (
    "the reception lead records the final handoff",
    "the evening coordinator controls the room key",
    "the clinical desk confirms any service interruption",
    "the information desk maintains the public notice",
    "the field lead stores the offline checklist",
    "the dispatcher records every maintenance window",
    "the volunteer captain confirms the next owner",
    "the study coordinator protects the source records",
    "the venue manager authorizes public-area changes",
    "the course lead signs the readiness entry",
    "the inventory clerk verifies movement between zones",
    "the grounds supervisor closes outdoor work orders",
    "the collections manager approves exhibit changes",
    "the bench technician checks returned equipment",
    "the mobile lead records conditions before departure",
    "the laboratory supervisor reviews student access",
    "the kitchen manager logs sanitation completion",
    "the archivist witnesses changes to secured records",
    "the branch manager confirms remote support results",
    "the shift coordinator briefs incoming volunteers",
    "the design owner preserves the current prototype",
    "the kiosk attendant redirects visitors during closure",
    "the garage foreperson signs the ventilation check",
    "the scheduler validates the sandbox result",
)


def practical_action_capacity() -> int:
    return len(_TASKS) * len(SITES)


def render_practical_action_rows() -> list[dict[str, object]]:
    rows = []
    for domain, goal, first, second, third in _TASKS:
        for site_index, site in enumerate(SITES):
            constraint = _CONSTRAINTS[site_index]
            local_detail = _LOCAL_DETAILS[site_index]
            brief = (
                f"At {site}, the team must {goal}; {constraint}, and "
                f"{local_detail}."
            )
            contextual_answers = tuple(
                f"{answer} Locally, {local_detail}." for answer in _ANSWERS
            )
            variables = RoleSeparatedVariableBy(
                VariableBy2D(
                    {
                        "scenario": {
                            "goal": (goal,), "first": (first,), "second": (second,),
                            "third": (third,), "constraint": (constraint,),
                            "site": (site,), "domain": (domain,),
                            "local_detail": (local_detail,), "brief": (brief,),
                        },
                        "prompt": {"action_request": _PROMPTS},
                        "answer": {"ordered_plan": contextual_answers},
                    }
                )
            )
            deck = V2RoleSeparatedDeck(
                name=f"{TASK}:{domain}:{goal}", variables=variables,
                prompt_pools=(V2SubcardPool("action_request", SurfaceRole.PROMPT, ("{prompt[action_request]}",)),),
                answer_pools=(V2SubcardPool("ordered_plan", SurfaceRole.ANSWER, ("{answer[ordered_plan]}",)),),
            )
            case_id = f"{domain}:{goal}:{site}"
            rows.append(
                render_v2_row(
                    task=TASK, case_id=case_id, domain=domain, difficulty="medium",
                    deck=deck,
                    facts={"goal": goal, "first": first, "second": second, "third": third, "constraint": constraint, "local_detail": local_detail, "site": site},
                    validator={"kind": "contains", "required": [first, second, third, local_detail]},
                )
            )
    return validate_complete_rows(TASK, rows, practical_action_capacity())


__all__ = ("practical_action_capacity", "render_practical_action_rows")
