from __future__ import annotations

from ...variable_by import VariableBy2D
from ..contracts import RoleSeparatedVariableBy, SurfaceRole
from ..decks import V2RoleSeparatedDeck, V2SubcardPool
from ._axes import SITES
from ._common import render_v2_row, validate_complete_rows


TASK = "planning_comparison"
_DECISIONS = (
    ("logistics", "daily supply run", "a direct van route", "two bicycle relays", "a direct van route", "it carries the full load in one controlled handoff"),
    ("logistics", "urgent document delivery", "the scheduled courier", "a secure digital transfer", "a secure digital transfer", "the verified file can arrive immediately without physical transit"),
    ("operations", "equipment maintenance", "one full shutdown", "staged service by zone", "staged service by zone", "essential service stays available in the untreated zones"),
    ("operations", "room allocation", "a fixed weekly timetable", "daily manual assignments", "a fixed weekly timetable", "recurring groups receive predictable access with fewer conflicts"),
    ("training", "staff onboarding", "one large lecture", "three guided workshops", "three guided workshops", "participants can practice and receive feedback in smaller groups"),
    ("training", "reference material", "a printed handbook", "a searchable online guide", "a searchable online guide", "updates become available without redistributing every copy"),
    ("research", "field measurements", "one intensive survey day", "four shorter observation visits", "four shorter observation visits", "the observations cover variation across multiple days"),
    ("research", "participant feedback", "an open discussion", "anonymous written responses", "anonymous written responses", "participants can report concerns without social pressure"),
    ("community", "event registration", "walk-in registration only", "advance registration with reserved walk-in places", "advance registration with reserved walk-in places", "capacity is visible while access remains available for unregistered visitors"),
    ("community", "meal distribution", "one central pickup point", "several neighborhood pickup points", "several neighborhood pickup points", "travel distance is reduced for recipients"),
    ("finance", "small purchases", "individual reimbursements", "a controlled purchasing card", "a controlled purchasing card", "receipts and limits can be reviewed in one account"),
    ("finance", "annual budgeting", "copy last year's allocations", "rebuild estimates from current needs", "rebuild estimates from current needs", "the plan reflects current quantities and prices"),
    ("software", "application release", "deploy every change together", "release behind a reversible feature flag", "release behind a reversible feature flag", "the change can be disabled quickly if monitoring detects harm"),
    ("software", "data migration", "replace all records in one run", "migrate validated batches", "migrate validated batches", "each batch can be checked before the next one moves"),
    ("communications", "urgent service notice", "wait for the monthly newsletter", "publish a short targeted alert", "publish a short targeted alert", "affected people receive the time-sensitive information promptly"),
    ("communications", "policy explanation", "send the full legal text alone", "pair a plain-language summary with the source", "pair a plain-language summary with the source", "readers get an accessible explanation and can verify the original"),
)
_CONSTRAINTS = (
    "public access must remain open", "the change window lasts thirty minutes",
    "no additional vehicles are available", "the team includes new volunteers",
    "records contain confidential details", "the result must be reversible",
    "weather may interrupt outdoor work", "the budget cannot exceed the approved amount",
    "several users rely on assistive technology", "the site has intermittent connectivity",
    "an audit trail is required", "the decision must work across two shifts",
    "storage space is limited", "the audience speaks several languages",
    "the equipment cannot leave the building", "the deadline is tomorrow morning",
    "a backup process must remain available", "only trained staff may authorize changes",
    "the number of participants may change", "the source data arrives in stages",
    "the public needs a simple explanation", "the result will be reviewed independently",
    "one route is temporarily closed", "the plan must support a later handoff",
)
_PROMPTS = (
    "Compare the two options and choose one under the stated constraint: {scenario[brief]}",
    "Which plan is better here, and why? {scenario[brief]}",
    "Make a constraint-aware choice between these alternatives: {scenario[brief]}",
    "Select the stronger option without inventing new resources: {scenario[brief]}",
    "Evaluate both approaches against the operational limit: {scenario[brief]}",
    "Recommend one plan and name the decisive reason: {scenario[brief]}",
)
_ANSWERS = (
    "For {scenario[goal]} at {scenario[site]}, choose {scenario[winner]}. It is the better fit because {scenario[reason]}, while respecting that {scenario[constraint]}.",
    "At {scenario[site]}, I recommend {scenario[winner]} for {scenario[goal]}: {scenario[reason]}. This also accommodates the requirement that {scenario[constraint]}.",
    "For the {scenario[goal]} decision at {scenario[site]}, {scenario[winner]} is stronger. The decisive advantage is that {scenario[reason]}.",
    "The preferable {scenario[goal]} plan for {scenario[site]} is {scenario[winner]}, since {scenario[reason]}; it can be implemented while {scenario[constraint]}.",
    "Use {scenario[winner]} for {scenario[goal]} at {scenario[site]}. Its practical edge is that {scenario[reason]}, which matters when {scenario[constraint]}.",
    "Between the two plans for {scenario[goal]} at {scenario[site]}, {scenario[winner]} best satisfies the situation: {scenario[reason]}. Keep the condition that {scenario[constraint]} explicit during execution.",
)


def planning_comparison_capacity() -> int:
    return len(_DECISIONS) * len(SITES)


def render_planning_comparison_rows() -> list[dict[str, object]]:
    rows = []
    for domain, goal, option_a, option_b, winner, reason in _DECISIONS:
        for site_index, site in enumerate(SITES):
            constraint = _CONSTRAINTS[site_index]
            brief = (
                f"At {site}, the goal is {goal}. Option A is {option_a}; option B is "
                f"{option_b}. The operating condition is that {constraint}."
            )
            variables = RoleSeparatedVariableBy(
                VariableBy2D(
                    {
                        "scenario": {
                            "goal": (goal,), "option_a": (option_a,),
                            "option_b": (option_b,), "winner": (winner,),
                            "reason": (reason,), "constraint": (constraint,),
                            "site": (site,), "brief": (brief,),
                        },
                        "prompt": {"decision": _PROMPTS},
                        "answer": {"recommendation": _ANSWERS},
                    }
                )
            )
            deck = V2RoleSeparatedDeck(
                name=f"{TASK}:{domain}:{goal}", variables=variables,
                prompt_pools=(V2SubcardPool("decision", SurfaceRole.PROMPT, ("{prompt[decision]}",)),),
                answer_pools=(V2SubcardPool("recommendation", SurfaceRole.ANSWER, ("{answer[recommendation]}",)),),
            )
            case_id = f"{domain}:{goal}:{site}"
            rows.append(
                render_v2_row(
                    task=TASK, case_id=case_id, domain=domain, difficulty="medium",
                    deck=deck,
                    facts={"goal": goal, "option_a": option_a, "option_b": option_b, "winner": winner, "reason": reason, "constraint": constraint, "site": site},
                    validator={"kind": "contains", "required": [winner, reason]},
                )
            )
    return validate_complete_rows(TASK, rows, planning_comparison_capacity())


__all__ = ("planning_comparison_capacity", "render_planning_comparison_rows")
