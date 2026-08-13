from __future__ import annotations

from ...variable_by import VariableBy2D
from ..contracts import RoleSeparatedVariableBy, SurfaceRole
from ..decks import (
    V2RoleSeparatedDeck,
    V2SubcardPool,
    answer_variant_plans,
    prompt_variant_plans,
)
from ._axes import SITES
from ._common import render_v2_row, validate_complete_rows
from .planning_comparison import _CONSTRAINTS


TASK = "brainstorming_creativity"
_CHALLENGES = (
    ("community", "help first-time visitors find the right service", "a color-and-symbol route card", "a staffed five-minute orientation point", "a pre-visit photo guide", "test the route card with three visitors who have never entered the building"),
    ("community", "increase volunteer attendance at early shifts", "a shift-buddy commitment", "a reminder chosen by each volunteer", "a standby pool with short assignments", "compare attendance for one week using volunteer-selected reminders"),
    ("training", "make a technical workshop less passive", "a live error-finding exercise", "paired teach-back rounds", "a small scenario lab", "replace one lecture section with the scenario lab and collect completion evidence"),
    ("training", "help learners remember a multi-step procedure", "a pocket decision tree", "spaced retrieval cards", "a peer-observation checklist", "pilot the retrieval cards after two different time intervals"),
    ("operations", "reduce room-booking conflicts", "automatic overlap warnings", "a visible priority rule", "a weekly exception review", "enable overlap warnings for one building and count prevented conflicts"),
    ("operations", "make maintenance handoffs clearer", "before-and-after photo pairs", "a remaining-work field", "a spoken two-minute shift recap", "add the remaining-work field to ten tickets and audit the next shift's questions"),
    ("logistics", "prevent parcels from missing transfer scans", "a scan-to-open cage latch", "a contrasting unscanned label", "a transfer completion board", "try the contrasting label on one route and measure missed scans"),
    ("logistics", "reduce wasted trips between storage zones", "a shared pickup queue", "scheduled consolidation windows", "a return-trip request board", "run one consolidation window and compare trip counts with the prior day"),
    ("research", "capture field context consistently", "a sensor-linked voice note", "a fixed observation snapshot", "a context prompt on data save", "add the save-time prompt to one instrument and inspect missing context fields"),
    ("research", "invite quieter participants into discussion", "silent idea collection first", "rotating written questions", "anonymous digital contributions", "start one session with silent collection and compare participation breadth"),
    ("software", "make configuration changes safer", "a generated before-and-after diff", "a one-click rollback snapshot", "a small-scope canary mode", "require a rollback snapshot for one service and time recovery in a simulation"),
    ("software", "help users notice stale dashboard data", "a human-readable freshness sentence", "a color-independent age badge", "an alert only when a normal refresh is missed", "show the freshness sentence to five users and ask them to identify data age"),
    ("communications", "make urgent notices easier to understand", "an impact-first opening", "a separate available-alternative box", "a one-line update timestamp", "compare comprehension of the impact-first version with the current notice"),
    ("communications", "collect corrections before publication", "a claim-by-claim review board", "a source-link requirement", "a red-team reading pass", "use the claim board on one article and count unsupported statements found"),
    ("environment", "reduce disposable cup use", "a borrow-and-return mug shelf", "a personal cup reward stamp", "a meeting-kit of washable cups", "pilot the mug shelf for two weeks and track returns and disposable use"),
    ("environment", "make energy waste visible", "a closing-shift energy walk", "device-level idle tags", "a weekly avoidable-use estimate", "run the energy walk for five evenings and record repeated waste sources"),
)
_PROMPTS = (
    "Generate three genuinely different ideas and one bounded first test: {scenario[brief]}",
    "Brainstorm three approaches that use different mechanisms, then pick a small experiment: {scenario[brief]}",
    "Offer three distinct options for this challenge without inventing extra resources: {scenario[brief]}",
    "Create a diverse idea set and identify the cheapest informative trial: {scenario[brief]}",
    "Propose alternatives rather than cosmetic variants of one solution: {scenario[brief]}",
    "Give three mechanisms for addressing the goal and a concrete way to test one: {scenario[brief]}",
)
_ANSWERS = (
    "For {scenario[challenge]} at {scenario[site]}: (1) {scenario[idea_one]}; (2) {scenario[idea_two]}; (3) {scenario[idea_three]}. First test: {scenario[test]}, while respecting that {scenario[constraint]}.",
    "Three different mechanisms fit the {scenario[site]} challenge. One is {scenario[idea_one]}; another is {scenario[idea_two]}; a third is {scenario[idea_three]}. Begin by having the team {scenario[test]} under the condition that {scenario[constraint]}.",
    "Option A uses {scenario[idea_one]}. Option B instead uses {scenario[idea_two]}. Option C explores {scenario[idea_three]}. The bounded experiment at {scenario[site]} is to {scenario[test]}, with {scenario[constraint]} kept explicit.",
    "The idea set for {scenario[challenge]} is deliberately varied: {scenario[idea_one]}, {scenario[idea_two]}, and {scenario[idea_three]}. To learn quickly, {scenario[test]} at {scenario[site]} without violating the fact that {scenario[constraint]}.",
    "At {scenario[site]}, consider {scenario[idea_one]} for guidance, {scenario[idea_two]} for human support, and {scenario[idea_three]} for a different delivery path. Evaluate the set by asking the team to {scenario[test]} while {scenario[constraint]}.",
    "A useful three-way exploration is {scenario[idea_one]} versus {scenario[idea_two]} versus {scenario[idea_three]}. The next evidence should come from this trial: {scenario[test]}; the local boundary is that {scenario[constraint]}.",
)
_PROMPT_FUNCTIONS = (
    ("request_ideas", "request_bounded_test"),
    ("request_mechanisms", "request_experiment"),
    ("request_options", "enforce_resource_boundary"),
    ("request_diverse_set", "request_low_cost_trial"),
    ("reject_cosmetic_variants", "request_alternatives"),
    ("request_mechanisms", "request_concrete_test"),
)
_ANSWER_FUNCTIONS = (
    ("frame_challenge", "enumerate_ideas", "propose_test", "state_boundary"),
    ("frame_mechanisms", "contrast_ideas", "propose_test", "state_boundary"),
    ("label_options", "contrast_ideas", "propose_test", "state_boundary"),
    ("state_goal", "group_ideas", "prioritize_learning", "state_boundary"),
    ("locate_context", "map_ideas_to_mechanisms", "propose_evaluation"),
    ("contrast_ideas", "request_evidence", "state_boundary"),
)


def brainstorming_creativity_capacity() -> int:
    return len(_CHALLENGES) * len(SITES)


def render_brainstorming_creativity_rows() -> list[dict[str, object]]:
    rows = []
    for domain, challenge, idea_one, idea_two, idea_three, test in _CHALLENGES:
        for site_index, site in enumerate(SITES):
            constraint = _CONSTRAINTS[site_index]
            brief = f"At {site}, the team wants to {challenge}; {constraint}."
            variables = RoleSeparatedVariableBy(
                VariableBy2D(
                    {
                        "scenario": {
                            "challenge": (challenge,), "idea_one": (idea_one,),
                            "idea_two": (idea_two,), "idea_three": (idea_three,),
                            "test": (test,), "constraint": (constraint,),
                            "site": (site,), "brief": (brief,),
                        },
                        "prompt": {"idea_request": _PROMPTS},
                        "answer": {"idea_set": _ANSWERS},
                    }
                )
            )
            deck = V2RoleSeparatedDeck(
                name=f"{TASK}:{domain}:{challenge}", variables=variables,
                prompt_pools=(V2SubcardPool("idea_request", SurfaceRole.PROMPT, ("{prompt[idea_request]}",)),),
                answer_pools=(V2SubcardPool("idea_set", SurfaceRole.ANSWER, ("{answer[idea_set]}",)),),
                prompt_plans=prompt_variant_plans(
                    sense="idea_request",
                    pool_name="idea_request",
                    functions=_PROMPT_FUNCTIONS,
                ),
                answer_plans=answer_variant_plans(
                    sense="idea_set",
                    pool_name="idea_set",
                    functions=_ANSWER_FUNCTIONS,
                ),
            )
            rows.append(
                render_v2_row(
                    task=TASK, case_id=f"{domain}:{challenge}:{site}", domain=domain,
                    difficulty="medium", deck=deck,
                    facts={"challenge": challenge, "ideas": [idea_one, idea_two, idea_three], "test": test, "constraint": constraint, "site": site},
                    validator={"kind": "contains", "required": [idea_one, idea_two, idea_three, test]},
                )
            )
    return validate_complete_rows(TASK, rows, brainstorming_creativity_capacity())


__all__ = ("brainstorming_creativity_capacity", "render_brainstorming_creativity_rows")
