from __future__ import annotations

from ...variable_by import VariableBy2D
from ..contracts import RoleSeparatedVariableBy, SurfaceRole
from ..decks import V2RoleSeparatedDeck, V2SubcardPool, prompt_variant_plans
from ..reservoirs.summarization import (
    SUMMARY_CASES,
    SUMMARY_CONSTRAINTS,
    SUMMARY_CONTEXTS,
)
from ._axes import PEOPLE, SITES
from ._common import render_v2_row, validate_complete_rows


TASK = "summarization_synthesis"

_FOCI = {
    "executive": "prioritize the issue, response, and verified outcome",
    "action": "prioritize completed work and the preventive next step",
    "impact": "prioritize the observable result and what produced it",
    "handoff": "prioritize current status and what the next owner must retain",
}

_DELIVERY_GUIDANCE = (
    "lead with the requested priority instead of retelling the record",
    "use two direct sentences and make the relationship between the facts explicit",
)

_PROMPTS = (
    "Summarize this record with an {scenario[focus]} focus; {scenario[guidance]}. Record: {scenario[record]}",
    "Produce a two-sentence {scenario[focus]} summary. In particular, {scenario[guidance]}. Source: {scenario[record]}",
    "Synthesize the following notes for a {scenario[focus]} handoff; {scenario[guidance]}. {scenario[record]}",
    "Condense this report without copying its wording. Use an {scenario[focus]} focus and {scenario[guidance]}. Report: {scenario[record]}",
    "Give a concise {scenario[focus]} synthesis; {scenario[guidance]}. Material: {scenario[record]}",
    "Turn the record below into a useful {scenario[focus]} brief; {scenario[guidance]}. {scenario[record]}",
    "Write the {scenario[focus]} summary a colleague would need. Make sure to {scenario[guidance]}. Notes: {scenario[record]}",
    "Reduce these notes to their {scenario[focus]} essentials while you {scenario[guidance]}. Notes: {scenario[record]}",
)

_PROMPT_FUNCTIONS = (
    ("request_summary", "specify_focus", "supply_record"),
    ("request_two_sentences", "specify_focus", "supply_source"),
    ("request_synthesis", "specify_handoff_focus"),
    ("request_condensation", "forbid_copying", "specify_focus"),
    ("request_concise_synthesis", "specify_focus", "supply_material"),
    ("request_brief", "specify_focus", "supply_record"),
    ("request_colleague_summary", "specify_priority", "supply_notes"),
    ("request_essential_summary", "specify_focus", "supply_notes"),
)

_RECORD_TEMPLATES = (
    "Incident: {issue}. Owner: {person} at {site}. Context: {context}. Response: the team {action}. Observed result: {outcome}. Recommendation: {follow_up}. Constraint: {constraint}",
    "At {site}, {person} recorded a {domain} issue {context}: {issue}. The team {action}. Afterwards, {outcome}. The proposed prevention step is to {follow_up}. {constraint}",
    "Issue — {issue}. Location — {site}. Owner — {person}. Timing — {context}. Action — the team {action}. Result — {outcome}. Next measure — {follow_up}. Note — {constraint}",
    "{person}'s {domain} update from {site}, {context}: {issue}. The team {action}; the result was that {outcome}. For later work, {follow_up}. {constraint}",
    "A record from {site} says that {issue} {context}. {person} coordinated the response, and the team {action}. It then reported that {outcome}, with a recommendation to {follow_up}. {constraint}",
    "Context: {site}, {context}. Responsible person: {person}. Problem: {issue}. Completed work: the team {action}. Verified outcome: {outcome}. Preventive follow-up: {follow_up}. Operating limit: {constraint}",
    "{domain_cap} note from {site}: {issue} {context}. {person} reports that the team {action}, after which {outcome}. The next control is to {follow_up}. {constraint}",
    "For review — place: {site}; owner: {person}; situation: {issue} {context}; response: the team {action}; result: {outcome}; prevention: {follow_up}; constraint: {constraint}",
)

_ANSWER_TEMPLATES = {
    "executive": (
        "At {site}, {issue_answer} {context_answer}. Under {person}'s ownership, the team {action_answer}; {outcome_answer}, and {constraint_answer}",
        "{person}'s team at {site} responded when {issue_answer} {context_answer}. They {action_answer}, so {outcome_answer} while {constraint_answer}",
        "The {domain} record from {site} shows that {issue_answer} {context_answer}. {person} coordinated the response: the team {action_answer}, {outcome_answer}, and {constraint_answer}",
        "When {issue_answer} at {site} {context_answer}, {person} led the response. The team {action_answer}; {outcome_answer}, with this condition met: {constraint_answer}",
    ),
    "action": (
        "Because {issue_answer} {context_answer}, the team at {site} {action_answer}. {person}'s next preventive step is to {follow_answer}, and {constraint_answer}",
        "The completed response at {site} was that the team {action_answer}. To prevent another case where {issue_answer}, {person} should {follow_answer}; {constraint_answer}",
        "{person} responded after {issue_answer} at {site} {context_answer}. The team {action_answer}; the follow-up is to {follow_answer}, while {constraint_answer}",
        "At {site}, the team {action_answer} when {issue_answer} {context_answer}. {person} should now {follow_answer}; {constraint_answer}",
    ),
    "impact": (
        "The observable result at {site} was that {outcome_answer}. This followed {person}'s response when {issue_answer} {context_answer}: the team {action_answer}, and {constraint_answer}",
        "At {site}, {outcome_answer}. The change came after {person}'s team {action_answer} in response to {issue_answer} {context_answer}, while {constraint_answer}",
        "The people affected by the {domain} issue saw a clear result: {outcome_answer}. {person} achieved this after the team {action_answer} at {site}; {constraint_answer}",
        "The response produced a measurable outcome at {site}: {outcome_answer}. It followed the team's work after {issue_answer} {context_answer}, coordinated by {person}, and {constraint_answer}",
    ),
    "handoff": (
        "Current status at {site}: {outcome_answer}. The next {domain} owner should {follow_answer}; {person} can explain how the team {action_answer} after {issue_answer}, and {constraint_answer}",
        "The handoff status is that {outcome_answer}. At {site}, the next owner must {follow_answer}; {person}'s response to {issue_answer} shows that {constraint_answer}",
        "At {site}, the response to {issue_answer} is complete and {outcome_answer}. The next owner should {follow_answer}; {person} coordinated the work, and {constraint_answer}",
        "For the next {domain} owner: {outcome_answer}. Remember to {follow_answer}, consult {person} about how the team {action_answer}, and note that {constraint_answer}",
    ),
}


def summarization_synthesis_capacity() -> int:
    return (
        len(SUMMARY_CASES)
        * len(SUMMARY_CONTEXTS)
        * len(SUMMARY_CONSTRAINTS)
        * len(_FOCI)
        * 2
    )


def _record(case: tuple[str, ...], context_index: int, constraint_index: int) -> dict[str, str]:
    (
        domain,
        issue,
        issue_answer,
        action,
        action_answer,
        outcome,
        outcome_answer,
        follow_up,
        follow_answer,
    ) = case
    context, context_answer = SUMMARY_CONTEXTS[context_index]
    constraint, constraint_answer = SUMMARY_CONSTRAINTS[constraint_index]
    return {
        "domain": domain,
        "issue": issue,
        "issue_answer": issue_answer,
        "action": action,
        "action_answer": action_answer,
        "outcome": outcome,
        "outcome_answer": outcome_answer,
        "follow_up": follow_up,
        "follow_answer": follow_answer,
        "context": context,
        "context_answer": context_answer,
        "constraint": constraint,
        "constraint_answer": constraint_answer,
    }


def render_summarization_synthesis_rows() -> list[dict[str, object]]:
    rows = []
    for case_index, case in enumerate(SUMMARY_CASES):
        for context_index in range(len(SUMMARY_CONTEXTS)):
            for constraint_index in range(len(SUMMARY_CONSTRAINTS)):
                facts = _record(case, context_index, constraint_index)
                person = PEOPLE[(case_index + context_index + constraint_index) % len(PEOPLE)]
                site = SITES[(case_index * 3 + context_index * 2 + constraint_index) % len(SITES)]
                facts.update({"person": person, "site": site})
                record_variant = (case_index + context_index + constraint_index) % len(_RECORD_TEMPLATES)
                record = _RECORD_TEMPLATES[record_variant].format(
                    **facts,
                    domain_cap=facts["domain"].capitalize(),
                )
                for focus_index, (focus, base_guidance) in enumerate(_FOCI.items()):
                    for style_offset in range(2):
                        guidance = (
                            f"{base_guidance}; "
                            f"{_DELIVERY_GUIDANCE[style_offset]}"
                        )
                        answer_variant = (
                            context_index + constraint_index + focus_index + style_offset
                        ) % len(_ANSWER_TEMPLATES[focus])
                        target = _ANSWER_TEMPLATES[focus][answer_variant].format(**facts) + "."
                        variables = RoleSeparatedVariableBy(
                            VariableBy2D(
                                {
                                    "scenario": {
                                        "focus": (focus,),
                                        "guidance": (guidance,),
                                        "record": (record,),
                                    },
                                    "prompt": {"summary_request": _PROMPTS},
                                    "answer": {"summary": (target,)},
                                }
                            )
                        )
                        deck = V2RoleSeparatedDeck(
                            name=f"{TASK}:{facts['domain']}:{focus}:{answer_variant}",
                            variables=variables,
                            prompt_pools=(
                                V2SubcardPool(
                                    "summary_request",
                                    SurfaceRole.PROMPT,
                                    ("{prompt[summary_request]}",),
                                ),
                            ),
                            answer_pools=(
                                V2SubcardPool(
                                    "summary",
                                    SurfaceRole.ANSWER,
                                    ("{answer[summary]}",),
                                ),
                            ),
                            prompt_plans=prompt_variant_plans(
                                sense="summary_request",
                                pool_name="summary_request",
                                functions=_PROMPT_FUNCTIONS,
                            ),
                        )
                        case_id = (
                            f"{facts['domain']}:{case_index}:{context_index}:"
                            f"{constraint_index}:{focus}:{style_offset}"
                        )
                        rows.append(
                            render_v2_row(
                                task=TASK,
                                case_id=case_id,
                                domain=facts["domain"],
                                difficulty="medium",
                                deck=deck,
                                facts={
                                    **facts,
                                    "focus": focus,
                                    "record_variant": record_variant,
                                    "answer_variant": answer_variant,
                                },
                                validator={"kind": "exact", "expected": target},
                            )
                        )
    return validate_complete_rows(TASK, rows, summarization_synthesis_capacity())


__all__ = ("render_summarization_synthesis_rows", "summarization_synthesis_capacity")
