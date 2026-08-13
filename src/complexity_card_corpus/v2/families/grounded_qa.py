from __future__ import annotations

from ...variable_by import VariableBy2D
from ..contracts import RoleSeparatedVariableBy, SurfaceRole
from ..decks import (
    V2RoleSeparatedDeck,
    V2SubcardPool,
    answer_variant_plans,
    prompt_variant_plans,
)
from ._axes import PEOPLE, SITES
from ._common import render_v2_row, validate_complete_rows


TASK = "grounded_qa"
_FACTS = (
    ("operations", "inspection", "The completion record at {site} lists {person} beside the inspection entry.", "Who completed the inspection?", "{person} completed the inspection at {site}."),
    ("operations", "location", "The relocation note assigns {person}'s equipment to {site} after servicing.", "Where should the serviced equipment go?", "The serviced equipment handled by {person} belongs at {site}."),
    ("planning", "deadline", "The schedule at {site} marks Thursday at 16:00 as the final submission point for {person}.", "When is the final submission due?", "The documented deadline at {site} is Thursday at 16:00 for {person}."),
    ("planning", "status", "The tracker at {site} shows the access review as approved, with {person} recording the decision.", "What is the access review status?", "The access review at {site} is approved; {person} recorded that decision."),
    ("inventory", "quantity", "The verified count at {site} is 48 sealed filter packs under {person}'s entry.", "How many sealed filter packs are verified?", "{person}'s verified inventory contains 48 sealed filter packs at {site}."),
    ("inventory", "material", "The specification for {person}'s panel requires recycled aluminum at {site}.", "Which material does the specification require?", "Recycled aluminum is required for {person}'s panel at {site}."),
    ("support", "contact", "The escalation sheet names {person} as the after-hours contact for {site}.", "Who is the after-hours contact?", "For after-hours issues at {site}, the listed contact is {person}."),
    ("support", "channel", "The service note directs {person}'s urgent requests from {site} through the staffed telephone line.", "Which channel should urgent requests use?", "{person} should route urgent requests from {site} through the staffed telephone line."),
    ("research", "method", "The protocol says {person} will collect readings with a calibrated optical sensor at {site}.", "How will the readings be collected?", "The recorded method uses a calibrated optical sensor, operated by {person}."),
    ("research", "sample", "The sampling plan reserves the northern transect at {site} for {person}'s morning survey.", "Which area is reserved for the morning survey?", "{person}'s morning survey is assigned to the northern transect at {site}."),
    ("training", "audience", "The course brief limits {person}'s session at {site} to first-time coordinators.", "Who is the session intended for?", "{person}'s intended audience is first-time coordinators at {site}."),
    ("training", "duration", "The agenda gives {person} a 75-minute teaching block at {site}.", "How long is the teaching block?", "{person}'s teaching block lasts 75 minutes at {site}."),
    ("finance", "limit", "The approval memo sets a maximum reimbursable amount of $240 for {person}'s visit to {site}.", "What is the reimbursement limit?", "The documented ceiling for {person}'s visit to {site} is $240."),
    ("finance", "code", "The ledger note assigns cost code R17 to {person}'s work at {site}.", "Which cost code applies?", "Cost code R17 applies to {person}'s work at {site}."),
    ("communications", "language", "The publication brief asks {person} to release the notice in English and Spanish at {site}.", "Which languages should the notice use?", "{person} should release the notice in English and Spanish at {site}."),
    ("communications", "approval", "The release log says {person} must approve the notice before it is posted at {site}.", "Whose approval is required before posting?", "Posting at {site} requires {person}'s approval."),
)
_PROMPTS = (
    "Answer only from the supplied note. Note: {scenario[note]} Question: {scenario[question]}",
    "Use this record as the sole source: {scenario[note]} Now answer: {scenario[question]}",
    "Ground the answer in the following entry. {scenario[note]} {scenario[question]}",
    "Read the documented fact, then answer without adding assumptions. {scenario[note]} Question: {scenario[question]}",
    "Based strictly on this note—{scenario[note]}—{scenario[question]}",
    "Consult the provided record: {scenario[note]} Please answer: {scenario[question]}",
)
_ANSWERS = (
    "{scenario[answer]}",
    "According to the record, {scenario[answer_lower]}",
    "The documented answer is clear: {scenario[answer]}",
    "The note supports this response: {scenario[answer]}",
    "The entry yields this result: {scenario[answer]}",
    "The record establishes that {scenario[answer_lower]}",
)
_PROMPT_FUNCTIONS = (
    ("restrict_source", "supply_note", "ask_question"),
    ("declare_sole_source", "supply_note", "ask_question"),
    ("require_grounding", "supply_note", "ask_question"),
    ("supply_fact", "forbid_assumptions", "ask_question"),
    ("require_strict_grounding", "supply_note", "ask_question"),
    ("request_record_consultation", "supply_note", "ask_question"),
)
_ANSWER_FUNCTIONS = (
    ("answer_from_record",),
    ("attribute_record", "answer_from_record"),
    ("signal_documented_certainty", "answer_from_record"),
    ("attribute_support", "answer_from_record"),
    ("derive_from_entry", "answer_from_record"),
    ("state_record_entailment", "answer_from_record"),
)


def grounded_qa_capacity() -> int:
    return len(_FACTS) * len(SITES) * len(PEOPLE)


def render_grounded_qa_rows() -> list[dict[str, object]]:
    rows = []
    for domain, fact_kind, note_template, question, answer_template in _FACTS:
        for site in SITES:
            for person in PEOPLE:
                note = note_template.format(site=site, person=person)
                answer = answer_template.format(site=site, person=person)
                answer_lower = answer[0].lower() + answer[1:]
                contextual_answers = (
                    f"For the {fact_kind} entry concerning {person} at {site}, {answer_lower}",
                    f"The {fact_kind} record for {person} at {site} supports this: {answer}",
                    f"Reading the {fact_kind} entry for {site} gives the following: {answer}",
                    f"The documented {fact_kind} result for {person} is unambiguous: {answer}",
                    f"From the {fact_kind} record associated with {site}, {answer_lower}",
                    f"The {site} entry concerning {person}'s {fact_kind} establishes that {answer_lower}",
                )
                variables = RoleSeparatedVariableBy(
                    VariableBy2D(
                        {
                            "scenario": {
                                "note": (note,), "question": (question,),
                                "answer": (answer,),
                                "answer_lower": (answer[0].lower() + answer[1:],),
                                "site": (site,), "person": (person,),
                                "fact_kind": (fact_kind,),
                            },
                            "prompt": {"grounded_question": _PROMPTS},
                            "answer": {"supported_answer": contextual_answers},
                        }
                    )
                )
                deck = V2RoleSeparatedDeck(
                    name=f"{TASK}:{domain}:{fact_kind}", variables=variables,
                    prompt_pools=(V2SubcardPool("grounded_question", SurfaceRole.PROMPT, ("{prompt[grounded_question]}",)),),
                    answer_pools=(V2SubcardPool("supported_answer", SurfaceRole.ANSWER, ("{answer[supported_answer]}",)),),
                    prompt_plans=prompt_variant_plans(
                        sense="grounded_question",
                        pool_name="grounded_question",
                        functions=_PROMPT_FUNCTIONS,
                    ),
                    answer_plans=answer_variant_plans(
                        sense="supported_answer",
                        pool_name="supported_answer",
                        functions=_ANSWER_FUNCTIONS,
                    ),
                )
                case_id = f"{domain}:{fact_kind}:{site}:{person}"
                rows.append(
                    render_v2_row(
                        task=TASK, case_id=case_id, domain=domain, difficulty="easy",
                        deck=deck,
                        facts={"fact_kind": fact_kind, "note": note, "answer": answer, "site": site, "person": person},
                        validator={"kind": "contains", "required": [answer]},
                    )
                )
    return validate_complete_rows(TASK, rows, grounded_qa_capacity())


__all__ = ("grounded_qa_capacity", "render_grounded_qa_rows")
