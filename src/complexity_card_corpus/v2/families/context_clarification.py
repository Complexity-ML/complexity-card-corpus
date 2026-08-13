from __future__ import annotations

import hashlib
import json
from itertools import product

from ...variable_by import VariableBy2D
from ..contracts import RoleSeparatedVariableBy, SurfaceRole
from ..decks import V2RoleSeparatedDeck, V2SubcardPool


TASK = "context_clarification"
_AMBIGUITIES = (
    ("project_management", "the milestone brief", "the risk log", "revise"),
    ("project_management", "the launch checklist", "the staffing plan", "approve"),
    ("software", "the API contract", "the migration guide", "update"),
    ("software", "the test report", "the deployment note", "publish"),
    ("research", "the interview transcript", "the survey summary", "share"),
    ("research", "the methods appendix", "the findings table", "review"),
    ("education", "the lesson outline", "the assessment rubric", "simplify"),
    ("education", "the reading list", "the workshop handout", "translate"),
    ("healthcare_admin", "the intake form", "the referral letter", "forward"),
    ("healthcare_admin", "the rota draft", "the supply request", "confirm"),
    ("community", "the volunteer roster", "the venue request", "send"),
    ("community", "the event poster", "the accessibility note", "edit"),
    ("finance", "the expense forecast", "the invoice register", "reconcile"),
    ("finance", "the grant budget", "the purchase request", "sign off"),
    ("logistics", "the route sheet", "the packing manifest", "replace"),
    ("logistics", "the delivery schedule", "the warehouse map", "annotate"),
    ("publishing", "the author biography", "the cover brief", "shorten"),
    ("publishing", "the copyedited chapter", "the citation list", "return"),
    ("operations", "the maintenance ticket", "the inspection record", "close"),
    ("operations", "the shift summary", "the incident timeline", "archive"),
    ("legal_admin", "the consent draft", "the evidence index", "file"),
    ("legal_admin", "the hearing note", "the contact sheet", "correct"),
    ("design", "the mobile mockup", "the component inventory", "export"),
    ("design", "the color specification", "the usability memo", "circulate"),
    ("data", "the schema diagram", "the quality dashboard", "refresh"),
    ("data", "the labeling guide", "the anomaly report", "distribute"),
    ("facilities", "the floor plan", "the access register", "print"),
    ("facilities", "the repair estimate", "the safety checklist", "resubmit"),
    ("communications", "the press response", "the internal announcement", "rewrite"),
    ("communications", "the newsletter draft", "the briefing note", "schedule"),
    ("procurement", "the supplier comparison", "the order summary", "finalize"),
    ("procurement", "the tender notice", "the evaluation matrix", "release"),
)
_SETTINGS = (
    "the harbor office", "the north campus", "the riverside clinic",
    "the central library", "the field station", "the transit depot",
    "the neighborhood hub", "the research annex", "the market pavilion",
    "the training center", "the coastal warehouse", "the public garden",
    "the museum studio", "the repair workshop", "the mobile unit",
    "the school laboratory", "the community kitchen", "the archive room",
    "the regional branch", "the volunteer base", "the design lab",
    "the health kiosk", "the service garage", "the planning suite",
)
_STAKEHOLDERS = (
    "Amina", "Bruno", "Chloe", "Darius", "Elena", "Farah",
    "Gavin", "Hana", "Imani", "Jonah", "Keira", "Leo",
)
_PROMPTS = (
    "Ask the single clarification needed before acting: {scenario[request]}",
    "What should be clarified here? {scenario[request]}",
    "Respond without guessing the missing reference: {scenario[request]}",
    "Identify the ambiguity and ask a concise question: {scenario[request]}",
    "Handle this underspecified request safely: {scenario[request]}",
    "Request only the detail required to continue: {scenario[request]}",
    "Do not choose a document on the user's behalf. {scenario[request]}",
    "Clarify the intended target in this message: {scenario[request]}",
    "Give the next conversational turn for this ambiguous instruction: {scenario[request]}",
    "Resolve the unclear pronoun by asking the user: {scenario[request]}",
    "Before performing the action, ask what the user means: {scenario[request]}",
    "Reply with a focused clarification question: {scenario[request]}",
)
_ANSWERS = (
    "Do you mean {scenario[left]} or {scenario[right]}? Once you choose, I can {scenario[action]} it for {scenario[stakeholder]}.",
    "Which target should I use: {scenario[left]} or {scenario[right]}? I will {scenario[action]} the one {scenario[stakeholder]} intends.",
    "Should I {scenario[action]} {scenario[left]}, or are you referring to {scenario[right]}?",
    "I need the target before proceeding. Is {scenario[stakeholder]} referring to {scenario[left]} or {scenario[right]}?",
    "There are two possible documents here: {scenario[left]} and {scenario[right]}. Which one should I {scenario[action]}?",
    "Can you specify whether the request concerns {scenario[left]} or {scenario[right]}? Then I can {scenario[action]} the correct item.",
    "Which document did {scenario[stakeholder]} intend me to {scenario[action]}—{scenario[left]} or {scenario[right]}?",
    "I can continue after one detail: does “it” refer to {scenario[left]} or {scenario[right]}?",
    "Please choose between {scenario[left]} and {scenario[right]}; I do not want to {scenario[action]} the wrong document.",
    "Is the intended object {scenario[left]}, or should I work on {scenario[right]} for {scenario[stakeholder]}?",
    "Before I {scenario[action]} anything, which one is meant: {scenario[left]} or {scenario[right]}?",
    "Could you name the target—{scenario[left]} or {scenario[right]}—that {scenario[stakeholder]} wants handled?",
)


def context_clarification_capacity() -> int:
    return len(_AMBIGUITIES) * len(_SETTINGS) * len(_STAKEHOLDERS)


def _deck(
    domain: str,
    left: str,
    right: str,
    action: str,
    setting: str,
    stakeholder: str,
) -> V2RoleSeparatedDeck:
    request = (
        f"At {setting}, {stakeholder} wrote, ‘Please {action} it before the next "
        f"review.’ Both {left} and {right} are currently open."
    )
    contextual_answers = tuple(
        (
            "For {scenario[stakeholder]}'s work at {scenario[setting]}, "
            + answer[0].lower()
            + answer[1:]
        )
        for answer in _ANSWERS
    )
    variables = RoleSeparatedVariableBy(
        VariableBy2D(
            {
                "scenario": {
                    "domain": (domain,),
                    "left": (left,),
                    "right": (right,),
                    "action": (action,),
                    "setting": (setting,),
                    "stakeholder": (stakeholder,),
                    "request": (request,),
                },
                "prompt": {"clarification_request": _PROMPTS},
                "answer": {"target_question": contextual_answers},
            }
        )
    )
    return V2RoleSeparatedDeck(
        name=f"{TASK}:{domain}:{action}",
        variables=variables,
        prompt_pools=(
            V2SubcardPool(
                "clarification_request",
                SurfaceRole.PROMPT,
                ("{prompt[clarification_request]}",),
            ),
        ),
        answer_pools=(
            V2SubcardPool(
                "target_question",
                SurfaceRole.ANSWER,
                ("{answer[target_question]}",),
            ),
        ),
    )


def render_context_clarification_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain, left, right, action in _AMBIGUITIES:
        for setting, stakeholder in product(_SETTINGS, _STAKEHOLDERS):
            case_id = ":".join((domain, left, right, setting, stakeholder))
            deck = _deck(domain, left, right, action, setting, stakeholder)
            pair = deck.deal(case_id)
            rendered = f"User: {pair.prompt}\nAssistant: {pair.answer}"
            rows.append(
                {
                    "example_id": "v2:clarification:"
                    + hashlib.sha256(rendered.encode()).hexdigest()[:24],
                    "task": TASK,
                    "mode": "chat",
                    "difficulty": "easy",
                    "domain": domain,
                    "language": "en",
                    "split": "train",
                    "messages": [
                        {"role": "user", "content": pair.prompt},
                        {"role": "assistant", "content": pair.answer},
                    ],
                    "prompt": pair.prompt,
                    "response": pair.answer,
                    "reasoning_envelope": False,
                    "reasoning_trace": "",
                    "final_response": pair.answer,
                    "source_representation": json.dumps(
                        {
                            "case_id": case_id,
                            "facts": {
                                "domain": domain,
                                "left": left,
                                "right": right,
                                "action": action,
                                "setting": setting,
                                "stakeholder": stakeholder,
                            },
                            "prompt_subcards": pair.prompt_subcards,
                            "answer_subcards": pair.answer_subcards,
                            "variable_by": deck.variables.matrix.field_names(),
                            "deck_name": deck.name,
                            "variable_indices": pair.variable_indices,
                            "variable_card_counts": pair.variable_card_counts,
                            "dependency_graph": pair.dependency_graph,
                            "validator": {
                                "kind": "contains",
                                "required": [left, right],
                            },
                        },
                        sort_keys=True,
                    ),
                    "source": "AETHORIA-AI Card Corpus V2 authored decks",
                    "license": "CC BY-NC 4.0",
                    "version": "2.0.0",
                }
            )
    rows.sort(key=lambda row: str(row["example_id"]))
    if len(rows) != context_clarification_capacity():
        raise ValueError(f"{TASK} did not render its complete capacity")
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError(f"{TASK} produced duplicate example IDs")
    return rows


__all__ = (
    "context_clarification_capacity",
    "render_context_clarification_rows",
)
