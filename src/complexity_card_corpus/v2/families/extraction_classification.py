from __future__ import annotations

import hashlib
import json
from itertools import product

from ...variable_by import VariableBy2D
from ..contracts import RoleSeparatedVariableBy, SurfaceRole
from ..decks import V2RoleSeparatedDeck, V2SubcardPool


TASK = "extraction_classification"
_CASES = (
    ("access", "identity", "high", "Security", "a staff badge no longer opens the secure entrance"),
    ("access", "identity", "medium", "Accounts", "a new volunteer cannot activate their portal login"),
    ("equipment", "hardware", "high", "Facilities", "a refrigeration sensor is reporting impossible temperatures"),
    ("equipment", "hardware", "medium", "Repair", "a shared tablet shuts down after a few minutes"),
    ("delivery", "logistics", "high", "Dispatch", "a medical parcel has not reached its scheduled stop"),
    ("delivery", "logistics", "low", "Warehouse", "a box arrived with an outdated location label"),
    ("data", "information", "high", "Data", "a daily dashboard omits the latest inspection records"),
    ("data", "information", "medium", "Analytics", "a report displays duplicate survey responses"),
    ("billing", "finance", "high", "Finance", "an invoice contains a charge for an unreceived order"),
    ("billing", "finance", "low", "Procurement", "a receipt uses the previous department name"),
    ("scheduling", "operations", "medium", "Planning", "two confirmed workshops occupy the same room"),
    ("scheduling", "operations", "low", "Coordination", "a calendar invitation shows the wrong time zone"),
)
_SITES = (
    "Harbor Office", "North Campus", "Riverside Clinic", "Central Library",
    "Field Station", "Transit Depot", "Neighborhood Hub", "Research Annex",
    "Market Pavilion", "Training Center", "Coastal Warehouse", "Public Garden",
    "Museum Studio", "Repair Workshop", "Mobile Unit", "School Laboratory",
    "Community Kitchen", "Archive Room", "Regional Branch", "Volunteer Base",
    "Design Lab", "Health Kiosk", "Service Garage", "Planning Suite",
)
_REQUESTERS = (
    "Amina Cole", "Bruno Diaz", "Chloe Evans", "Darius Ford",
    "Elena Grant", "Farah Hall", "Gavin Ito", "Hana Jones",
    "Imani Khan", "Jonah Li", "Keira Moss", "Leo Novak",
)
_PROMPTS = (
    "Extract the ticket fields and return only JSON: {scenario[ticket]}",
    "Classify this service report as structured JSON: {scenario[ticket]}",
    "Read the report and emit its category, domain, priority, owner, requester, and site in JSON: {scenario[ticket]}",
    "Convert this support message into the required JSON record: {scenario[ticket]}",
    "Return a machine-readable classification for this ticket: {scenario[ticket]}",
    "Identify the routing fields in this report and answer with JSON only: {scenario[ticket]}",
)


def extraction_classification_capacity() -> int:
    return len(_CASES) * len(_SITES) * len(_REQUESTERS)


def render_extraction_classification_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (category, domain, priority, owner, detail), site, requester in product(
        _CASES, _SITES, _REQUESTERS
    ):
        ticket = (
            f"Requester {requester} at {site} reports that {detail}. "
            f"Route the issue to {owner} with {priority} priority."
        )
        expected = {
            "category": category,
            "domain": domain,
            "owner": owner,
            "priority": priority,
            "requester": requester,
            "site": site,
        }
        answer = json.dumps(expected, sort_keys=True, separators=(",", ":"))
        answer_template = answer.replace("{", "{{").replace("}", "}}")
        variables = RoleSeparatedVariableBy(
            VariableBy2D(
                {
                    "scenario": {
                        "ticket": (ticket,),
                        "category": (category,),
                        "domain": (domain,),
                        "priority": (priority,),
                        "owner": (owner,),
                        "requester": (requester,),
                        "site": (site,),
                    },
                    "prompt": {"classification_request": _PROMPTS},
                    "answer": {"json_record": (answer_template,)},
                }
            )
        )
        deck = V2RoleSeparatedDeck(
            name=f"{TASK}:{domain}:{category}",
            variables=variables,
            prompt_pools=(
                V2SubcardPool(
                    "classification_request",
                    SurfaceRole.PROMPT,
                    ("{prompt[classification_request]}",),
                ),
            ),
            answer_pools=(
                V2SubcardPool(
                    "json_record",
                    SurfaceRole.ANSWER,
                    ("{answer[json_record]}",),
                ),
            ),
        )
        case_id = ":".join((category, site, requester))
        pair = deck.deal(case_id)
        rendered = f"User: {pair.prompt}\nAssistant: {pair.answer}"
        rows.append(
            {
                "example_id": "v2:extraction:"
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
                        "facts": expected,
                        "prompt_subcards": pair.prompt_subcards,
                        "answer_subcards": pair.answer_subcards,
                        "variable_by": deck.variables.matrix.field_names(),
                        "deck_name": deck.name,
                        "variable_indices": pair.variable_indices,
                        "variable_card_counts": pair.variable_card_counts,
                        "dependency_graph": pair.dependency_graph,
                        "validator": {"kind": "json_equal", "expected": expected},
                    },
                    sort_keys=True,
                ),
                "source": "AETHORIA-AI Card Corpus V2 authored decks",
                "license": "CC BY-NC 4.0",
                "version": "2.0.0",
            }
        )
    rows.sort(key=lambda row: str(row["example_id"]))
    if len(rows) != extraction_classification_capacity():
        raise ValueError(f"{TASK} did not render its complete capacity")
    return rows


__all__ = (
    "extraction_classification_capacity",
    "render_extraction_classification_rows",
)
