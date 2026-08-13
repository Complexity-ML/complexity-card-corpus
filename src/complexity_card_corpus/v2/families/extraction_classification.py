from __future__ import annotations

import json
from itertools import product

from ...variable_by import VariableBy2D
from ..contracts import RoleSeparatedVariableBy, SurfaceRole
from ..decks import V2RoleSeparatedDeck, V2SubcardPool, prompt_variant_plans
from ._common import render_v2_row, validate_complete_rows


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
_PROMPT_FUNCTIONS = (
    ("request_extraction", "require_json_only"),
    ("request_classification", "require_json"),
    ("enumerate_fields", "require_json"),
    ("request_record_conversion", "require_json"),
    ("request_machine_readable_classification",),
    ("request_routing_fields", "require_json_only"),
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
            prompt_plans=prompt_variant_plans(
                sense="classification_request",
                pool_name="classification_request",
                functions=_PROMPT_FUNCTIONS,
            ),
        )
        case_id = ":".join((category, site, requester))
        rows.append(
            render_v2_row(
                task=TASK,
                case_id=case_id,
                domain=domain,
                difficulty="easy",
                deck=deck,
                facts=expected,
                validator={"kind": "json_equal", "expected": expected},
            )
        )
    return validate_complete_rows(TASK, rows, extraction_classification_capacity())


__all__ = (
    "extraction_classification_capacity",
    "render_extraction_classification_rows",
)
