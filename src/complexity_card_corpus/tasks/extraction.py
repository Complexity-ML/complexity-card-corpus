from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .core import (
    TaskHand,
    _code,
    _compose_subcards,
    _deal_task_frames,
    _number,
    _pick,
    _render_domain,
)


@dataclass(frozen=True)
class ExtractionRecord:
    fields: dict[str, Any]
    optional_key: str
    optional_value: Any
    category: str


def _record(row: dict[str, Any]) -> ExtractionRecord:
    """Deal one domain-grounded record from authored value reservoirs."""

    code = _code(row)
    amount = _number(f"amount:{code}", 12, 188)
    day = _number(f"extract-day:{code}", 2, 27)
    hour = _pick(
        f"extract-hour:{code}", ("07:20", "09:30", "11:45", "14:15", "16:40", "18:30")
    )
    person = _pick(
        f"person:{code}",
        (
            "Sam Iri",
            "Rin Vale",
            "Maya Chen",
            "Omar Bell",
            "Lea North",
            "Theo Marin",
            "Nia Sol",
            "Jon Reed",
        ),
    )
    organization = _pick(
        f"organization:{code}",
        (
            "North Review",
            "Cedar Works",
            "Harbor Lab",
            "Field Commons",
            "Lumen Press",
            "Westmere Studio",
        ),
    )
    city = _pick(
        f"city:{code}",
        ("Westmere", "Lumenport", "Cedar Bay", "Northbridge", "Arden", "Rivermark"),
    )

    cases: dict[str, ExtractionRecord] = {
        "receipt": ExtractionRecord(
            {
                "merchant": _pick(
                    f"merchant:{code}",
                    (
                        "North Market",
                        "Cedar Books",
                        "Harbor Grocer",
                        "Lumen Hardware",
                        "Field Bakery",
                        "Arden Pharmacy",
                    ),
                ),
                "date": f"2026-08-{day:02d}",
                "total": f"{amount}.50 USD",
                "tax": f"{_number(f'tax:{code}', 1, 14)}.00 USD",
                "cashier": None,
            },
            "cashier",
            person,
            "purchase",
        ),
        "event_listing": ExtractionRecord(
            {
                "title": _pick(
                    f"event:{code}",
                    (
                        "Open Lab",
                        "Repair Evening",
                        "Methods Workshop",
                        "Public Archive Tour",
                        "Field Notes Forum",
                        "Community Studio",
                    ),
                ),
                "date": f"2026-08-{day:02d}",
                "venue": f"{_pick(f'venue:{code}', ('Room', 'Hall', 'Studio', 'Gallery'))} {amount}",
                "starts": hour,
                "eligibility": None,
            },
            "eligibility",
            _pick(
                f"eligibility:{code}",
                ("open to all", "members", "ages 16+", "registered guests"),
            ),
            "event",
        ),
        "contact_record": ExtractionRecord(
            {
                "name": person,
                "role": _pick(
                    f"role:{code}",
                    (
                        "Editor",
                        "Coordinator",
                        "Engineer",
                        "Researcher",
                        "Archivist",
                        "Producer",
                    ),
                ),
                "organization": organization,
                "email": f"{person.split()[0].lower()}.{code.lower()}@example.org",
                "phone": None,
            },
            "phone",
            f"+44 20 7{amount:03d} {day:04d}",
            "contact",
        ),
        "issue_ticket": ExtractionRecord(
            {
                "ticket": code,
                "environment": _pick(
                    f"environment:{code}",
                    ("Linux", "macOS", "Windows", "Android", "iOS", "web"),
                ),
                "severity": _pick(
                    f"severity:{code}", ("low", "medium", "high", "critical")
                ),
                "status": _pick(
                    f"ticket-status:{code}",
                    ("pending", "triaged", "reproduced", "blocked", "resolved"),
                ),
                "owner": None,
            },
            "owner",
            person,
            "software_issue",
        ),
        "survey_response": ExtractionRecord(
            {
                "response": code,
                "rating": _number(f"rating:{code}", 1, 5),
                "topic": _pick(
                    f"topic:{code}",
                    (
                        "navigation",
                        "search",
                        "onboarding",
                        "accessibility",
                        "billing",
                        "performance",
                    ),
                ),
                "comment": _pick(
                    f"comment:{code}",
                    (
                        "clear after retry",
                        "easy to locate",
                        "slow on mobile",
                        "missing one label",
                        "worked as expected",
                        "needs a shorter path",
                    ),
                ),
                "follow_up": None,
            },
            "follow_up",
            _pick(
                f"follow-up:{code}",
                ("email requested", "interview accepted", "no contact needed"),
            ),
            "feedback",
        ),
        "inventory_record": ExtractionRecord(
            {
                "item": code,
                "quantity": amount,
                "location": f"{_pick(f'zone:{code}', ('A', 'B', 'C', 'Cold', 'Secure'))}-{day}",
                "condition": _pick(
                    f"condition:{code}",
                    ("good", "sealed", "damaged", "reserved", "inspection_due"),
                ),
                "checked_by": None,
            },
            "checked_by",
            person,
            "inventory",
        ),
        "schedule_entry": ExtractionRecord(
            {
                "event": f"{_pick(f'schedule-kind:{code}', ('Review', 'Planning', 'Interview', 'Maintenance', 'Workshop'))} {code}",
                "date": f"2026-08-{day:02d}",
                "starts": hour,
                "duration_minutes": _pick(f"duration:{code}", (25, 30, 45, 60, 90)),
                "room": None,
            },
            "room",
            f"Room {_number(f'room:{code}', 101, 480)}",
            "schedule",
        ),
        "case_note": ExtractionRecord(
            {
                "case": code,
                "observed": _pick(
                    f"observed:{code}",
                    (
                        "package sealed",
                        "account locked",
                        "sensor offline",
                        "document unsigned",
                        "item photographed",
                    ),
                ),
                "reported": _pick(
                    f"reported:{code}",
                    (
                        "item incomplete",
                        "login rejected",
                        "reading unstable",
                        "page missing",
                        "delivery delayed",
                    ),
                ),
                "action": _pick(
                    f"case-action:{code}",
                    (
                        "photographs retained",
                        "audit log saved",
                        "device isolated",
                        "copy preserved",
                        "carrier contacted",
                    ),
                ),
                "next_owner": None,
            },
            "next_owner",
            person,
            "case_note",
        ),
        "address_record": ExtractionRecord(
            {
                "recipient": person,
                "street": f"{amount} {_pick(f'street:{code}', ('Cedar Lane', 'Harbor Road', 'Field Street', 'Lumen Way', 'North Avenue'))}",
                "locality": city,
                "postal_code": f"{day}804",
                "country": _pick(f"country:{code}", ("GB", "FR", "DE", "CA", "IE")),
                "delivery_note": None,
            },
            "delivery_note",
            _pick(
                f"delivery-note:{code}",
                (
                    "side entrance",
                    "reception desk",
                    "call on arrival",
                    "leave with concierge",
                ),
            ),
            "address",
        ),
        "booking_record": ExtractionRecord(
            {
                "reference": code,
                "service": _pick(
                    f"service:{code}",
                    (
                        "Harbor tour",
                        "Rail transfer",
                        "Studio visit",
                        "Museum entry",
                        "Workshop seat",
                        "Equipment rental",
                    ),
                ),
                "date": f"2026-08-{day:02d}",
                "starts": hour,
                "party_size": _number(f"party:{code}", 1, 8),
                "cancellation_status": None,
            },
            "cancellation_status",
            _pick(f"cancel:{code}", ("refundable", "non_refundable", "free_until_24h")),
            "booking",
        ),
        "product_record": ExtractionRecord(
            {
                "sku": code,
                "product": _pick(
                    f"product:{code}",
                    (
                        "Desk lamp",
                        "Travel kettle",
                        "Field recorder",
                        "Repair kit",
                        "Reading stand",
                        "USB hub",
                    ),
                ),
                "variant": _pick(
                    f"variant:{code}",
                    (
                        "blue",
                        "graphite",
                        "small",
                        "recycled",
                        "EU plug",
                        "second edition",
                    ),
                ),
                "price": f"{amount}.00 USD",
                "availability": _pick(
                    f"availability:{code}",
                    ("in_stock", "preorder", "backorder", "limited", "discontinued"),
                ),
                "warranty": None,
            },
            "warranty",
            _pick(f"warranty:{code}", ("12 months", "24 months", "repair only")),
            "product",
        ),
        "expense_record": ExtractionRecord(
            {
                "expense": code,
                "category": _pick(
                    f"expense-category:{code}",
                    (
                        "local travel",
                        "equipment",
                        "lodging",
                        "printing",
                        "training",
                        "meals",
                    ),
                ),
                "amount": f"{amount}.25 {_pick(f'currency:{code}', ('EUR', 'USD', 'GBP', 'CAD'))}",
                "date": f"2026-08-{day:02d}",
                "payment_method": _pick(
                    f"payment:{code}", ("card", "cash", "transfer", "voucher")
                ),
                "approval": None,
            },
            "approval",
            _pick(f"approval:{code}", ("approved", "pending", "rejected")),
            "expense",
        ),
        "shipment_record": ExtractionRecord(
            {
                "tracking": code,
                "carrier": _pick(
                    f"carrier:{code}",
                    (
                        "North Parcel",
                        "Harbor Express",
                        "Cedar Freight",
                        "Field Courier",
                        "Lumen Post",
                    ),
                ),
                "destination": city,
                "milestone": _pick(
                    f"milestone:{code}",
                    (
                        "sorting_center",
                        "customs",
                        "out_for_delivery",
                        "delivered",
                        "held",
                    ),
                ),
                "timestamp": f"2026-08-{day:02d}T{hour}:00Z",
                "exception": None,
            },
            "exception",
            _pick(
                f"exception:{code}",
                ("weather_delay", "address_check", "customs_review"),
            ),
            "shipment",
        ),
        "calendar_record": ExtractionRecord(
            {
                "event": f"{_pick(f'calendar-kind:{code}', ('Planning', 'Review', 'Demo', 'Interview', 'Training'))} {code}",
                "organizer": person,
                "participants": list(
                    _pick(
                        f"participants:{code}",
                        (
                            ("Jon", "Lea"),
                            ("Maya", "Theo"),
                            ("Nia", "Omar"),
                            ("Sam", "Rin"),
                        ),
                    )
                ),
                "starts": f"2026-08-{day:02d}T{hour}",
                "ends": f"2026-08-{day:02d}T{_pick(f'end:{code}', ('10:15', '12:30', '15:00', '17:25', '19:15'))}",
                "location": None,
                "response": _pick(
                    f"calendar-response:{code}",
                    ("tentative", "accepted", "declined", "needs_action"),
                ),
            },
            "location",
            f"Room {_number(f'calendar-room:{code}', 101, 480)}",
            "calendar",
        ),
        "measurement_record": ExtractionRecord(
            {
                "sample": code,
                "value": amount + 0.4,
                "unit": _pick(f"unit:{code}", ("cm", "mm", "g", "mL", "°C", "kPa")),
                "instrument": _pick(
                    f"instrument:{code}",
                    ("caliper-2", "scale-4", "probe-7", "meter-3", "sensor-8"),
                ),
                "timestamp": f"2026-08-{day:02d}T{hour}:00Z",
                "tolerance": _pick(f"tolerance:{code}", ("0.2", "0.5", "1.0", "2%")),
                "observer": None,
            },
            "observer",
            person,
            "measurement",
        ),
        "bibliographic_record": ExtractionRecord(
            {
                "title": _pick(
                    f"title:{code}",
                    (
                        "Bounded Systems",
                        "Field Methods",
                        "Repair Cultures",
                        "Archive Design",
                        "Measured Change",
                        "Public Interfaces",
                    ),
                ),
                "author": _pick(
                    f"author:{code}",
                    ("I. North", "M. Chen", "O. Bell", "L. Vale", "T. Marin"),
                ),
                "year": _number(f"year:{code}", 1998, 2026),
                "venue": _pick(
                    f"bib-venue:{code}",
                    (
                        "Field Notes",
                        "Systems Review",
                        "Methods Quarterly",
                        "Open Archive",
                        "Design Studies",
                    ),
                ),
                "identifier": code,
                "pages": None,
            },
            "pages",
            f"{_number(f'page-start:{code}', 1, 80)}-{_number(f'page-end:{code}', 81, 180)}",
            "publication",
        ),
        "shipment_manifest": ExtractionRecord(
            {
                "manifest": code,
                "carrier": _pick(
                    f"freight-carrier:{code}",
                    (
                        "North Freight",
                        "Cedar Cargo",
                        "Harbor Logistics",
                        "Field Transit",
                    ),
                ),
                "origin": _pick(
                    f"origin:{code}", ("Bay 3", "Depot 7", "Dock 2", "Warehouse 5")
                ),
                "destination": city,
                "packages": amount,
                "seal": _pick(
                    f"seal:{code}", ("verified", "pending", "broken", "not_required")
                ),
                "customs_note": None,
            },
            "customs_note",
            _pick(
                f"customs:{code}",
                ("cleared", "inspection_required", "documents_requested"),
            ),
            "manifest",
        ),
        "survey_export": ExtractionRecord(
            {
                "export": code,
                "responses": amount,
                "mean_rating": _pick(f"mean-rating:{code}", (2.8, 3.4, 4.0, 4.2, 4.7)),
                "completion_rate": f"{_number(f'completion:{code}', 62, 98)}%",
                "segment": _pick(
                    f"segment:{code}",
                    (
                        "trial_users",
                        "returning_members",
                        "mobile_visitors",
                        "educators",
                        "administrators",
                    ),
                ),
                "notes": None,
            },
            "notes",
            _pick(
                f"survey-note:{code}",
                ("partial export", "weighted sample", "duplicates removed"),
            ),
            "survey_export",
        ),
        "api_response": ExtractionRecord(
            {
                "request_id": code,
                "status": _pick(
                    f"http-status:{code}", (200, 201, 202, 400, 404, 409, 503)
                ),
                "duration_ms": amount,
                "records": day,
                "next_cursor": None,
                "error": None,
            },
            "next_cursor",
            f"cursor_{code.lower()}",
            "api_result",
        ),
        "administrative_form": ExtractionRecord(
            {
                "form": code,
                "applicant": person,
                "submitted": f"2026-08-{day:02d}",
                "department": _pick(
                    f"department:{code}",
                    (
                        "Permits",
                        "Archives",
                        "Licensing",
                        "Community Services",
                        "Procurement",
                    ),
                ),
                "status": _pick(
                    f"form-status:{code}",
                    ("received", "under_review", "incomplete", "approved", "returned"),
                ),
                "reviewer": None,
            },
            "reviewer",
            person,
            "administrative_form",
        ),
    }
    return cases[_render_domain(row)]


def _raw_value(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, list):
        return "[" + ", ".join(map(str, value)) + "]"
    return str(value)


def _answer_payload(
    row: dict[str, Any], record: ExtractionRecord
) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    code = _code(row)
    fields = {"record_id": code, **record.fields}
    state = row["state"].lower()
    if "absent" not in state and "missing" not in state:
        fields[record.optional_key] = record.optional_value

    missing = [key for key, value in fields.items() if value is None]
    present = [key for key, value in fields.items() if value is not None]
    evidence_items = [(key, fields[key]) for key in present if key != "record_id"][:3]
    evidence = {key: value for key, value in evidence_items}
    intent = row["intent"]
    requested = ", ".join(fields)

    if intent == "classify":
        labels = [record.category, "other", "needs_review"]
        selected = "needs_review" if "ambiguous" in state else record.category
        payload = {
            "record_id": code,
            "label": selected,
            "label_set": labels,
            "completeness": "partial" if missing else "complete",
            "basis": [f"{key}={_raw_value(value)}" for key, value in evidence_items],
        }
        goal = (
            f"Classify this record as one of {', '.join(labels)}. Return JSON with "
            "record_id, label, label_set, completeness, and basis."
        )
        contract = ("json", "defined_label", "evidence_basis")
    elif intent == "missing":
        payload = {
            "record_id": code,
            "record_type": record.category,
            "missing_fields": missing,
            "present_fields": present,
            "present_sample": evidence,
            "next_action": (
                f"request {missing[0]}"
                if missing
                else f"validate the completed {record.category} record"
            ),
        }
        goal = (
            "Identify missing required fields. Return JSON with record_id, record_type, "
            "missing_fields, present_fields, present_sample, and next_action."
        )
        contract = ("json", "missing_fields", "next_action")
    elif intent == "standardize_records":
        payload = {
            "record_id": code,
            "record_type": record.category,
            "normalized_record": fields,
            "conflicts": [],
        }
        goal = (
            "Standardize the source into JSON with record_id, record_type, "
            "normalized_record, and conflicts. Preserve null values."
        )
        contract = ("json", "normalized_record", "conflicts")
    else:
        payload = fields
        verb = {
            "extract": "Extract",
            "normalize": "Normalize",
            "structure": "Structure",
        }.get(intent, "Extract")
        goal = f"{verb} {requested} as JSON. Use null for an absent value."
        contract = ("json", "requested_fields", "missing_is_null")
    return payload, goal, contract


def render_extraction(row: dict[str, Any], variant: int) -> TaskHand:
    record = _record(row)
    payload, requested_goal, contract = _answer_payload(row, record)
    fields = {"record_id": _code(row), **record.fields}
    state = row["state"].lower()
    if "absent" not in state and "missing" not in state:
        fields[record.optional_key] = record.optional_value
    raw = "; ".join(f"{key}={_raw_value(value)}" for key, value in fields.items())
    if "mixed" in state:
        raw += "; unrelated_note=copy order is not evidence"

    record_label = row["domain"].replace("_", " ")
    if not record_label.endswith("record"):
        record_label = f"{record_label} record"
    data, goal = _deal_task_frames(
        row,
        variant,
        "extraction",
        (
            f"Raw {record_label}: {raw}.",
            f"Unnormalized {record_label}: {raw}.",
            f"Source fields for this {record_label}: {raw}.",
        ),
        (
            requested_goal,
            f"Required output: {requested_goal}",
            f"Use only the supplied record. {requested_goal}",
        ),
    )
    answer = _compose_subcards(
        row,
        variant,
        "extraction-json-layout",
        (
            (
                json.dumps(payload, separators=(",", ":")),
                json.dumps(payload),
                json.dumps(payload, separators=(",", ": "), sort_keys=True),
            ),
        ),
        pool_names=("json_layout",),
        cycle_first_pool=True,
    )
    return TaskHand(data, goal, answer, contract)
