from __future__ import annotations

import json
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


def _grounded_qa(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    year = _number(f"year:{code}", 2014, 2022)
    battery_hours = _number(f"battery:{code}", 7, 18)
    return_days = _pick(f"return-days:{code}", ("14", "21", "30", "45"))
    exposure_hours = _number(f"exposure-hours:{code}", 3, 9)
    temperature_change = _number(f"temperature-change:{code}", 2, 7)
    owner = _pick(
        f"project-owner:{code}", ("Nia", "Omar", "Maya", "Theo", "Lea", "Sam")
    )
    delivery_day = _number(f"project-day:{code}", 12, 26)
    train_number = _number(f"train-number:{code}", 180, 780)
    departure_hour = _number(f"departure-hour:{code}", 6, 19)
    departure_minute = _pick(f"departure-minute:{code}", ("05", "20", "35", "50"))
    platform = _number(f"platform:{code}", 1, 12)
    python_minor = _pick(f"python-minor:{code}", ("10", "11", "12", "13"))
    release_major = _number(f"release-major:{code}", 2, 6)
    release_minor = _number(f"release-minor:{code}", 0, 9)
    longest_battery = _number(f"table-battery:{code}", 8, 14)
    other_battery = max(4, longest_battery - _number(f"table-gap:{code}", 1, 4))
    status_minute = _pick(f"status-minute:{code}", ("05", "10", "20", "35"))
    ticket_minute = f"{(int(status_minute) + _number(f'ticket-gap:{code}', 1, 4)):02d}"
    available_region = _pick(f"available-region:{code}", ("EU", "US", "Asia-Pacific"))
    ticket_region = _pick(
        f"ticket-region:{code}",
        tuple(
            region
            for region in ("EU", "US", "Asia-Pacific")
            if region != available_region
        ),
    )
    failed_operation = _pick(
        f"failed-operation:{code}",
        ("sign in", "upload a file", "open the dashboard", "submit a request"),
    )
    event_day = _number(f"event-day:{code}", 3, 27)
    event_room = _number(f"event-room:{code}", 101, 418)
    energy_kwh = _number(f"energy-kwh:{code}", 180, 640)
    energy_rate = _number(f"energy-rate:{code}", 17, 34)
    course_number = _number(f"course-number:{code}", 110, 480)
    maintenance_day = _number(f"maintenance-day:{code}", 2, 28)
    sensor_count = _number(f"sensor-count:{code}", 3, 12)
    measured_value = _number(f"measured-value:{code}", 18, 86)
    notice_days = _number(f"notice-days:{code}", 10, 45)
    sample_count = _number(f"lab-samples:{code}", 6, 24)
    ph_value = _number(f"lab-ph-whole:{code}", 6, 8)
    quote_units = _number(f"quote-units:{code}", 12, 80)
    quote_price = _number(f"quote-price:{code}", 18, 95)
    tested_pages = _number(f"accessibility-pages:{code}", 8, 30)
    operating_limit = _number(f"equipment-limit:{code}", 30, 75)
    cases = {
        "product_specs": (
            f"The Lumen Mini supports Wi-Fi 6 and USB-C charging. Its rated battery life is {battery_hours} hours. No water-resistance rating is listed.",
            "State the rated battery life and whether water resistance is documented.",
            f"The rated battery life is {battery_hours} hours. A water-resistance rating is unknown because the specification does not list one.",
        ),
        "policy_excerpt": (
            f"Returns are accepted within {return_days} days with proof of purchase. Opened safety equipment is excluded. The text gives no holiday extension.",
            "State the ordinary return window and whether a holiday extension is defined.",
            f"The ordinary return window is {return_days} days with proof of purchase. A holiday extension is unknown because the policy does not define one.",
        ),
        "science_passage": (
            f"A {year} trial exposed identical samples to light for {exposure_hours} hours. The treated sample warmed by {temperature_change}°C. The passage does not identify the molecular mechanism.",
            "State the observed temperature change and whether the mechanism is established.",
            f"The treated sample warmed by {temperature_change}°C. The molecular mechanism is unknown because the passage reports no mechanism.",
        ),
        "historical_note": (
            f"The archive records that the bridge opened in {year} under mayor Elena Voss. It does not name the original architect.",
            "State the opening year and whether the architect is identified.",
            f"The bridge opened in {year}. The original architect is unknown because the note does not name one.",
        ),
        "project_brief": (
            f"The brief assigns the prototype to {owner} and sets delivery for day {delivery_day}. Hosting approval remains pending, and no approver is named.",
            "State the prototype owner and whether the hosting approver is known.",
            f"{owner} owns the prototype. The hosting approver is unknown because the brief names none.",
        ),
        "travel_information": (
            f"Train {train_number} departs at {departure_hour:02d}:{departure_minute} from platform {platform}. Bicycles require a reservation. The notice gives no information about onboard meals.",
            "State the departure details and whether meal service is documented.",
            f"Train {train_number} departs at {departure_hour:02d}:{departure_minute} from platform {platform}. Meal service is unknown because the notice does not mention it.",
        ),
        "technical_documentation": (
            f"Version {release_major}.{release_minor} requires Python 3.{python_minor} and supports Linux arm64. Offline activation is not described in this excerpt.",
            "State the Python requirement and whether offline activation is supported by the excerpt.",
            f"The requirement is Python 3.{python_minor}. Offline activation is unknown because the excerpt does not describe it.",
        ),
        "comparison_table": (
            f"Table: Cedar—$48, {longest_battery} hours, repairable yes; Flint—$42, {other_battery} hours, repairable no; Vale—$45, battery value missing, repairable yes.",
            "Identify the longest stated battery life and whether Vale's battery life can be compared.",
            f"Cedar has the longest stated battery life at {longest_battery} hours. Vale's battery life is unknown, so it cannot be compared on that field.",
        ),
        "conflicting_service_reports": (
            f"At 09:{status_minute}, the public status check reports that the {available_region} service endpoint is available. At 09:{ticket_minute}, a support ticket reports that one {ticket_region} account cannot {failed_operation}. The reports cover different regions, scopes, times, and operations.",
            "Explain what the two reports establish, what remains unknown, and the next direct verification step.",
            f"The reports appear to conflict, but they describe different scopes and therefore do not establish one global service state; that remains unknown. Do not choose either report as universally correct. Compare the same time window, region, account scope, and operation, then reproduce the attempt to {failed_operation} with a direct check.",
        ),
        "public_event_notice": (
            f"The Open Methods workshop starts on August {event_day} at 18:30 in Room {event_room}. Step-free access is available. The notice does not say whether advance registration is required.",
            "State the workshop time and venue, and whether advance registration is required.",
            f"The workshop starts on August {event_day} at 18:30 in Room {event_room}. Advance registration is unknown because the notice does not state whether it is required.",
        ),
        "energy_bill": (
            f"The statement covers 30 days and records {energy_kwh} kWh at {energy_rate} cents per kWh before fixed charges. It lists no rebate or credit.",
            "State the recorded usage and unit rate, and whether a rebate is documented.",
            f"The statement records {energy_kwh} kWh at {energy_rate} cents per kWh before fixed charges. A rebate is unknown because none is documented.",
        ),
        "course_catalog": (
            f"Course CS-{course_number} meets on Tuesdays and requires Introductory Programming. The entry describes weekly labs but does not identify the final assessment format.",
            "State the prerequisite and whether the final assessment format is documented.",
            f"The prerequisite for CS-{course_number} is Introductory Programming. The final assessment format is unknown because the entry does not identify it.",
        ),
        "maintenance_log": (
            f"On August {maintenance_day}, a technician replaced the intake filter after recording reduced airflow. A follow-up test restored normal flow. The log does not establish the original cause of the blockage.",
            "State the maintenance action and test result, and whether the original cause is established.",
            "The intake filter was replaced and the follow-up test restored normal airflow. The original cause of the blockage is unknown because the log does not establish it.",
        ),
        "environmental_report": (
            f"At the north site, {sensor_count} sensors recorded a median value of {measured_value} units during the survey window. The report provides no pre-survey baseline.",
            "State the reported measurement and whether change from baseline can be calculated.",
            f"The north site had a median measurement of {measured_value} units across {sensor_count} sensors. Change from baseline is unknown because no pre-survey baseline is provided.",
        ),
        "software_release_note": (
            f"Release {release_major}.{release_minor} adds export filters and fixes duplicate notifications on Linux. A legacy import option is marked deprecated, but no removal date is given.",
            "State the added behavior and whether the legacy import removal date is known.",
            f"Release {release_major}.{release_minor} adds export filters. The legacy import removal date is unknown because the note gives no date.",
        ),
        "contract_clause": (
            f"Clause 8 requires {notice_days} calendar days of written notice before termination. Notices must be sent to the registered office. The clause does not identify an arbitration venue.",
            "State the notice period and whether an arbitration venue is identified.",
            f"The notice period is {notice_days} calendar days in writing. The arbitration venue is unknown because Clause 8 does not identify one.",
        ),
        "lab_report": (
            f"The laboratory tested {sample_count} samples with the same calibrated probe and recorded a median pH of {ph_value}. The report does not state when the probe was last calibrated.",
            "State the median pH and whether the last calibration date is documented.",
            f"The median pH is {ph_value} across {sample_count} samples. The last calibration date is unknown because the report does not state it.",
        ),
        "procurement_quote": (
            f"Quote Q-{code} offers {quote_units} units at ${quote_price} each and remains valid for 21 days. Tax is included, but shipping time is not listed.",
            "State the quoted quantity and unit price, and whether shipping time is documented.",
            f"The quote covers {quote_units} units at ${quote_price} each. Shipping time is unknown because the quote does not list it.",
        ),
        "accessibility_statement": (
            f"The accessibility statement reports a WCAG 2.2 AA review of {tested_pages} public pages and names keyboard navigation as tested. It gives no remediation date for remaining issues.",
            "State the reported conformance target and tested scope, and whether a remediation date is documented.",
            f"The reported target is WCAG 2.2 AA across {tested_pages} public pages. A remediation date is unknown because the statement gives none.",
        ),
        "equipment_manual": (
            f"The manual permits continuous operation below {operating_limit}°C in supervised mode and requires a five-minute cooldown after an overload warning. Remote control is not described.",
            "State the operating limit and cooldown requirement, and whether remote control is documented.",
            f"The operating limit is below {operating_limit}°C, with a five-minute cooldown after an overload warning. Remote control is unknown because the manual does not describe it.",
        ),
    }
    passage, requested_answer, supported = cases[_render_domain(row)]
    data, goal = _deal_task_frames(
        row,
        variant,
        "grounded",
        (
            f"Source {code}: {passage}",
            f"Evidence excerpt {code}: {passage}",
            f"Use this supplied record only — {code}: {passage}",
        ),
        (
            requested_answer,
            f"Using only Source {code}, {requested_answer[:1].lower() + requested_answer[1:]}",
            f"Answer the documented parts of this request and mark the rest unknown: {requested_answer}",
        ),
    )
    answer = _compose_subcards(
        row,
        variant,
        "grounded-answer",
        (
            (
                f"Based on Source {code}:",
                f"Source {code} supports this answer:",
                "The documented answer is:",
                f"According to Source {code}:",
            ),
            (
                supported,
                f"Supported facts: {supported}",
                f"The supplied record establishes this: {supported}",
            ),
            (
                f"The answer remains limited to Source {code}.",
                "No unstated detail is inferred.",
                "Anything not documented remains unknown.",
            ),
        ),
        pool_names=("evidence_scope", "supported_answer", "unknown_boundary"),
    )
    subject = row["domain"].replace("_", " ").title()
    return TaskHand(
        data,
        goal,
        answer,
        ("direct_answer", "evidence", "unknown"),
        situation_title=f"{subject} — answer from the supplied source",
        situation=(
            "The supplied source answers the documented part of the request and leaves "
            "one requested field undocumented."
        ),
        rule=(
            "Use only the supplied source. Mark any requested field that the source does "
            "not document as unknown."
        ),
    )


def _summary(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    owner = _pick(f"summary-owner:{code}", ("Mina", "Paul", "Sora", "Theo", "Lina"))
    day = _number(f"summary-day:{code}", 12, 27)
    cases = {
        "meeting_transcript": (
            "approve the interface copy",
            "run two accessibility checks",
            "release timing",
        ),
        "research_notes": (
            "retain the observed temperature result",
            "replicate two uncertain measurements",
            "the causal explanation",
        ),
        "support_thread": (
            "keep the case open",
            "test two account-recovery paths",
            "whether the issue is device-specific",
        ),
        "project_update": (
            "accept the completed prototype",
            "finish two integration checks",
            "the launch date",
        ),
        "policy_memo": (
            "adopt the revised access rule",
            "document two listed exceptions",
            "the enforcement start date",
        ),
        "article_excerpt": (
            "retain the article's central claim",
            "verify two cited examples",
            "whether the pattern generalizes",
        ),
        "incident_log": (
            "keep the service in monitored recovery",
            "inspect two remaining error sources",
            "the incident's root cause",
        ),
        "learning_notes": (
            "retain the working definition",
            "test it on two new examples",
            "where the rule stops applying",
        ),
    }
    decision, action, open_point = cases[_render_domain(row)]
    source = (
        f"The recorded decision is to {decision}. {owner} will {action} by day {day}. "
        f"The source leaves {open_point} unresolved."
    )
    data = _compose_subcards(
        row,
        variant,
        "summary-input",
        (
            (f"Source {code}:", f"Notes {code} to condense:", f"Summary input {code} —"),
            (source,),
        ),
        pool_names=("source_label", "source_record"),
    )
    goal = _compose_subcards(
        row,
        variant,
        "summary-objective",
        (
            (
                "Summarize the record in three concise parts.",
                "Extract a compact three-part summary.",
                "Condense the record without adding context.",
            ),
            (
                "Preserve the decision, assigned action, owner, and timing.",
                "Name the decision and the owned, timed action.",
                "Keep ownership and deadline attached to the action.",
            ),
            (
                "Leave the unresolved point explicitly open.",
                "Do not resolve the uncertainty on the source's behalf.",
                "Report the open point as unresolved.",
            ),
        ),
        pool_names=("summary_request", "required_fields", "open_point_rule"),
    )
    answer = _compose_subcards(
        row,
        variant,
        "summary-answer",
        (
            (
                f"Decision: {decision}.",
                f"Decision: the record is to {decision}.",
                f"Decision: proceed by choosing to {decision}.",
                f"Decision: the agreed direction is to {decision}.",
            ),
            (
                f"Action: {owner} will {action} by day {day}.",
                f"Action: by day {day}, {owner} will {action}.",
                f"Action: {action}, owned by {owner}, is due by day {day}.",
                f"Action: {owner} owns {action} for day {day}.",
            ),
            (
                f"Open point: {open_point} remains unresolved.",
                f"Open point: the source does not resolve {open_point}.",
                f"Open point: {open_point} is still unresolved.",
                f"Open point: no resolution is recorded for {open_point}.",
            ),
        ),
        pool_names=("decision", "owned_action", "open_point"),
    )
    return TaskHand(data, goal, answer, ("decision", "action", "open_point"))


def _extraction(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    amount = _number(f"amount:{code}", 12, 88)
    day = _number(f"extract-day:{code}", 10, 27)
    cases: dict[str, tuple[str, dict[str, Any]]] = {
        "receipt": (
            f"merchant=North Market; date=2026-08-{day:02d}; total=${amount}.50; tax=$4.00; cashier missing",
            {
                "merchant": "North Market",
                "date": f"2026-08-{day:02d}",
                "total": f"{amount}.50 USD",
                "tax": "4.00 USD",
                "cashier": None,
            },
        ),
        "event_listing": (
            f"title=Open Lab; date=2026-08-{day:02d}; venue=Room {amount}; starts=18:30; eligibility missing",
            {
                "title": "Open Lab",
                "date": f"2026-08-{day:02d}",
                "venue": f"Room {amount}",
                "starts": "18:30",
                "eligibility": None,
            },
        ),
        "contact_record": (
            f"name=Sam Iri; role=Editor; organization=North Review; email=sam.{code.lower()}@example.org; phone missing",
            {
                "name": "Sam Iri",
                "role": "Editor",
                "organization": "North Review",
                "email": f"sam.{code.lower()}@example.org",
                "phone": None,
            },
        ),
        "issue_ticket": (
            f"ticket={code}; environment=Linux; severity=medium; status=pending; owner missing",
            {
                "ticket": code,
                "environment": "Linux",
                "severity": "medium",
                "status": "pending",
                "owner": None,
            },
        ),
        "survey_response": (
            f"response={code}; rating=4; topic=navigation; comment=clear after retry; follow_up missing",
            {
                "response": code,
                "rating": 4,
                "topic": "navigation",
                "comment": "clear after retry",
                "follow_up": None,
            },
        ),
        "inventory_record": (
            f"item={code}; quantity={amount}; location=A-{day}; condition=good; checked_by missing",
            {
                "item": code,
                "quantity": amount,
                "location": f"A-{day}",
                "condition": "good",
                "checked_by": None,
            },
        ),
        "schedule_entry": (
            f"event=Review {code}; date=2026-08-{day:02d}; starts=09:30; duration=45 minutes; room missing",
            {
                "event": f"Review {code}",
                "date": f"2026-08-{day:02d}",
                "starts": "09:30",
                "duration_minutes": 45,
                "room": None,
            },
        ),
        "case_note": (
            f"case={code}; observed=package sealed; reported=item incomplete; action=photographs retained; next_owner missing",
            {
                "case": code,
                "observed": "package sealed",
                "reported": "item incomplete",
                "action": "photographs retained",
                "next_owner": None,
            },
        ),
        "address_record": (
            f"recipient=Rin Vale; street={amount} Cedar Lane; locality=Westmere; postal_code={day}804; country=GB; delivery_note missing",
            {
                "recipient": "Rin Vale",
                "street": f"{amount} Cedar Lane",
                "locality": "Westmere",
                "postal_code": f"{day}804",
                "country": "GB",
                "delivery_note": None,
            },
        ),
        "booking_record": (
            f"reference={code}; service=Harbor tour; date=2026-08-{day:02d}; starts=14:15; party_size=3; cancellation_status missing",
            {
                "reference": code,
                "service": "Harbor tour",
                "date": f"2026-08-{day:02d}",
                "starts": "14:15",
                "party_size": 3,
                "cancellation_status": None,
            },
        ),
        "product_record": (
            f"sku={code}; product=Desk lamp; variant=blue; price=${amount}.00; availability=in_stock; warranty missing",
            {
                "sku": code,
                "product": "Desk lamp",
                "variant": "blue",
                "price": f"{amount}.00 USD",
                "availability": "in_stock",
                "warranty": None,
            },
        ),
        "expense_record": (
            f"expense={code}; category=local travel; amount={amount}.25 EUR; date=2026-08-{day:02d}; payment_method=card; approval missing",
            {
                "expense": code,
                "category": "local travel",
                "amount": f"{amount}.25 EUR",
                "date": f"2026-08-{day:02d}",
                "payment_method": "card",
                "approval": None,
            },
        ),
        "shipment_record": (
            f"tracking={code}; carrier=North Parcel; destination=Westmere; milestone=sorting_center; timestamp=2026-08-{day:02d}T07:30Z; exception missing",
            {
                "tracking": code,
                "carrier": "North Parcel",
                "destination": "Westmere",
                "milestone": "sorting_center",
                "timestamp": f"2026-08-{day:02d}T07:30Z",
                "exception": None,
            },
        ),
        "calendar_record": (
            f"event=Planning {code}; organizer=Maya; participants=Jon and Lea; starts=2026-08-{day:02d}T10:00; ends=2026-08-{day:02d}T10:45; location missing; response=tentative",
            {
                "event": f"Planning {code}",
                "organizer": "Maya",
                "participants": ["Jon", "Lea"],
                "starts": f"2026-08-{day:02d}T10:00",
                "ends": f"2026-08-{day:02d}T10:45",
                "location": None,
                "response": "tentative",
            },
        ),
        "measurement_record": (
            f"sample={code}; value={amount}.4; unit=cm; instrument=caliper-2; timestamp=2026-08-{day:02d}T11:20Z; tolerance=0.2 cm; observer missing",
            {
                "sample": code,
                "value": amount + 0.4,
                "unit": "cm",
                "instrument": "caliper-2",
                "timestamp": f"2026-08-{day:02d}T11:20Z",
                "tolerance": "0.2 cm",
                "observer": None,
            },
        ),
        "bibliographic_record": (
            f"title=Bounded Systems; author=I. North; year=2024; venue=Field Notes; identifier={code}; pages missing",
            {
                "title": "Bounded Systems",
                "author": "I. North",
                "year": 2024,
                "venue": "Field Notes",
                "identifier": code,
                "pages": None,
            },
        ),
        "shipment_manifest": (
            f"manifest={code}; carrier=North Freight; origin=Bay 3; destination=Westmere; packages={amount}; seal=verified; customs_note missing",
            {
                "manifest": code,
                "carrier": "North Freight",
                "origin": "Bay 3",
                "destination": "Westmere",
                "packages": amount,
                "seal": "verified",
                "customs_note": None,
            },
        ),
        "survey_export": (
            f"export={code}; responses={amount}; mean_rating=4.2; completion_rate=87%; segment=trial_users; notes missing",
            {
                "export": code,
                "responses": amount,
                "mean_rating": 4.2,
                "completion_rate": "87%",
                "segment": "trial_users",
                "notes": None,
            },
        ),
        "api_response": (
            f"request_id={code}; status=200; duration_ms={amount}; records={day}; next_cursor missing; error missing",
            {
                "request_id": code,
                "status": 200,
                "duration_ms": amount,
                "records": day,
                "next_cursor": None,
                "error": None,
            },
        ),
        "administrative_form": (
            f"form={code}; applicant=Rin Vale; submitted=2026-08-{day:02d}; department=Permits; status=received; reviewer missing",
            {
                "form": code,
                "applicant": "Rin Vale",
                "submitted": f"2026-08-{day:02d}",
                "department": "Permits",
                "status": "received",
                "reviewer": None,
            },
        ),
    }
    raw, fields = cases[_render_domain(row)]
    record_label = row["domain"].replace("_", " ")
    if not record_label.endswith("record"):
        record_label = f"{record_label} record"
    requested_fields = ", ".join(fields)
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
            f"Extract {requested_fields} as JSON. Use null for an absent value.",
            f"Return valid JSON with fields {requested_fields}; represent missing values as null.",
            f"Structure only these fields as JSON: {requested_fields}. Do not infer an absent value.",
        ),
    )
    answer = _compose_subcards(
        row,
        variant,
        "extraction-json-layout",
        ((
            json.dumps(fields, separators=(",", ":")),
            json.dumps(fields),
            json.dumps(fields, separators=(",", ": "), sort_keys=True),
        ),),
        pool_names=("json_layout",),
        cycle_first_pool=True,
    )
    return TaskHand(data, goal, answer, ("json", "requested_fields", "missing_is_null"))
