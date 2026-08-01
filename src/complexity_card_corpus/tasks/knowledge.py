from __future__ import annotations

import json
from typing import Any

from .core import TaskHand, _card_pick, _code, _number, _pick


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
    }
    passage, goal, supported = cases[row["domain"]]
    data = f"Source {code}: {passage}"
    answer_cards = (
        f"Based on Source {code}: {supported}",
        f"Source {code} supports this answer: {supported}",
        f"The documented answer is: {supported} This is limited to Source {code}.",
        f"According to Source {code}: {supported}",
    )
    answer = _card_pick(row, variant, "grounded-answer", answer_cards)
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
    decision, action, open_point = cases[row["domain"]]
    data = (
        f"Source {code}: The recorded decision is to {decision}. {owner} will {action} by day {day}. "
        f"The source leaves {open_point} unresolved."
    )
    goal = "Summarize the decision, action, owner, timing, and unresolved point in three concise lines."
    answer = (
        f"Decision: {decision}. Action: {owner} will {action} by day {day}. "
        f"Open point: {open_point} remains unresolved."
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
    }
    raw, fields = cases[row["domain"]]
    record_label = row["domain"].replace("_", " ")
    if not record_label.endswith("record"):
        record_label = f"{record_label} record"
    data = f"Raw {record_label}: {raw}."
    goal = f"Extract {', '.join(fields)} as JSON. Use null for an absent value."
    answer = json.dumps(fields, separators=(",", ":"))
    return TaskHand(data, goal, answer, ("json", "requested_fields", "missing_is_null"))
