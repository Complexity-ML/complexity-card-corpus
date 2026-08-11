from __future__ import annotations

from typing import Any

from ..variable_by.reservoirs import (
    GroundedQAFacts,
    grounded_qa_variable_by,
    meeting_summary_cards,
)
from ..variable_by.templates import GROUNDED_QA_TEMPLATES
from .core import (
    TaskHand,
    _code,
    _compose_subcards,
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
    facts = GroundedQAFacts(
        code=code,
        year=year,
        battery_hours=battery_hours,
        return_days=return_days,
        exposure_hours=exposure_hours,
        temperature_change=temperature_change,
        owner=owner,
        delivery_day=delivery_day,
        train_number=train_number,
        departure_hour=departure_hour,
        departure_minute=departure_minute,
        platform=platform,
        python_minor=python_minor,
        release_major=release_major,
        release_minor=release_minor,
        longest_battery=longest_battery,
        other_battery=other_battery,
        status_minute=status_minute,
        ticket_minute=ticket_minute,
        available_region=available_region,
        ticket_region=ticket_region,
        failed_operation=failed_operation,
        event_day=event_day,
        event_room=event_room,
        energy_kwh=energy_kwh,
        energy_rate=energy_rate,
        course_number=course_number,
        maintenance_day=maintenance_day,
        sensor_count=sensor_count,
        measured_value=measured_value,
        notice_days=notice_days,
        sample_count=sample_count,
        ph_value=ph_value,
        quote_units=quote_units,
        quote_price=quote_price,
        tested_pages=tested_pages,
        operating_limit=operating_limit,
    )
    variables = grounded_qa_variable_by(_render_domain(row), facts)
    data = _compose_subcards(
        row,
        variant,
        "grounded-data",
        (GROUNDED_QA_TEMPLATES["data"],),
        pool_names=("source",),
        variable_by=variables,
    )
    goal = _compose_subcards(
        row,
        variant,
        "grounded-goal",
        (GROUNDED_QA_TEMPLATES["goal"],),
        pool_names=("request",),
        variable_by=variables,
    )
    answer = _compose_subcards(
        row,
        variant,
        "grounded-answer",
        (
            GROUNDED_QA_TEMPLATES["answer_scope"],
            GROUNDED_QA_TEMPLATES["answer_complete"],
        ),
        pool_names=("evidence_scope", "grounded_result"),
        variable_by=variables,
    )
    subject = row["domain"].replace("_", " ").title()
    return TaskHand(
        data,
        goal,
        answer,
        ("direct_answer", "evidence", "unknown"),
        situation_title=f"{subject} — answer from the supplied source",
        situation=_compose_subcards(
            row,
            variant,
            "grounded-situation",
            (GROUNDED_QA_TEMPLATES["situation"],),
            pool_names=("situation",),
            variable_by=variables,
        ),
        rule=_compose_subcards(
            row,
            variant,
            "grounded-rule",
            (GROUNDED_QA_TEMPLATES["rule"],),
            pool_names=("rule",),
            variable_by=variables,
        ),
    )


_SUMMARY_COUNT_RESERVOIRS: dict[str, tuple[int, int]] = {
    "contrast_ratio": (3, 21),
    "sample_count": (8, 96),
    "case_count": (20, 850),
    "test_coverage": (60, 99),
    "employee_count": (15, 480),
    "citation_count": (5, 140),
    "downtime_minutes": (5, 240),
    "example_count": (10, 220),
}


def _summary(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    owner = _pick(f"summary-owner:{code}", ("Mina", "Paul", "Sora", "Theo", "Lina"))
    day = _number(f"summary-day:{code}", 12, 27)

    def count(name: str) -> int:
        low, high = _SUMMARY_COUNT_RESERVOIRS[name]
        return _number(f"summary-{name}:{code}", low, high)

    contrast_ratio = count("contrast_ratio")
    sample_count = count("sample_count")
    case_count = count("case_count")
    test_coverage = count("test_coverage")
    employee_count = count("employee_count")
    citation_count = count("citation_count")
    downtime_minutes = count("downtime_minutes")
    example_count = count("example_count")

    cases = {
        "meeting_transcript": (
            f"approve the revised interface copy for the settings page redesign, targeting a {contrast_ratio}:1 contrast ratio",
            f"run two accessibility checks (targeting a {contrast_ratio}:1 contrast ratio): screen-reader navigation and color contrast ratios",
            f"the exact release date and rollout order for the change ahead of {contrast_ratio}:1 sign-off",
        ),
        "research_notes": (
            f"retain the observed temperature result from the first of {sample_count} recorded trial runs",
            f"replicate two uncertain measurements across {sample_count} samples: the peak temperature and the cooling rate",
            f"the underlying causal explanation for the observed thermal effect across {sample_count} samples",
        ),
        "support_thread": (
            f"keep the support case open pending further diagnosis, alongside {case_count} similar cases",
            f"test two account-recovery paths across the {case_count} cases: email reset and device verification",
            f"whether the reported issue, seen in {case_count} similar cases, is limited to one device type",
        ),
        "project_update": (
            f"accept the completed prototype at {test_coverage}% test coverage pending two remaining rounds of final integration testing",
            f"finish two integration checks at {test_coverage}% coverage: payment gateway and notification delivery",
            f"the confirmed public launch date for the wider release, currently at {test_coverage}% coverage",
        ),
        "policy_memo": (
            f"adopt the revised after-hours access rule for the {employee_count}-person shared workspace",
            f"document two exceptions for the {employee_count}-person workspace: emergency access and approved contractors",
            f"the confirmed enforcement start date for the revised rule affecting {employee_count} employees",
        ),
        "article_excerpt": (
            f"retain the article's central claim about the observed pattern, backed by {citation_count} citations",
            f"verify two examples among the {citation_count} citations: pilot study and follow-up survey",
            f"whether the pattern, cited {citation_count} times, generalizes beyond the cited examples",
        ),
        "incident_log": (
            f"keep the affected service running in monitored recovery mode after {downtime_minutes} minutes of downtime",
            f"after {downtime_minutes} minutes, inspect two remaining sources: load balancer and cache layer",
            f"the incident's confirmed, precise underlying root cause behind the {downtime_minutes}-minute outage",
        ),
        "learning_notes": (
            f"retain the current working definition of the rule, validated against {example_count} training examples",
            f"extend the {example_count}-example check with a boundary case and a negative case",
            f"the exact point where the rule, tested on {example_count} examples, stops applying",
        ),
    }
    summary_domain = _render_domain(row)
    decision, action, open_point = cases[summary_domain]
    decision_cards = (decision,)
    open_point_cards = (open_point,)
    if summary_domain == "meeting_transcript":
        decision_cards, open_point_cards = meeting_summary_cards(
            contrast_ratio,
            default_decision=decision,
            default_open_point=open_point,
        )
    decision = _pick(f"summary-decision:{code}:{variant}", decision_cards)
    open_point = _pick(f"summary-open-point:{code}:{variant}", open_point_cards)
    source = (
        f"The recorded decision is to {decision}. {owner} will {action} by day {day}. "
        f"The source leaves {open_point} unresolved."
    )
    data = _compose_subcards(
        row,
        variant,
        "summary-input",
        (
            (
                f"Source {code}:",
                f"Notes {code} to condense:",
                f"Summary input {code} —",
            ),
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
                f"Preserve the decision, assigned action, owner {owner}, and day {day} timing.",
                f"Name the decision and the action owned by {owner} for day {day}.",
                f"Keep {owner}'s ownership and the day {day} deadline attached to the action.",
            ),
            (
                "Leave the unresolved point explicitly open.",
                f"Do not turn {owner}'s day {day} assignment into an answer to the unresolved point.",
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
                f"Action: due day {day}, {owner} will {action}.",
                f"Action: no later than day {day}, {owner} will {action}.",
                f"Action: {action}, owned by {owner}, closing out on day {day}, once confirmed.",
                f"Action: {owner} is assigned to {action}; day {day} is the outside limit.",
            ),
            (
                f"Open point: {open_point} remains unresolved.",
                f"Open point: nothing in the source resolves {open_point}.",
                f"Open point: {open_point} is still unresolved.",
                f"Open point: no resolution is recorded for {open_point}.",
            ),
        ),
        pool_names=("decision", "owned_action", "open_point"),
    )
    return TaskHand(data, goal, answer, ("decision", "action", "open_point"))
