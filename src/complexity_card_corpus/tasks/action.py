from __future__ import annotations

from typing import Any

from ..variable_by.reservoirs import (
    planning_answer_cards,
    planning_option_cards,
    practical_answer_cards,
    practical_cards,
    troubleshooting_cards,
    troubleshooting_comparison_cards,
    troubleshooting_diagnostic_surfaces,
    troubleshooting_failure_cards,
    troubleshooting_opening_cards,
    troubleshooting_verification_cards,
)
from .core import (
    TaskHand,
    _code,
    _compose_subcards,
    _deal_task_frames,
    _number,
    _payload,
    _pick,
    _render_domain,
)


def _without_leading_article(value: str) -> str:
    """Return a noun phrase safe after a determiner supplied by a template."""

    lowered = value.casefold()
    for article in ("the ", "an ", "a "):
        if lowered.startswith(article):
            return value[len(article) :]
    return value




def _practical(row: dict[str, Any], variant: int) -> TaskHand:
    payload = _payload(row)
    code = _code(row)
    day = _number(f"day:{code}", 8, 24)
    hour = _number(f"hour:{code}", 9, 16)
    cost = _number(f"cost:{code}", 18, 95)
    provider, option, action, confirmation, protected_state = practical_cards(
        _render_domain(row)
    )
    if isinstance(confirmation, tuple):
        confirmation = _pick(
            f"practical-confirmation:{code}:{variant}", confirmation
        )
    bare_subject = _without_leading_article(payload["subject"])
    constraint = row.get("constraint", "")
    constraint_fact = ""
    if "cost" in constraint.lower():
        limit = cost + _number(f"cost-headroom:{code}", 5, 18)
        constraint_fact = f" The authorized maximum is ${limit}."
    elif "accessibility" in constraint.lower():
        constraint_fact = " Required accessibility support must be confirmed."
        confirmation = f"{confirmation}, including required accessibility support"
    case_data = (
        f"Case {code} concerns {payload['subject']}. Reference {code}-A lists day "
        f"{day}, while reference {code}-B lists day {day + 1}. {option.capitalize()} "
        f"is available at {hour}:00 with a quoted cost of ${cost}.{constraint_fact}"
    )
    data, goal = _deal_task_frames(
        row,
        variant,
        "practical",
        (
            case_data,
            f"Action record {code}: {case_data.split('. ', 1)[-1]}",
            f"Decision input — {case_data}",
        ),
        (
            f"Resolve the record conflict for {payload['subject']} before day {day}, keep the ${cost} quote visible, and require {confirmation} before commitment.",
            f"Choose one bounded action for {payload['subject']} at {hour}:00, assign ownership before day {day}, and verify {confirmation}.",
            f"Reconcile {bare_subject} records before day {day} under the ${cost} quote, preserve {protected_state}; gate commitment on {confirmation}.",
        ),
    )
    answer = _compose_subcards(
        row,
        variant,
        "practical-answer",
        practical_answer_cards(
            provider=provider,
            code=code,
            action=action,
            day=day,
            confirmation=confirmation,
            protected_state=protected_state,
        ),
        pool_names=("next_step", "owner", "timing", "confirmation"),
    )
    return TaskHand(data, goal, answer, ("next_step", "owner", "timing", "check"))




def _troubleshooting(row: dict[str, Any], variant: int) -> TaskHand:
    env, error, change, diagnostic_templates, rollback_cards = troubleshooting_cards(
        _render_domain(row)
    )
    code = _code(row)
    rollback = _pick(f"troubleshooting-rollback:{code}:{variant}", rollback_cards)
    no_admin = "administrator access is unavailable" in row["constraint"].lower()
    access_note = (
        f" Any test in {env} that requires a system-level change is out of scope."
        if no_admin
        else ""
    )
    diagnostic_record = (
        f"Environment: {env}. Observed error: {error}. Last change: {change}. "
        f"Control run {code} succeeded before that change; user data is backed up read-only."
        f"{access_note}"
    )
    data, goal = _deal_task_frames(
        row,
        variant,
        "troubleshooting",
        (
            diagnostic_record,
            f"Diagnostic record {code}: {diagnostic_record}",
            f"Troubleshooting input — {diagnostic_record}",
        ),
        (
            f"Give a reversible sequence in {env} for '{error}', isolate the effect of {change}, then run direct and regression checks.",
            f"Design three bounded steps for '{error}' after {change}, verify the repair in {env}, and preserve the prior good behavior.",
            f"Use the successful control state in {env} to isolate {change}, remove '{error}', and confirm the former behavior still works.",
        ),
    )
    scope = "a user-level test profile" if no_admin else "an isolated test environment"
    diagnostic_template = _pick(
        f"troubleshooting-diagnostic:{code}:{variant}", diagnostic_templates
    )
    diagnostic_step = diagnostic_template.format(code=code, scope=scope)
    opening_cards = troubleshooting_opening_cards(code)
    comparison_cards = troubleshooting_comparison_cards(code)
    verification_cards = troubleshooting_verification_cards(error)
    failure_cards = troubleshooting_failure_cards(rollback)
    diagnostic_cards = troubleshooting_diagnostic_surfaces(
        diagnostic_step,
        scope,
    )
    answer = _compose_subcards(
        row,
        variant,
        "troubleshooting-answer",
        (
            tuple(f"1. {opening}" for opening in opening_cards),
            diagnostic_cards,
            tuple(f"3. {comparison}" for comparison in comparison_cards),
            verification_cards,
            failure_cards,
        ),
        pool_names=(
            "preserve_state",
            "diagnostic",
            "comparison",
            "verification",
            "rollback",
        ),
    )
    return TaskHand(data, goal, answer, ("steps", "direct_check", "regression_check"))


def _planning(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    budget = _number(f"budget:{code}", 90, 180)
    a = budget - 20
    b = budget + 25
    option_cards = planning_option_cards(_render_domain(row))
    option_a, option_b, option_c = option_cards[
        _number(
            f"planning-options:{row['scenario_id']}:{variant}",
            0,
            len(option_cards) - 1,
        )
    ]
    comparison_record = (
        f"Option A: {option_a}; cost ${a}; duration 3 days; every required condition met. "
        f"Option B: {option_b}; cost ${b}; duration 2 days; every required condition met. "
        f"Option C: {option_c}; cost ${budget - 35}; duration 5 days; misses one "
        f"non-negotiable requirement. Maximum budget: ${budget}; deadline: 4 days. "
        "Availability of Option A has not yet been confirmed."
    )
    data, goal = _deal_task_frames(
        row,
        variant,
        "planning",
        (
            comparison_record,
            f"Planning comparison {code}: {comparison_record}",
            f"Decision table {code} — {comparison_record}",
        ),
        (
            f"Apply the ${budget} cap and four-day deadline to {option_a}, {option_b}, and {option_c}; choose, sequence, and set a fallback.",
            f"Eliminate {option_b} above ${budget} and {option_c} beyond four days, then give {option_a} a reversible sequence and fallback.",
            f"Compare {option_a}, {option_b}, and {option_c} against ${budget} and four days before recommending the compliant plan.",
        ),
    )
    answer_cards = planning_answer_cards(
        option_a=option_a,
        budget=budget,
        option_b_cost=b,
    )
    answer = _compose_subcards(
        row,
        variant,
        "planning-answer",
        answer_cards,
        pool_names=("criteria", "choice", "sequence", "fallback"),
    )
    return TaskHand(data, goal, answer, ("criteria", "choice", "sequence", "fallback"))
