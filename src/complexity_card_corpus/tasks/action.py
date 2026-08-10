from __future__ import annotations

from typing import Any

from ..variable_by.reservoirs import (
    planning_option_cards,
    practical_cards,
    troubleshooting_cards,
)
from .core import (
    TaskHand,
    _code,
    _compose_subcards,
    _deal_task_frames,
    _number,
    _payload,
    _render_domain,
)




def _practical(row: dict[str, Any], variant: int) -> TaskHand:
    payload = _payload(row)
    code = _code(row)
    day = _number(f"day:{code}", 8, 24)
    hour = _number(f"hour:{code}", 9, 16)
    cost = _number(f"cost:{code}", 18, 95)
    provider, option, action, confirmation, protected_state = practical_cards(
        _render_domain(row)
    )
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
            f"Reconcile the {payload['subject']} records under the ${cost} quote, preserve {protected_state}, and gate commitment on {confirmation}.",
        ),
    )
    answer = _compose_subcards(
        row,
        variant,
        "practical-answer",
        (
            (
                f"Next step: ask {provider} to reconcile references {code}-A and {code}-B before {action}.",
                f"Next step: place references {code}-A and {code}-B before {provider} for reconciliation.",
                f"Next step: pause {action} and request one corrected record covering {code}-A and {code}-B.",
                f"Next step: have {provider} confirm which of {code}-A or {code}-B is current.",
            ),
            (
                f"Owner: the requester contacts {provider}; the provider owns the corrected record.",
                f"Owner: {provider} returns the correction once the requester opens the query.",
                f"Owner: once the requester supplies both references, {provider} resolves the discrepancy.",
            ),
            (
                f"Timing: complete the check before day {day}.",
                f"Timing: obtain the corrected record before day {day}.",
                f"Timing: resolve the discrepancy, given a day {day} cutoff.",
            ),
            (
                f"Confirm the {confirmation} in writing before commitment; otherwise preserve {protected_state}.",
                f"Proceed only after the {confirmation} is verified; if not, retain {protected_state}.",
                f"Do not continue with {action} until the {confirmation} is recorded; leave {protected_state} unchanged meanwhile.",
            ),
        ),
        pool_names=("next_step", "owner", "timing", "confirmation"),
    )
    return TaskHand(data, goal, answer, ("next_step", "owner", "timing", "check"))




def _troubleshooting(row: dict[str, Any], variant: int) -> TaskHand:
    env, error, change, diagnostic_template, rollback = troubleshooting_cards(
        _render_domain(row)
    )
    code = _code(row)
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
    diagnostic_step = diagnostic_template.format(code=code, scope=scope)
    opening_cards = (
            f"Preserve log {code}, then reproduce once without changing user data.",
            f"Begin by retaining control log {code} and repeating the failure one time.",
            f"Record the current state beside log {code}; make no change before one controlled reproduction.",
            f"Protect the existing data and use log {code} as the comparison baseline.",
    )
    comparison_cards = (
            f"Repeat the failing operation in the same setup and compare the resulting log with {code}.",
            f"Run the failing action once more under the test condition, then compare both observations with {code}.",
            f"Keep every other variable fixed, repeat the operation, and inspect the difference from control {code}.",
    )
    verification_cards = (
            f"Direct check: confirm that '{error}' no longer appears. Regression check: repeat the last known-good operation.",
            f"Verify the fix by checking that '{error}' is absent, then rerun the documented good case.",
            f"The direct test passes only when '{error}' disappears; the regression test must also preserve the former good behavior.",
    )
    failure_cards = (
            f"If either check fails, {rollback}.",
            f"If the direct or regression result is negative, {rollback}.",
            f"Do not widen the change after a failed check; {rollback}.",
    )
    diagnostic_cards = (
        f"2. {diagnostic_step}.",
        f"2. Run this bounded diagnostic: {diagnostic_step}.",
        f"2. In {scope}, perform this check: {diagnostic_step}.",
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
        _number(f"planning-options:{row['scenario_id']}", 0, len(option_cards) - 1)
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
    reasons = (
        f"Reject B because ${b} exceeds the ${budget} cap. Reject C because it misses a non-negotiable requirement and exceeds the deadline.",
        f"The hard constraints remove B at ${b}, above the ${budget} limit, and C, which is late and fails one mandatory condition.",
        f"B fails the budget test (${b} versus ${budget}); C fails both the four-day deadline and one unwaivable requirement.",
    )
    sequences = (
        f"Sequence: confirm availability of {option_a} today, hold it reversibly, then verify every requirement before payment.",
        f"Sequence: verify {option_a} against the ${budget} cap and four-day deadline, request a reversible hold, then await written confirmation.",
        f"Sequence: check {option_a}'s availability, confirm every hard requirement under ${budget}, and make payment last.",
    )
    fallbacks = (
        f"Fallback trigger: if {option_a} cannot be confirmed by tomorrow, pause and reopen the shortlist rather than selecting B or C.",
        f"Fallback trigger: an unverified requirement for {option_a} means stopping to seek another option under ${budget}.",
        f"Fallback trigger: if the hold on {option_a} expires before confirmation, return to comparison instead of accepting a failed option.",
    )
    answer = _compose_subcards(
        row,
        variant,
        "planning-answer",
        (
            reasons,
            (
                f"Choose A: {option_a}.",
                f"Choose Option A, {option_a}, as the compliant candidate.",
                f"Choose the viable option, A: {option_a}.",
            ),
            sequences,
            fallbacks,
        ),
        pool_names=("criteria", "choice", "sequence", "fallback"),
    )
    return TaskHand(data, goal, answer, ("criteria", "choice", "sequence", "fallback"))
