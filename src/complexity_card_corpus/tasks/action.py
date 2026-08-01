from __future__ import annotations

from typing import Any

from .core import (
    TaskHand,
    _code,
    _compose_subcards,
    _deal_task_frames,
    _number,
    _payload,
)


_PRACTICAL_CARDS = {
    "public_transit": (
        "the transit operator",
        "a provisional departure",
        "reserving the journey",
        "travel day, departure time, fare, and accessibility status",
        "the existing ticket",
    ),
    "healthcare_admin": (
        "the clinic",
        "a provisional appointment",
        "confirming the appointment",
        "appointment day, arrival time, fee, and preparation instructions",
        "the current appointment record",
    ),
    "air_travel": (
        "the airline",
        "a refundable itinerary",
        "booking the flight",
        "travel day, departure time, fare, and refund conditions",
        "the existing itinerary",
    ),
    "retail_returns": (
        "the retailer",
        "a reversible return appointment",
        "sending the item back",
        "return deadline, drop-off time, fee, and refund method",
        "the item and proof of purchase",
    ),
    "account_access": (
        "official account support",
        "a reversible identity-check session",
        "starting account recovery",
        "review day, support time, identity-check method, and any fee",
        "the current credentials and security settings",
    ),
    "event_registration": (
        "the event organizer",
        "a provisional registration slot",
        "submitting the registration",
        "event day, arrival time, fee, and cancellation terms",
        "the current registration state",
    ),
    "subscriptions": (
        "the subscription provider",
        "a reversible plan change",
        "changing the subscription",
        "effective day, plan, price, and renewal terms",
        "the current plan",
    ),
    "course_enrolment": (
        "the course administrator",
        "a provisional class place",
        "enrolling in the course",
        "start day, session time, fee, and prerequisites",
        "the current enrolment state",
    ),
    "banking_admin": (
        "the bank through its official channel",
        "a provisional service appointment",
        "authorizing the account request",
        "service day, appointment time, fee, and required documents",
        "the current account instructions",
    ),
    "appointments": (
        "the service desk",
        "a provisional appointment",
        "confirming the appointment",
        "appointment day, time, price, and cancellation terms",
        "the current appointment",
    ),
    "home_repair": (
        "the repair provider",
        "a provisional visit",
        "authorizing the repair visit",
        "visit day, arrival window, quoted price, and work scope",
        "the property and existing work order",
    ),
    "parcel_delivery": (
        "the carrier",
        "a reversible redelivery slot",
        "requesting redelivery",
        "delivery day, time window, fee, and destination",
        "the current delivery instruction",
    ),
}


def _practical(row: dict[str, Any], variant: int) -> TaskHand:
    payload = _payload(row)
    code = _code(row)
    day = _number(f"day:{code}", 8, 24)
    hour = _number(f"hour:{code}", 9, 16)
    cost = _number(f"cost:{code}", 18, 95)
    provider, option, action, confirmation, protected_state = _PRACTICAL_CARDS[
        row["domain"]
    ]
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
            "Resolve the record conflict and give one next step, its owner, its timing, and the confirmation to obtain before commitment.",
            "Choose one bounded next action, assign responsibility and timing, and name the check required before commitment.",
            "Reconcile the conflicting records and provide an owned, timed, reversible next step with a confirmation gate.",
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
                f"Owner: the requester opens the query and {provider} returns the correction.",
                f"Owner: the requester supplies both references; {provider} resolves the discrepancy.",
            ),
            (
                f"Timing: complete the check before day {day}.",
                f"Timing: obtain the corrected record before day {day}.",
                f"Timing: resolve the discrepancy no later than day {day}.",
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


_ERRORS = {
    "software_install": (
        "macOS 15",
        "installer exits with code 73",
        "the install directory was changed",
        (
            "Compare the installer settings with control run {code}. If the previous "
            "install directory is documented, select it in {scope}; otherwise stop and "
            "request that value"
        ),
        "discard the test configuration and leave the original installer settings unchanged",
    ),
    "network_connection": (
        "a laptop on Wi-Fi",
        "requests time out after DNS lookup",
        "a custom DNS server was enabled",
        (
            "Compare the DNS settings with control run {code}. If the previous resolver "
            "is documented, select it in {scope}; otherwise stop and request that value"
        ),
        "discard the test profile and leave the original network settings unchanged",
    ),
    "file_sync": (
        "a desktop sync client",
        "local changes remain queued",
        "the remote folder was renamed",
        (
            "Read the remote folder listing without modifying it. If the former folder "
            "name is still present, point a disposable sync profile at it; otherwise stop "
            "and preserve the queue"
        ),
        "discard the disposable sync profile and leave the queued changes untouched",
    ),
    "peripheral": (
        "a USB keyboard",
        "the device powers on but sends no input",
        "it was moved through a hub",
        "Bypass the hub and connect the keyboard directly for one test",
        "restore the original hub arrangement and stop with both observations intact",
    ),
    "web_form": (
        "a current browser",
        "submission returns HTTP 422",
        "a required profile field was removed",
        (
            "Duplicate the draft in {scope}. If the required field's previous value is "
            "documented, restore it only in the duplicate; otherwise stop and request the value"
        ),
        "discard the duplicate draft and leave the original form unchanged",
    ),
    "data_pipeline": (
        "a nightly ETL job",
        "the transform stage reports a schema mismatch",
        "a source column changed type",
        (
            "Use {scope} with a read-only input copy. If control run {code} documents the "
            "previous column type, cast only the copy to that type; otherwise stop and "
            "request the schema"
        ),
        "discard the isolated pipeline copy and leave the source data unchanged",
    ),
    "audio_output": (
        "a laptop with external speakers",
        "the level meter moves but no sound is audible",
        "the preferred output device was changed",
        (
            "Play control tone {code} in {scope}, then select the documented speaker "
            "for that profile only and compare the level meter with the audible result"
        ),
        "restore the former output selection and retain both test observations",
    ),
    "printer_queue": (
        "a shared office printer",
        "the first queued document remains in processing",
        "a new paper-size default was selected",
        (
            "Preserve the source document, pause the queue, and send one disposable "
            "one-page file from {scope} using the paper size recorded in control {code}"
        ),
        "remove only the disposable job and leave the original document and queue intact",
    ),
    "account_login": (
        "a browser login page",
        "the password is accepted but the verification code does not arrive",
        "the notification address was edited",
        (
            "Check the masked delivery destination against control {code} without "
            "changing credentials. If it differs, correct it only through the official "
            "account recovery flow in {scope}"
        ),
        "end the recovery session without altering the current password or security factors",
    ),
    "storage_space": (
        "a user laptop",
        "the system reports less than one gigabyte free",
        "a local snapshot was created",
        (
            "Use the read-only storage summary in {scope} to compare snapshots, temporary "
            "files, and user folders with control {code}; do not delete any category"
        ),
        "close the inspection without deleting files and retain the category totals",
    ),
    "battery_charging": (
        "a tablet at room temperature",
        "the charging indicator stays off",
        "the charging cable was replaced",
        (
            "Test the documented cable from control {code} with the same power outlet, "
            "then test the replacement cable once while recording the indicator state"
        ),
        "disconnect the test cable and restore the documented charging arrangement",
    ),
    "browser_session": (
        "a signed-in web application",
        "one page reloads into a blank state",
        "a browser extension was enabled",
        (
            "Open the same page in {scope} with extensions disabled. Compare its network "
            "result with control {code} without clearing the original profile"
        ),
        "discard the test profile and preserve the original browser session",
    ),
    "email_delivery": (
        "a desktop mail client",
        "one message remains in the outbox",
        "the outgoing server port was changed",
        (
            "Compare the outgoing settings with control {code}, then send a disposable "
            "message to the sender's own address from {scope} using the documented port"
        ),
        "remove only the disposable message and restore the previous test settings",
    ),
    "spreadsheet_formula": (
        "a workbook copy",
        "the monthly total is lower than the visible line items",
        "one row was inserted above the total",
        (
            "Inspect the total formula in {scope} and compare its referenced range with "
            "control {code}; adjust only the copied workbook if the inserted row is excluded"
        ),
        "discard the workbook copy and preserve the original values and formulas",
    ),
}


def _troubleshooting(row: dict[str, Any], variant: int) -> TaskHand:
    env, error, change, diagnostic_template, rollback = _ERRORS[row["domain"]]
    code = _code(row)
    no_admin = "administrator access is unavailable" in row["constraint"].lower()
    access_note = (
        " Any test that requires a system-level change is out of scope."
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
            "Give a reversible diagnostic sequence, a direct fix check, and a regression check.",
            "Design three bounded diagnostic steps, then state both the fix verification and the regression test.",
            "Use the control run to isolate the change, verify the repair directly, and preserve the last known-good behavior.",
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
    option_sets = {
        "travel_plan": (
            (
                "a refundable rail itinerary",
                "a non-refundable flight",
                "a bus route without step-free access",
            ),
            (
                "a flexible direct coach",
                "a premium same-day flight",
                "a late train without an accessible transfer",
            ),
            (
                "a cancellable daytime train",
                "a costly express flight",
                "an overnight coach missing the required luggage allowance",
            ),
        ),
        "learning_plan": (
            (
                "a guided practice course",
                "an intensive workshop",
                "a self-study bundle missing the required exercises",
            ),
            (
                "a mentored three-day lab",
                "a premium bootcamp",
                "a recorded course without feedback exercises",
            ),
            (
                "a structured tutorial series",
                "a private accelerated class",
                "a reading pack missing the required assessment",
            ),
        ),
        "small_project": (
            (
                "a scoped three-day implementation",
                "an accelerated external build",
                "a five-day build missing the acceptance test",
            ),
            (
                "a reversible configuration change",
                "a premium rush deployment",
                "a longer patch without rollback verification",
            ),
            (
                "a tested minimum feature",
                "an outsourced express package",
                "a broad rewrite missing the required audit log",
            ),
        ),
        "purchase_decision": (
            (
                "a supported standard model",
                "a faster premium model",
                "a discounted model missing the required connector",
            ),
            (
                "a repairable base unit",
                "an over-budget performance unit",
                "a cheaper unit without the required warranty",
            ),
            (
                "a compatible standard package",
                "a premium bundle",
                "a clearance package missing the required adapter",
            ),
        ),
        "meal_plan": (
            (
                "a three-day menu reusing staple ingredients",
                "a premium prepared-meal package",
                "a low-cost menu missing the required allergen restriction",
            ),
            (
                "a batch-cooked weekday menu",
                "a same-day delivery plan",
                "a five-day menu requiring unavailable evening preparation",
            ),
            (
                "a flexible pantry-first menu",
                "a restaurant meal bundle",
                "a discounted menu missing the required protein option",
            ),
        ),
        "household_move": (
            (
                "a staged move with labeled essentials",
                "a premium same-day moving crew",
                "a self-move plan missing protected transport for fragile items",
            ),
            (
                "a three-day room-by-room move",
                "an express full-service move",
                "a five-day plan without the required key handover check",
            ),
            (
                "a reversible first load of non-essential boxes",
                "a high-cost direct transfer",
                "a cheap transport option missing adequate capacity",
            ),
        ),
        "community_event": (
            (
                "an accessible indoor workshop",
                "a premium auditorium booking",
                "an outdoor gathering without the required rain plan",
            ),
            (
                "a small library event with a backup room",
                "an express commercial venue",
                "a five-day street event missing the accessibility check",
            ),
            (
                "a volunteer-led community session",
                "a catered conference package",
                "a low-cost venue without the required capacity",
            ),
        ),
        "appointment_schedule": (
            (
                "a route with confirmed travel buffers",
                "a premium transport-assisted schedule",
                "a compact schedule missing the required preparation window",
            ),
            (
                "a three-day sequence ordered by location",
                "a same-day private service",
                "a five-day sequence with an impossible transfer",
            ),
            (
                "a reversible hold on compatible time slots",
                "an over-budget coordination service",
                "a cheap sequence missing one fixed appointment window",
            ),
        ),
        "maintenance_plan": (
            (
                "a staged inspection before replacement",
                "a premium full-system replacement",
                "a low-cost repair without the required rollback test",
            ),
            (
                "a three-day service with checkpoints",
                "an express external service",
                "a five-day overhaul missing the spare-part check",
            ),
            (
                "a reversible component-level repair",
                "a high-cost preventative package",
                "a discounted repair missing the required safety inspection",
            ),
        ),
        "reading_plan": (
            (
                "a guided sequence with recall checks",
                "a premium intensive seminar",
                "a long reading list missing the required review session",
            ),
            (
                "a three-day core-reading plan",
                "an accelerated tutoring package",
                "a five-day plan that exceeds the reading deadline",
            ),
            (
                "a prerequisite-first reading sequence",
                "a costly annotated edition bundle",
                "a cheap summary pack missing the required primary text",
            ),
        ),
    }
    option_cards = option_sets[row["domain"]]
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
            "Apply the hard constraints, choose an option, order the next steps, and name a fallback trigger.",
            "Eliminate non-compliant options, choose the viable one, and provide a reversible sequence with a fallback.",
            "Compare the options against budget, deadline, and requirements before recommending a sequenced plan.",
        ),
    )
    reasons = (
        f"Reject B because ${b} exceeds the ${budget} cap. Reject C because it misses a non-negotiable requirement and exceeds the deadline.",
        f"The hard constraints remove B at ${b}, above the ${budget} limit, and C, which is late and misses a non-negotiable requirement.",
        f"B fails the budget test (${b} versus ${budget}); C fails both the four-day deadline and one non-negotiable requirement.",
    )
    sequences = (
        "Sequence: confirm availability today, hold it reversibly, then verify the requirement before payment.",
        "Sequence: verify the listed conditions, request a reversible hold, and commit only after written confirmation.",
        "Sequence: check availability, confirm every hard requirement, and make payment last.",
    )
    fallbacks = (
        "Fallback trigger: if A cannot be confirmed by tomorrow, pause and reopen the shortlist rather than selecting B or C.",
        "Fallback trigger: if any requirement for A remains unverified, stop and seek another compliant option.",
        "Fallback trigger: if the reversible hold expires before confirmation, return to comparison instead of accepting a failed option.",
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
