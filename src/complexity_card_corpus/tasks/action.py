from __future__ import annotations

from typing import Any

from .core import TaskHand, _card_pick, _code, _number, _payload


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
    data = (
        f"Case {code} concerns {payload['subject']}. Reference {code}-A lists day "
        f"{day}, while reference {code}-B lists day {day + 1}. {option.capitalize()} "
        f"is available at {hour}:00 with a quoted cost of ${cost}.{constraint_fact}"
    )
    goal = (
        "Resolve the record conflict and give one next step, its owner, its timing, "
        "and the confirmation to obtain before commitment."
    )
    answer_cards = (
        (
            f"Next step: ask {provider} to reconcile references {code}-A and "
            f"{code}-B before {action}. Owner: the requester contacts {provider}; "
            f"the provider confirms the {confirmation}. Timing: complete the check "
            f"before day {day}. Proceed only after written confirmation; otherwise "
            f"preserve {protected_state}."
        ),
        (
            f"Next step: place references {code}-A and {code}-B before {provider} "
            f"for reconciliation. Owner: the requester opens the query and {provider} "
            f"returns the corrected record. Timing: resolve it before day {day}. "
            f"Confirmation required: {confirmation}. Until then, do not proceed with "
            f"{action}; preserve {protected_state}."
        ),
        (
            f"Next step: pause {action} and request one corrected record covering "
            f"{code}-A and {code}-B. Owner: the requester sends the discrepancy to "
            f"{provider}, which confirms the {confirmation}. Timing: obtain that reply "
            f"before day {day}. If it does not arrive, retain {protected_state}."
        ),
        (
            f"Next step: have {provider} confirm which of {code}-A or {code}-B is "
            f"current. Owner: the requester supplies both references; {provider} owns "
            f"the correction. Timing: before day {day}. Check the {confirmation} in "
            f"writing before {action}, or leave {protected_state} unchanged."
        ),
    )
    answer = _card_pick(row, variant, "practical-answer", answer_cards)
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
    data = (
        f"Environment: {env}. Observed error: {error}. Last change: {change}. "
        f"Control run {code} succeeded before that change; user data is backed up read-only."
        f"{access_note}"
    )
    goal = "Give a reversible diagnostic sequence, a direct fix check, and a regression check."
    scope = "a user-level test profile" if no_admin else "an isolated test environment"
    diagnostic_step = diagnostic_template.format(code=code, scope=scope)
    answer = (
        f"1. Preserve log {code} and reproduce once without changing data. "
        f"2. {diagnostic_step}. 3. Repeat the failing operation in the same test setup and "
        f"compare the new "
        f"log with {code}. Direct check: confirm that '{error}' no longer appears. Regression "
        f"check: repeat the last known-good operation in the test setup. If either check "
        f"fails, {rollback}."
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
    }
    option_cards = option_sets[row["domain"]]
    option_a, option_b, option_c = option_cards[
        _number(f"planning-options:{row['scenario_id']}", 0, len(option_cards) - 1)
    ]
    data = (
        f"Option A: {option_a}; cost ${a}; duration 3 days; every required condition met. "
        f"Option B: {option_b}; cost ${b}; duration 2 days; every required condition met. "
        f"Option C: {option_c}; cost ${budget - 35}; duration 5 days; misses one "
        f"non-negotiable requirement. Maximum budget: ${budget}; deadline: 4 days. "
        "Availability of Option A has not yet been confirmed."
    )
    goal = "Apply the hard constraints, choose an option, order the next steps, and name a fallback trigger."
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
    answer = (
        f"{_card_pick(row, variant, 'planning-reason', reasons)} Choose A: {option_a}. "
        f"{_card_pick(row, variant, 'planning-sequence', sequences)} "
        f"{_card_pick(row, variant, 'planning-fallback', fallbacks)}"
    )
    return TaskHand(data, goal, answer, ("criteria", "choice", "sequence", "fallback"))
