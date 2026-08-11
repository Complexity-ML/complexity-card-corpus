from __future__ import annotations

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


def practical_cards(domain: str) -> tuple[str, str, str, str, str]:
    """Return localized semantic cells for one practical-action domain."""

    return _PRACTICAL_CARDS[domain]


def practical_answer_cards(
    *,
    provider: str,
    code: str,
    action: str,
    day: int,
    confirmation: str,
    protected_state: str,
) -> tuple[tuple[str, ...], ...]:
    """Build the linked action, owner, timing, and confirmation reservoirs."""

    return (
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
            f"Timing: resolve the discrepancy before the day {day} cutoff.",
            f"Timing: finish reconciliation no later than day {day}.",
            f"Timing: secure the provider's correction by day {day}.",
            f"Timing: keep commitment paused until the records agree, with day {day} as the deadline.",
            f"Timing: close the reference check ahead of day {day}.",
            f"Timing: require the authoritative record by the end of day {day}.",
            f"Timing: complete provider verification prior to day {day}.",
            f"Timing: use day {day} as the latest date for resolving both references.",
            f"Timing: reconcile the conflicting entries by day {day}, before commitment.",
            f"Timing: obtain one confirmed version before day {day} begins.",
            f"Timing: keep the action on hold and settle the reference conflict by day {day}.",
            f"Timing: ask for the corrected source now and require it no later than day {day}.",
            f"Timing: use the period ending on day {day} to verify which record governs.",
            f"Timing: complete the comparison and receive provider confirmation by day {day}.",
            f"Timing: do not pass day {day} without either a reconciled record or a continued hold.",
            f"Timing: resolve both dated entries before the close of day {day}.",
            f"Timing: make day {day} the deadline for one authoritative response from {provider}.",
            f"Timing: finish the record check ahead of any commitment and no later than day {day}.",
            f"Timing: obtain written reconciliation during the window that ends on day {day}.",
            f"Timing: leave the provisional option unchanged until verification arrives by day {day}.",
            f"Timing: require agreement between the references on or before day {day}.",
            f"Timing: conclude the provider query by day {day}, keeping the current state until then.",
        ),
        (
            f"Confirm the {confirmation} in writing before commitment; otherwise preserve {protected_state}.",
            f"Proceed only after the {confirmation} is verified; if not, retain {protected_state}.",
            f"Do not continue with {action} until the {confirmation} is recorded; leave {protected_state} unchanged meanwhile.",
        ),
    )
