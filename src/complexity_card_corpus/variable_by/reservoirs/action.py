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
