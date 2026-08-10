from __future__ import annotations

_PLANNING_OPTION_SETS = {
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


def planning_option_cards(domain: str) -> tuple[tuple[str, str, str], ...]:
    """Return localized comparison options for one planning domain."""

    return _PLANNING_OPTION_SETS[domain]
