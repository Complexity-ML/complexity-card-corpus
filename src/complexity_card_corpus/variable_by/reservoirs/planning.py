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


def planning_answer_cards(
    *,
    option_a: str,
    budget: int,
    option_b_cost: int,
) -> tuple[tuple[str, ...], ...]:
    """Build comparison, choice, sequence, and fallback reservoirs."""

    reasons = (
        f"Reject B because ${option_b_cost} exceeds the ${budget} cap. Reject C because it misses a non-negotiable requirement and exceeds the deadline.",
        f"The hard constraints remove B at ${option_b_cost}, above the ${budget} limit, and C, which is late and fails one mandatory condition.",
        f"B fails the budget test (${option_b_cost} versus ${budget}); C fails both the four-day deadline and one unwaivable requirement.",
        f"The ${budget} ceiling excludes B at ${option_b_cost}; the deadline and mandatory-condition checks exclude C.",
        f"B is ineligible on price because ${option_b_cost} is above ${budget}. C is ineligible on both duration and one required condition.",
        f"Applying the two hard limits leaves neither B nor C: B costs ${option_b_cost}, while C takes five days and misses a required condition.",
        f"Price removes B (${option_b_cost} against ${budget}); C cannot pass the four-day limit or the non-negotiable requirement.",
        f"B breaches the budget by costing ${option_b_cost} against a ${budget} maximum, and C violates two separate hard constraints.",
        f"The comparison drops B for its ${option_b_cost} price and C for its five-day duration plus the unmet mandatory condition.",
        f"Only A survives screening: B exceeds ${budget} at ${option_b_cost}, whereas C is late and misses one required condition.",
        f"B does not fit the ${budget} budget, and C's lower price cannot offset its late finish and failed required condition.",
        f"The budget test rejects B at ${option_b_cost}; independently, the deadline and requirement tests reject C.",
        f"Screening against the hard limits removes B for a ${option_b_cost} cost above ${budget} and removes C for both lateness and one missing condition.",
        f"A is the only compliant candidate: B exceeds the budget by ${option_b_cost - budget}, while C takes five days and omits a mandatory element.",
        f"B fails the ${budget} ceiling at ${option_b_cost}. C remains cheaper but cannot meet the four-day schedule or every required condition.",
        f"Neither alternative survives the gate: B is over budget, and C combines a five-day duration with an unmet requirement.",
        f"The price comparison excludes B because ${option_b_cost} is greater than ${budget}; the time and completeness checks separately exclude C.",
        f"B violates the financial cap by ${option_b_cost - budget}. C violates both the deadline and the requirement-completeness rule.",
        f"Under the fixed criteria, B's ${option_b_cost} price is disqualifying and C's low price cannot cure its late, incomplete plan.",
        f"The viable set contains only A after B misses the ${budget} cap and C misses both the four-day limit and one mandatory condition.",
        f"Budget compliance removes B at ${option_b_cost}; deadline and scope compliance remove C despite its lower price.",
        f"B cannot proceed because its cost is above ${budget}. C cannot proceed because five days is too long and one required item is absent.",
        f"Two independent checks reject the alternatives: price rejects B, while duration plus requirement coverage reject C.",
        f"Comparing each option with the hard gates leaves A alone; B costs ${option_b_cost}, and C is late with an incomplete requirement set.",
    )
    choices = (
        f"Choose A: {option_a}.",
        f"Choose Option A, {option_a}, as the compliant candidate.",
        f"Choose the viable option, A: {option_a}.",
        f"Select A: {option_a}.",
        f"Proceed with A: {option_a}.",
        f"Use Option A, {option_a}.",
        f"Pick A, {option_a}, because it meets every hard constraint.",
        f"A is the viable choice: {option_a}.",
        f"A is the only compliant option: {option_a}.",
        f"Option A meets the binding criteria: {option_a}.",
        f"The viable option is A: {option_a}.",
        f"The comparison leaves A as the choice: {option_a}.",
        f"Only A remains eligible: {option_a}.",
        f"The hard gates leave Option A: {option_a}.",
        f"A survives every required check, so choose {option_a}.",
        f"The compliant candidate is A, {option_a}.",
        f"The decision is A: {option_a}.",
        f"On the stated constraints, select A: {option_a}.",
        f"After screening the options, use A: {option_a}.",
        f"The feasible selection is A: {option_a}.",
        f"All binding tests point to A: {option_a}.",
        f"A alone satisfies the fixed requirements: {option_a}.",
        f"The shortlist resolves to A: {option_a}.",
        f"Under the hard limits, proceed with A: {option_a}.",
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
    return reasons, choices, sequences, fallbacks
