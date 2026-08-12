from __future__ import annotations


def planning_constraint_surfaces(constraint: str) -> tuple[str, ...]:
    """Return equivalent phrasings for one planning evidence boundary."""

    normalized = constraint.strip().rstrip(".")
    known = {
        "A fixed budget is a hard upper bound": (
            "The stated budget cannot be exceeded",
            "Cost must remain at or below the fixed ceiling",
            "No viable plan may cross the supplied spending limit",
            "The budget acts as a non-negotiable upper boundary",
        ),
        "A fixed deadline limits the feasible sequence": (
            "The sequence must finish within the fixed deadline",
            "Timing rules out any plan that completes too late",
            "Every viable ordering has to fit the supplied time limit",
            "The deadline constrains which sequence can be used",
        ),
        "One stated requirement cannot be traded away": (
            "The mandatory requirement must remain satisfied",
            "No trade-off may remove the stated non-negotiable condition",
            "A viable option has to preserve the required feature",
            "The fixed requirement cannot be exchanged for another benefit",
        ),
        "One important input remains uncertain": (
            "A decisive input has not yet been confirmed",
            "The plan must preserve uncertainty around one material fact",
            "One important value remains open and cannot be assumed",
            "Commitment should wait for the unresolved input",
        ),
        "Early steps should preserve the ability to change direction": (
            "Initial actions need to remain reversible",
            "The first steps must keep another direction available",
            "Early sequencing should avoid an irreversible commitment",
            "The plan needs flexibility until the evidence is stronger",
        ),
        "Stay within the stated resource ceiling": (
            "Resource use must remain under the supplied ceiling",
            "The plan cannot require capacity beyond the stated limit",
            "Every step has to fit the available resources",
            "The fixed resource boundary controls the feasible sequence",
        ),
    }
    return known.get(normalized, (normalized,))

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
        (
            "a two-stage move beginning with labeled low-priority items",
            "an expedited crew priced above the cap",
            "a bargain van plan that cannot protect the fragile load",
        ),
        (
            "a cancellable first-day transfer with an essentials inventory",
            "a premium door-to-door moving package",
            "a slow self-move missing the required key exchange",
        ),
        (
            "a room-sequenced move with a reversible booking",
            "an over-budget express removal service",
            "a low-price carrier without enough covered capacity",
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
        (
            "a step-free library workshop with a reservable side room",
            "a premium convention hall booking",
            "an inexpensive outdoor session without weather cover",
        ),
        (
            "a neighborhood session in an accessible civic room",
            "an express event-service package above budget",
            "a five-day venue plan lacking the mandatory access check",
        ),
        (
            "a small indoor gathering with a documented backup space",
            "a full-service commercial event bundle",
            "a cheap room whose stated capacity is insufficient",
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
        f"B cannot enter the shortlist at ${option_b_cost}, above the ${budget} maximum. C misses both the required timing and one mandatory feature.",
        f"The eligible set excludes B on cost and C on schedule plus completeness, leaving only the fully compliant candidate.",
        f"At ${option_b_cost}, B fails the financial boundary; C separately fails the deadline and required-scope boundaries.",
        f"Checking price, duration, and mandatory coverage removes B for price and C for two non-financial failures.",
        f"B's cost breaches the cap by ${option_b_cost - budget}, while C's five-day plan omits a required condition; neither qualifies.",
        f"The fixed tests disqualify B at the budget gate and C at both the schedule and requirement gates.",
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
        f"Screening identifies A as viable: {option_a}.",
        f"The constraints support A: {option_a}.",
        f"After applying every fixed condition, A remains: {option_a}.",
        f"The eligible candidate is A, namely {option_a}.",
        f"Only {option_a} passes the complete set of gates, so use A.",
        f"Select {option_a} as Option A after the other candidates fail mandatory checks.",
        f"The comparison resolves in favor of A, {option_a}.",
        f"A, {option_a}, is the supported selection under the supplied limits.",
    )
    sequences = (
        f"Sequence: confirm availability of {option_a} today, hold it reversibly, then verify every requirement before payment.",
        f"Sequence: verify {option_a} against the ${budget} cap and four-day deadline, request a reversible hold, then await written confirmation.",
        f"Sequence: check {option_a}'s availability, confirm every hard requirement under ${budget}, and make payment last.",
        f"Sequence: screen {option_a} against every fixed condition, secure a cancellable hold, and commit only after confirmation.",
        f"Sequence: ask whether {option_a} is available, document its compliance under ${budget}, then decide after the reply.",
        f"Sequence: preserve the option provisionally, verify timing and requirements, and leave payment until all checks pass.",
        f"Sequence: obtain written availability for {option_a}, recheck the hard limits, and convert the hold only at the end.",
        f"Sequence: validate cost and duration first, place a reversible hold on {option_a}, then confirm the remaining requirement.",
        f"Sequence: keep {option_a} uncommitted while verifying availability, full compliance, and the final payment condition.",
        f"Sequence: request confirmation from the provider, compare it with the ${budget} ceiling, and commit only if every gate remains satisfied.",
        f"Sequence: reserve {option_a} without payment, verify each mandatory condition in writing, then make the final choice.",
        f"Sequence: confirm that {option_a} can be delivered on time, check the complete requirement list, and pay last.",
        f"Sequence: use a cancellable hold for {option_a}, close the evidence gaps, and proceed only after the record is complete.",
        f"Sequence: verify the deadline, price, and mandatory features of {option_a} before turning the provisional choice into a commitment.",
        f"Sequence: keep the shortlist open, seek written confirmation for {option_a}, and finalize only when every hard test passes.",
        f"Sequence: first establish availability, next verify the ${budget} and timing limits, and finally authorize {option_a}.",
        f"Sequence: document why {option_a} qualifies, preserve a reversible position, and postpone payment until confirmation arrives.",
        f"Sequence: check the unresolved availability of {option_a}, confirm all binding facts, and make commitment the last step.",
        f"Sequence: place no irreversible order until {option_a} is available, compliant, and verified against the stated limits.",
        f"Sequence: obtain a provisional hold, test {option_a} against each requirement, and release payment only after written verification.",
        f"Sequence: ask for the current status of {option_a}, retain the evidence, and proceed after cost, deadline, and scope all pass.",
        f"Sequence: preserve reversibility while availability is checked, then validate every constraint before selecting {option_a} definitively.",
        f"Sequence: confirm {option_a} in writing, review its fit with the hard criteria, and convert the hold only when nothing remains uncertain.",
        f"Sequence: treat {option_a} as provisional until its availability and full compliance are established, with payment deferred.",
        f"Sequence: verify the candidate, retain a cancellable position, and authorize the plan only after the final requirement check.",
    )
    fallbacks = (
        f"Fallback trigger: if {option_a} cannot be confirmed by tomorrow, pause and reopen the shortlist rather than selecting either rejected candidate.",
        f"Fallback trigger: an unverified requirement for {option_a} means stopping to seek another option under ${budget}.",
        f"Fallback trigger: if the hold on {option_a} expires before confirmation, return to comparison instead of accepting a failed option.",
        f"Fallback trigger: if written availability for {option_a} does not arrive in time, keep the decision open and compare only new compliant candidates.",
        f"Fallback trigger: any newly failed hard condition removes {option_a} and starts a fresh search within the same limits.",
        f"Fallback trigger: if {option_a} exceeds ${budget} after verification, preserve the funds and rebuild the eligible shortlist.",
        f"Fallback trigger: a missed deadline or missing requirement means abandoning the provisional hold without using either excluded alternative.",
        f"Fallback trigger: if the provider cannot verify {option_a}, stop commitment and seek another option that passes every gate.",
        f"Fallback trigger: conflicting evidence about {option_a} keeps payment paused and returns the decision to comparison.",
        f"Fallback trigger: if any mandatory fact remains unknown at the decision point, release the hold and reopen the search.",
        f"Fallback trigger: when confirmation arrives too late, retain a reversible position and identify a replacement under ${budget}.",
        f"Fallback trigger: if {option_a} loses its compliant status, reject it and compare newly eligible alternatives from the beginning.",
        f"Fallback trigger: an expired provisional reservation sends the plan back to the hard-constraint screen.",
        f"Fallback trigger: if timing, price, or scope changes, pause the selection and rebuild the viable set.",
        f"Fallback trigger: absent written proof that {option_a} qualifies, make no payment and resume the constrained search.",
        f"Fallback trigger: if the final requirement check fails, keep B and C excluded and locate another compliant candidate.",
        f"Fallback trigger: if {option_a} is unavailable, return to the criteria rather than relaxing a hard condition.",
        f"Fallback trigger: if the reversible hold cannot be maintained through verification, stop and compare alternatives again.",
        f"Fallback trigger: any provider response that contradicts the recorded facts reopens the decision before commitment.",
        f"Fallback trigger: if {option_a} cannot satisfy all requirements simultaneously, preserve the current state and search again.",
        f"Fallback trigger: a new cost above ${budget} cancels the provisional choice and restores the full comparison step.",
        f"Fallback trigger: if confirmation remains incomplete tomorrow, leave the choice unresolved and seek another verified option.",
        f"Fallback trigger: when the candidate no longer meets the four-day limit, release it and restart with the same criteria.",
        f"Fallback trigger: if evidence for {option_a} is missing or inconsistent, do not infer compliance; return to the shortlist.",
        f"Fallback trigger: failure of any hard gate ends the provisional path and requires a fresh qualifying alternative.",
        f"Fallback trigger: if {option_a} is not fully documented as eligible, cancel the hold and apply every original constraint to a new candidate.",
        f"Fallback trigger: any unresolved conflict in price, timing, or scope keeps payment stopped and sends the choice back to review.",
        f"Fallback trigger: loss of written availability closes this path; preserve the budget and seek another candidate meeting all conditions.",
        f"Fallback trigger: if verification changes one binding fact, withdraw the provisional selection and rerun the hard-constraint comparison.",
        f"Fallback trigger: when {option_a} cannot be reserved reversibly, make no commitment and search within the original criteria.",
        f"Fallback trigger: a failed final check means preserving the current position while a different eligible option is identified.",
        f"Fallback trigger: if the provider leaves a mandatory condition unanswered, release the hold without weakening any requirement.",
        f"Fallback trigger: an adverse confirmation result removes {option_a}; restart the search under the unchanged ${budget} ceiling.",
    )
    return reasons, choices, sequences, fallbacks
