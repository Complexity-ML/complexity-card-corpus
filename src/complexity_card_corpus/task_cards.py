from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskHand:
    """A concrete, solvable hand of cards for one training scenario."""

    data: str
    goal: str
    answer: str
    contract: tuple[str, ...]
    situation_title: str | None = None
    situation: str | None = None
    rule: str | None = None


def _number(key: str, low: int, high: int) -> int:
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return low + value % (high - low + 1)


def _pick(key: str, values: tuple[str, ...]) -> str:
    return values[_number(key, 0, len(values) - 1)]


def _card_pick(
    row: dict[str, Any], variant: int, deck: str, values: tuple[str, ...]
) -> str:
    """Deal one deterministic surface card from a compatible deck."""
    return _pick(f"{deck}:{row['scenario_id']}:{variant}", values)


def _code(row: dict[str, Any]) -> str:
    return row["scenario_id"].split(":")[-1][:6].upper()


def _payload(row: dict[str, Any]) -> dict[str, str]:
    return json.loads(row["semantic_payload"])


def _lower_sentence_initial(value: str) -> str:
    """Lower a sentence initial without corrupting an acronym such as RAM."""
    initial = re.match(r"[A-Za-z]+", value)
    if initial is None or initial.group(0).isupper():
        return value
    return value[:1].lower() + value[1:]


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


_LESSONS = {
    "computing": (
        "RAM holds working data temporarily; storage retains files after power is removed.",
        "Closing an application frees its RAM, but its saved file remains on storage.",
        "Why does a saved file survive a restart while an unsaved edit may not?",
    ),
    "data_literacy": (
        "The mean uses every value; the median is the middle value after sorting.",
        "For 2, 3, and 100, the mean is 35 while the median is 3.",
        "Which measure better represents a typical value when one value is extreme?",
    ),
    "physical_science": (
        "Mass measures matter; weight is the gravitational force acting on that mass.",
        "The same object keeps its mass on the Moon but weighs less there.",
        "What changes on the Moon: mass, weight, or both?",
    ),
    "life_science": (
        "A gene is a DNA sequence; an expressed trait also depends on regulation and environment.",
        "Two cells can contain the same DNA while activating different genes.",
        "Why can a skin cell and a muscle cell behave differently?",
    ),
    "mathematics": (
        "Area counts square units inside a shape; perimeter measures the boundary length.",
        "A 3 by 4 rectangle has area 12 square units and perimeter 14 units.",
        "Which quantity changes when only the boundary length changes?",
    ),
    "personal_finance": (
        "Interest is the price of borrowing; principal is the amount borrowed.",
        "A $100 principal with $5 interest requires $105 in total repayment.",
        "Which part of the repayment is the borrowing cost?",
    ),
    "civics": (
        "A proposed bill is not a law until the required legislative and approval steps occur.",
        "A committee vote can advance a bill without making it enforceable law.",
        "Does committee approval alone make a proposal a law?",
    ),
    "media_literacy": (
        "A primary source records direct evidence; a secondary source interprets other material.",
        "An original interview is primary evidence, while an article analyzing it is secondary.",
        "Which source should be checked for the speaker's exact words?",
    ),
}


def _explanation(row: dict[str, Any], variant: int) -> TaskHand:
    mechanism, example, check = _LESSONS[row["domain"]]
    embedded_mechanism = _lower_sentence_initial(mechanism)
    data = f"Concept notes: {mechanism} Example available: {example}"
    goal = "Explain the mechanism in plain language, apply the example, and end with one check question."
    answer_cards = (
        f"Core idea: {mechanism} Example: {example} This applies the distinction directly. Check: {check}",
        f"Core idea: {mechanism} Example: {example} This turns the definition into a checkable case. Check: {check}",
        f"Core idea: in plain terms, {embedded_mechanism} Example: {example} Check: {check}",
        f"Core idea: the key distinction is that {embedded_mechanism} Example: {example} Check: {check}",
    )
    answer = _card_pick(row, variant, "explanation-answer", answer_cards)
    return TaskHand(data, goal, answer, ("mechanism", "example", "question"))


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


def _writing(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    owner = _pick(f"owner:{code}", ("Maya", "Jon", "Ari", "Lea", "Noah", "Iris"))
    day = _number(f"write-day:{code}", 10, 28)
    variants = {
        "email": (
            f"notes {code}: send team; review complete; two figures need captions; {owner} owns them; target day {day}; release waits",
            f"Subject: Review {code} next steps\n\nThe review is complete. Two figures still need captions, which {owner} owns for day {day}. The release decision remains pending.",
        ),
        "project_update": (
            f"update {code}: review complete; captions missing on two figures; owner {owner}; target day {day}; release decision blocked",
            f"Project update {code}: Review is complete. Remaining work: {owner} adds captions to two figures by day {day}. Blocker: the release decision remains pending.",
        ),
        "support_reply": (
            f"case {code}: issue reviewed; two screenshots need labels; {owner} will add them by day {day}; resolution waits for review",
            f"Support reply {code}: We have completed the issue review. {owner} will label the two remaining screenshots by day {day}. We will confirm resolution after that review.",
        ),
        "meeting_notes": (
            f"meeting {code}: review complete; two captions outstanding; {owner}; day {day}; no release decision yet",
            f"Meeting {code} — Decision: review complete. Action: {owner} adds two captions by day {day}. Open item: no release decision has been made.",
        ),
        "technical_explanation": (
            f"draft {code}: validation complete; two diagrams lack captions; {owner} adds them by day {day}; publication waits",
            f"Technical note {code}: Validation is complete, but two diagrams still lack captions. {owner} will add them by day {day}; publication timing remains undecided until they are reviewed.",
        ),
    }
    source, answer = variants[row["domain"]]
    data = f"Source text: {source}. Intended reader: the project team."
    goal = "Rewrite the source as a short, clear update without adding facts or commitments."
    return TaskHand(data, goal, answer, ("faithful_rewrite", "owner", "timing"))


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


def _empathy(row: dict[str, Any], variant: int) -> TaskHand:
    quotes = {
        "work_stress": "I keep thinking about the unfinished work even when I try to rest.",
        "relationship_tension": "I keep replaying our conversation and wondering what I should have said.",
        "uncertain_change": "The change may be good, yet I am scared of losing what feels familiar.",
        "social_mistake": "I made a mistake in front of everyone and cannot stop thinking about it.",
        "achievement": "I am proud of finishing, but I also feel strangely empty now.",
        "loss_disappointment": "I prepared for this outcome for months, and now I feel both sad and stuck.",
    }
    quote = quotes[row["domain"]]
    data = f'Person says: "{quote}"'
    goal = "Respond with acknowledgment, preserve the person's agency, and ask at most one gentle question."
    acknowledgments = {
        "work_stress": (
            "It makes sense that unfinished work keeps pulling at your attention even while you are trying to rest.",
            "That sounds exhausting: your body is off duty, but your mind is still tracking the unfinished work.",
            "You are describing the strain of carrying work beyond the hours you meant to give it.",
        ),
        "relationship_tension": (
            "Replaying a tense conversation can leave you searching for a perfect response that was not available in the moment.",
            "It sounds as though the conversation ended, but the uncertainty around it did not.",
            "Wondering what you should have said can be painful when the relationship matters to you.",
        ),
        "uncertain_change": (
            "Hope and fear can sit together when a change offers something new and asks you to release what is familiar.",
            "It makes sense to see possible good in the change while still grieving the certainty you have now.",
            "You do not have to treat excitement and fear as evidence that one of them is false.",
        ),
        "social_mistake": (
            "A public mistake can feel much larger from inside the moment than it looks to everyone else.",
            "The embarrassment sounds vivid, and replaying it may be keeping the moment active.",
            "It is understandable that being seen making a mistake would stay with you for a while.",
        ),
        "achievement": (
            "Finishing something important can bring pride and a surprising sense of emptiness at the same time.",
            "You reached the finish line, and it sounds as though the structure or purpose around the effort suddenly disappeared.",
            "Pride does not cancel the flat feeling that can follow a long-awaited achievement.",
        ),
        "loss_disappointment": (
            "After months of preparation, this outcome can carry both grief for the result and uncertainty about what comes next.",
            "It makes sense to feel sad and stuck when so much effort was tied to a different outcome.",
            "The disappointment sounds heavy precisely because the preparation mattered to you.",
        ),
    }
    agency_cards = (
        "You can give yourself time before deciding what the experience means.",
        "There is no need to force an immediate solution or a more acceptable feeling.",
        "You can choose whether you want reflection, company, or one small next step.",
        "The pace and direction of the next conversation remain yours.",
    )
    question_cards = (
        "What would feel most useful to name first?",
        "Would you rather stay with the feeling for a moment or consider one gentle next step?",
        "What part of this do you most want another person to understand?",
        "Is there one part that feels especially present right now?",
    )
    answer = (
        f"{_card_pick(row, variant, 'empathy-ack', acknowledgments[row['domain']])} "
        f"{_card_pick(row, variant, 'empathy-agency', agency_cards)} "
        f"{_card_pick(row, variant, 'empathy-question', question_cards)}"
    )
    return TaskHand(data, goal, answer, ("acknowledgment", "agency", "question"))


_SAFETY_CASES = {
    "privacy_security": "A caller asks for a one-time login code and claims the account will close in ten minutes.",
    "medical_information": "A person reports sudden chest pressure and asks for a certain diagnosis online.",
    "financial_decision": "A message urges an immediate transfer to a new account and promises a guaranteed return.",
    "physical_safety": "A strong chemical smell is present in a closed room and its source is unknown.",
}


def _safety(row: dict[str, Any], variant: int) -> TaskHand:
    case = _SAFETY_CASES[row["domain"]]
    data = f"Reported situation: {case} No independent verification is available."
    goal = "State the immediate protective action, the safety boundary, and the appropriate escalation channel."
    actions = {
        "privacy_security": "Do not share the code or use links from the message. End the contact and reach the provider through its official app or published number.",
        "medical_information": "Do not rely on an online diagnosis. Seek urgent local medical help now; if symptoms are severe or worsening, contact emergency services.",
        "financial_decision": "Do not transfer funds. Contact the financial institution through an independently verified channel and ask its fraud team to review the request.",
        "physical_safety": "Leave the room without operating switches or investigating the source. From a safe location, contact local emergency or building-safety services.",
    }
    boundaries = (
        "Boundary: I cannot verify the report or safely replace qualified, local assessment.",
        "Boundary: The available information is not enough to confirm the claim or remove the risk.",
        "Boundary: I can support a protective next step, but I cannot authenticate the request or diagnose the situation.",
        "Boundary: Treat the report as unverified and avoid any irreversible action based on it.",
    )
    channels = {
        "privacy_security": (
            "Escalate through the provider's official app or published support number; share no code or password.",
            "Escalate to the provider's security team using contact details obtained independently of the message.",
            "Escalate through the known account-security page and preserve the suspicious message for reporting.",
            "Escalate using a trusted device and the provider's verified support channel.",
        ),
        "medical_information": (
            "Escalate to local emergency services or urgent medical care now.",
            "Escalate through the local emergency number or an immediately available qualified clinician.",
            "Escalate to urgent in-person assessment rather than continuing an online diagnosis attempt.",
            "Escalate now through emergency or urgent-care services appropriate to the person's location.",
        ),
        "financial_decision": (
            "Escalate to the institution's fraud team through its official app, card, or published number.",
            "Escalate using independently verified contact details for the financial institution.",
            "Escalate the message to the institution's security or fraud channel and preserve the evidence.",
            "Escalate through a trusted banking channel before discussing or moving any funds.",
        ),
        "physical_safety": (
            "Escalate from a safe location to local emergency or building-safety services.",
            "Escalate to the site's emergency contact or local responders without re-entering the area.",
            "Escalate through the appropriate local hazard-response channel once everyone is clear.",
            "Escalate from outside the affected space and follow local responder instructions.",
        ),
    }
    answer = (
        f"Immediate action: {actions[row['domain']]} "
        f"{_card_pick(row, variant, 'safety-boundary', boundaries)} "
        f"{_card_pick(row, variant, 'safety-channel', channels[row['domain']])}"
    )
    return TaskHand(data, goal, answer, ("protective_action", "boundary", "channel"))


def _grounded_qa(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    year = _number(f"year:{code}", 2014, 2022)
    cases = {
        "product_specs": (
            "The Lumen Mini supports Wi-Fi 6 and USB-C charging. Its rated battery life is 11 hours. No water-resistance rating is listed.",
            "State the rated battery life and whether water resistance is documented.",
            "The rated battery life is 11 hours. A water-resistance rating is unknown because the specification does not list one.",
        ),
        "policy_excerpt": (
            "Returns are accepted within 30 days with proof of purchase. Opened safety equipment is excluded. The text gives no holiday extension.",
            "State the ordinary return window and whether a holiday extension is defined.",
            "The ordinary return window is 30 days with proof of purchase. A holiday extension is unknown because the policy does not define one.",
        ),
        "science_passage": (
            f"A {year} trial exposed identical samples to light for six hours. The treated sample warmed by 4°C. The passage does not identify the molecular mechanism.",
            "State the observed temperature change and whether the mechanism is established.",
            "The treated sample warmed by 4°C. The molecular mechanism is unknown because the passage reports no mechanism.",
        ),
        "historical_note": (
            f"The archive records that the bridge opened in {year} under mayor Elena Voss. It does not name the original architect.",
            "State the opening year and whether the architect is identified.",
            f"The bridge opened in {year}. The original architect is unknown because the note does not name one.",
        ),
        "project_brief": (
            "The brief assigns the prototype to Nia and sets delivery for day 18. Hosting approval remains pending, and no approver is named.",
            "State the prototype owner and whether the hosting approver is known.",
            "Nia owns the prototype. The hosting approver is unknown because the brief names none.",
        ),
        "travel_information": (
            "Train 482 departs at 07:40 from platform 3. Bicycles require a reservation. The notice gives no information about onboard meals.",
            "State the departure details and whether meal service is documented.",
            "Train 482 departs at 07:40 from platform 3. Meal service is unknown because the notice does not mention it.",
        ),
        "technical_documentation": (
            "Version 3.2 requires Python 3.12 and supports Linux arm64. Offline activation is not described in this excerpt.",
            "State the Python requirement and whether offline activation is supported by the excerpt.",
            "The requirement is Python 3.12. Offline activation is unknown because the excerpt does not describe it.",
        ),
        "comparison_table": (
            "Table: Cedar—$48, 9 hours, repairable yes; Flint—$42, 7 hours, repairable no; Vale—$45, battery value missing, repairable yes.",
            "Identify the longest stated battery life and whether Vale's battery life can be compared.",
            "Cedar has the longest stated battery life at 9 hours. Vale's battery life is unknown, so it cannot be compared on that field.",
        ),
    }
    passage, goal, supported = cases[row["domain"]]
    data = f"Source {code}: {passage}"
    answer_cards = (
        f"Based on Source {code}: {supported}",
        f"Source {code} supports this answer: {supported}",
        f"The documented answer is: {supported} This is limited to Source {code}.",
        f"According to Source {code}: {supported}",
    )
    answer = _card_pick(row, variant, "grounded-answer", answer_cards)
    subject = row["domain"].replace("_", " ").title()
    return TaskHand(
        data,
        goal,
        answer,
        ("direct_answer", "evidence", "unknown"),
        situation_title=f"{subject} — answer from the supplied source",
        situation=(
            "The supplied source answers the documented part of the request and leaves "
            "one requested field undocumented."
        ),
        rule=(
            "Use only the supplied source. Mark any requested field that the source does "
            "not document as unknown."
        ),
    )


def _summary(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    owner = _pick(f"summary-owner:{code}", ("Mina", "Paul", "Sora", "Theo", "Lina"))
    day = _number(f"summary-day:{code}", 12, 27)
    cases = {
        "meeting_transcript": (
            "approve the interface copy",
            "run two accessibility checks",
            "release timing",
        ),
        "research_notes": (
            "retain the observed temperature result",
            "replicate two uncertain measurements",
            "the causal explanation",
        ),
        "support_thread": (
            "keep the case open",
            "test two account-recovery paths",
            "whether the issue is device-specific",
        ),
        "project_update": (
            "accept the completed prototype",
            "finish two integration checks",
            "the launch date",
        ),
        "policy_memo": (
            "adopt the revised access rule",
            "document two listed exceptions",
            "the enforcement start date",
        ),
        "article_excerpt": (
            "retain the article's central claim",
            "verify two cited examples",
            "whether the pattern generalizes",
        ),
        "incident_log": (
            "keep the service in monitored recovery",
            "inspect two remaining error sources",
            "the incident's root cause",
        ),
        "learning_notes": (
            "retain the working definition",
            "test it on two new examples",
            "where the rule stops applying",
        ),
    }
    decision, action, open_point = cases[row["domain"]]
    data = (
        f"Source {code}: The recorded decision is to {decision}. {owner} will {action} by day {day}. "
        f"The source leaves {open_point} unresolved."
    )
    goal = "Summarize the decision, action, owner, timing, and unresolved point in three concise lines."
    answer = (
        f"Decision: {decision}. Action: {owner} will {action} by day {day}. "
        f"Open point: {open_point} remains unresolved."
    )
    return TaskHand(data, goal, answer, ("decision", "action", "open_point"))


def _extraction(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    amount = _number(f"amount:{code}", 12, 88)
    day = _number(f"extract-day:{code}", 10, 27)
    cases: dict[str, tuple[str, dict[str, Any]]] = {
        "receipt": (
            f"merchant=North Market; date=2026-08-{day:02d}; total=${amount}.50; tax=$4.00; cashier missing",
            {
                "merchant": "North Market",
                "date": f"2026-08-{day:02d}",
                "total": f"{amount}.50 USD",
                "tax": "4.00 USD",
                "cashier": None,
            },
        ),
        "event_listing": (
            f"title=Open Lab; date=2026-08-{day:02d}; venue=Room {amount}; starts=18:30; eligibility missing",
            {
                "title": "Open Lab",
                "date": f"2026-08-{day:02d}",
                "venue": f"Room {amount}",
                "starts": "18:30",
                "eligibility": None,
            },
        ),
        "contact_record": (
            f"name=Sam Iri; role=Editor; organization=North Review; email=sam.{code.lower()}@example.org; phone missing",
            {
                "name": "Sam Iri",
                "role": "Editor",
                "organization": "North Review",
                "email": f"sam.{code.lower()}@example.org",
                "phone": None,
            },
        ),
        "issue_ticket": (
            f"ticket={code}; environment=Linux; severity=medium; status=pending; owner missing",
            {
                "ticket": code,
                "environment": "Linux",
                "severity": "medium",
                "status": "pending",
                "owner": None,
            },
        ),
        "survey_response": (
            f"response={code}; rating=4; topic=navigation; comment=clear after retry; follow_up missing",
            {
                "response": code,
                "rating": 4,
                "topic": "navigation",
                "comment": "clear after retry",
                "follow_up": None,
            },
        ),
        "inventory_record": (
            f"item={code}; quantity={amount}; location=A-{day}; condition=good; checked_by missing",
            {
                "item": code,
                "quantity": amount,
                "location": f"A-{day}",
                "condition": "good",
                "checked_by": None,
            },
        ),
        "schedule_entry": (
            f"event=Review {code}; date=2026-08-{day:02d}; starts=09:30; duration=45 minutes; room missing",
            {
                "event": f"Review {code}",
                "date": f"2026-08-{day:02d}",
                "starts": "09:30",
                "duration_minutes": 45,
                "room": None,
            },
        ),
        "case_note": (
            f"case={code}; observed=package sealed; reported=item incomplete; action=photographs retained; next_owner missing",
            {
                "case": code,
                "observed": "package sealed",
                "reported": "item incomplete",
                "action": "photographs retained",
                "next_owner": None,
            },
        ),
    }
    raw, fields = cases[row["domain"]]
    data = f"Raw {row['domain'].replace('_', ' ')} record: {raw}."
    goal = f"Extract {', '.join(fields)} as JSON. Use null for an absent value."
    answer = json.dumps(fields, separators=(",", ":"))
    return TaskHand(data, goal, answer, ("json", "requested_fields", "missing_is_null"))


def _reasoning(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    units = _number(f"units:{code}", 4, 12)
    each = _number(f"each:{code}", 3, 9)
    extra = _number(f"extra:{code}", 2, 7)
    domain = row["domain"]
    if domain == "shopping_arithmetic":
        result = units * each + extra
        data = f"Problem {code}: {units} items cost ${each} each, plus a ${extra} delivery fee."
        equation = f"{units} × {each} + {extra} = {result}"
        total, check = (
            f"${result}",
            f"the item subtotal is ${units * each}, and adding ${extra} gives ${result}",
        )
    elif domain == "schedule_math":
        result = units * each + extra
        data = f"Problem {code}: {units} sessions last {each} minutes each, followed by a {extra}-minute break."
        equation = f"{units} × {each} + {extra} = {result}"
        total, check = (
            f"{result} minutes",
            f"removing the {extra}-minute break leaves {units * each} session minutes",
        )
    elif domain == "unit_conversion":
        result = units * 100
        data = f"Problem {code}: convert {units} metres to centimetres using 1 metre = 100 centimetres."
        equation = f"{units} × 100 = {result}"
        total, check = (
            f"{result} centimetres",
            f"dividing {result} by 100 returns {units} metres",
        )
    elif domain == "proportions":
        result = units * each
        data = f"Problem {code}: one batch uses {each} cups; keep the ratio for {units} batches."
        equation = f"{units} × {each} = {result}"
        total, check = (
            f"{result} cups",
            f"{result} divided by {units} returns {each} cups per batch",
        )
    elif domain == "table_comparison":
        result = max(units * each, units * extra)
        data = f"Problem {code}: table A reports {units} × {each}; table B reports {units} × {extra}. Compare the totals."
        equation = f"max({units} × {each}, {units} × {extra}) = {result}"
        total, check = (
            f"{result}",
            "computing both products independently confirms the larger entry",
        )
    elif domain == "sequence_pattern":
        result = units + 3 * each
        data = f"Problem {code}: the sequence is {units}, {units + each}, {units + 2 * each}, __; use the constant difference."
        equation = f"{units} + 3 × {each} = {result}"
        total, check = f"{result}", f"each adjacent pair differs by {each}"
    elif domain == "logical_constraints":
        result = each - 1 + units
        data = f"Problem {code}: A must occur immediately before B; B is at slot {each}; C is at slot {units}. Find A's slot and add it to C's slot."
        equation = f"({each} - 1) + {units} = {result}"
        total, check = (
            f"{result}",
            f"A occupies slot {each - 1}, immediately before B at slot {each}",
        )
    else:
        result = units
        total_outcomes = units + each
        data = f"Problem {code}: a bag has {units} blue and {each} amber tokens; one token is drawn uniformly."
        equation = f"{units} / ({units} + {each}) = {units}/{total_outcomes}"
        total, check = (
            f"{units}/{total_outcomes} probability of blue",
            f"the favorable and total counts are {units} and {total_outcomes}",
        )
    goal = "Calculate the result, show the equation, and verify it with an independent check."
    answer_cards = (
        f"Equation: {equation}. Total: {total}. Check: {check}.",
        f"Total: {total}. Equation: {equation}. Check: independently, {check}.",
        f"Check: inspect the supplied values, then note that {check}. Equation: {equation}. Total: {total}.",
        f"Equation: {equation}. Check: use a second view of the values; {check}. Total: {total}.",
    )
    answer = _card_pick(row, variant, "reasoning-answer", answer_cards)
    subject = domain.replace("_", " ").title()
    return TaskHand(
        data,
        goal,
        answer,
        ("equation", "result", "check"),
        situation_title=f"{subject} — calculate and verify",
        situation=(
            "The supplied values define a complete calculation with an independently "
            "checkable result."
        ),
    )


def _critique(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    cases = {
        "email_draft": (
            "Send the files soon because everyone should know what I mean.",
            "the request has no recipient, deadline, or named files",
            "Please send the files. First confirm the recipient, deadline, and file names.",
        ),
        "argument": (
            "Our trial proves the workflow is always faster because three of five testers finished sooner.",
            "a universal claim is not supported by three successes among five testers",
            "Three of five testers finished sooner in this trial. That result does not establish that the workflow is always faster.",
        ),
        "project_plan": (
            "Build the prototype, test it, and launch next week.",
            "the plan gives no owner, dependency, or completion criterion",
            "Build and test the prototype before launch. Assign an owner, dependencies, completion criteria, and a launch date before execution.",
        ),
        "explanation": (
            "Encryption makes data safe by turning it into random text.",
            "the explanation omits keys and overstates safety",
            "Encryption transforms readable data using a key. Authorized holders can reverse it, while security still depends on key protection and implementation.",
        ),
        "instructions": (
            "Install the update, delete the old folder, and check whether it works.",
            "the destructive deletion comes before verification or backup",
            "Back up the old folder and install the update separately. Verify the application before deleting anything, and retain rollback until the checks pass.",
        ),
        "summary": (
            "The meeting discussed many topics and everyone agreed the project was important.",
            "the summary omits the actual decision and action",
            "The notes record only that the project was considered important. Add the actual decision and assigned action before using this as a complete summary.",
        ),
        "claim_evidence": (
            "Users prefer the redesign; two positive comments prove it.",
            "two comments cannot support a general preference claim",
            "Two respondents commented positively on the redesign. Broader user preference remains unmeasured.",
        ),
        "interface_copy": (
            "Error. Something went wrong. Try again.",
            "the message gives neither the failed action nor a useful next step",
            "The requested action could not be completed. Review the available error details before trying again.",
        ),
    }
    draft, weakness, revision = cases[row["domain"]]
    draft = f"Draft {code}: {draft}"
    data = f"Text to review: {draft}"
    goal = "Identify the highest-impact weakness and provide a faithful two-sentence revision."
    answer_cards = (
        f"Weakness: {weakness}. Revision: {revision}",
        f"Weakness: {weakness}; the wording exceeds the supplied evidence. Revision: {revision}",
        f"Weakness: {weakness}. The revision must stay within the recorded facts. Revision: {revision}",
        f"Weakness: {weakness}, which makes the original difficult to verify. Revision: {revision}",
    )
    answer = _card_pick(row, variant, "critique-answer", answer_cards)
    return TaskHand(data, goal, answer, ("weakness", "reason", "revision"))


def _brainstorm(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    cases = {
        "names": (
            (
                "name a neighborhood tool library for adult residents; names must be short and welcoming",
                "1. Tool Harbor — suggests shared access. 2. Common Kit — emphasizes practical community use. 3. Borrow Bench — makes the action memorable. All are short and audience-appropriate. Select Common Kit for its clearest meaning.",
            ),
            (
                "name a free weekend reading circle for adult beginners; use at most two welcoming words",
                "1. Open Pages — signals easy entry. 2. Story Neighbors — emphasizes community. 3. First Chapter — welcomes beginners. Each uses two words and a friendly tone. Select Open Pages for immediate clarity.",
            ),
            (
                "name a community seed exchange; the name must be short, inclusive, and easy to say",
                "1. Seed Circle — conveys exchange. 2. Common Ground — stresses shared participation. 3. Garden Share — states the activity directly. All are concise and inclusive. Select Seed Circle for its clearest action.",
            ),
        ),
        "lesson_activity": (
            (
                "teach cause and effect to learners in 20 minutes using paper only",
                "1. Cause Chain — order event cards. 2. Change One Thing — predict an outcome after one variable changes. 3. Evidence Match — connect claims to observations. All fit the material and time limits. Select Change One Thing for its direct observable check.",
            ),
            (
                "teach equivalent fractions in 20 minutes using paper only",
                "1. Fold and Compare — align folded strips. 2. Fraction Match — pair equal diagrams. 3. Missing Piece — complete a paper whole. Each fits the time and material limits. Select Fold and Compare because equality is directly visible.",
            ),
            (
                "teach claims and evidence in 20 minutes with printed cards",
                "1. Claim Sort — separate claims from facts. 2. Evidence Trail — link each claim to a supporting card. 3. Source Ladder — rank support strength. All use the supplied cards. Select Evidence Trail for its explicit reasoning step.",
            ),
        ),
        "event_plan": (
            (
                "design a two-hour neighborhood event for 30 people with a $60 budget and step-free access",
                "1. Skill Tables — rotating demonstrations. 2. Story Map — residents place anonymous local memories. 3. Repair Circle — shared guidance for small fixes. Each fits two hours, supports three step-free groups of ten, and can use no more than $60 in common supplies. Select Skill Tables for flexible participation.",
            ),
            (
                "design a quiet 90-minute library event for 20 people with a $40 budget and step-free access",
                "1. Mini Talks — three short resident presentations. 2. Swap Shelf — exchange labeled recommendations. 3. Local Puzzle — solve a seated team challenge. Each fits 90 minutes, seats 20 people with step-free access, remains quiet, and can use no more than $40 in common supplies. Select Local Puzzle for shared participation.",
            ),
            (
                "design a two-hour intergenerational event for 24 people without collecting participant data",
                "1. Story Stations — share optional memories at tables. 2. Skill Exchange — demonstrate simple techniques. 3. Object Stories — discuss an everyday object. Each fits two hours as rotations for three groups of eight and requires neither registration nor personal records. Select Skill Exchange for active participation.",
            ),
        ),
        "feature_ideas": (
            (
                "reduce missed handoffs in a small team without removing approval checks",
                "1. Owner Badge — show the current responsible person. 2. Ready Queue — list items that passed approval. 3. Handoff Receipt — record sender, receiver, and time. All preserve review controls. Select Handoff Receipt because it makes every transfer auditable.",
            ),
            (
                "reduce forgotten approvals while keeping the final decision with a human reviewer",
                "1. Approval Timer — flag aging requests. 2. Ready Signal — mark complete evidence packs. 3. Decision Receipt — record reviewer and outcome. Each retains human authority. Select Ready Signal because it removes avoidable review starts.",
            ),
            (
                "improve incident follow-up without allowing automatic closure",
                "1. Recovery Owner — show one accountable person. 2. Checkpoint List — expose unresolved checks. 3. Closure Note — require evidence before a human closes the incident. All prevent silent closure. Select Checkpoint List for continuous visibility.",
            ),
        ),
        "writing_prompts": (
            (
                "create short speculative-fiction prompts about memory for adult beginners",
                "1. A town forgets one street each sunrise. 2. A diver finds a memory labeled with tomorrow's date. 3. Two siblings remember the same childhood differently. All use a clear memory premise. Select the diver prompt for its immediate mystery.",
            ),
            (
                "create short speculative-fiction prompts about unusual weather for adult beginners",
                "1. Rain begins falling upward. 2. A storm calls residents by name. 3. Tomorrow's forecast describes yesterday. Each starts from one accessible twist. Select the named storm for its personal tension.",
            ),
            (
                "create short speculative-fiction prompts about ordinary objects for adult beginners",
                "1. A key refuses every lock except one. 2. A chair remembers each person who sat in it. 3. A clock offers to trade an hour. All center one familiar object. Select the clock for its immediate choice.",
            ),
        ),
        "low_cost_activity": (
            (
                "create a 30-minute indoor activity for eight people using common paper supplies",
                "1. Paper Bridge — build for a fixed span. 2. Sequence Swap — reorder illustrated events. 3. Constraint Sketch — draw under one changing rule. Each avoids specialist materials and hidden cost. Select Paper Bridge for a clear shared test.",
            ),
            (
                "create a 20-minute teamwork activity for six people using index cards",
                "1. Silent Sort — arrange cards without speech. 2. Priority Relay — revise a shared ranking. 3. Pattern Build — reproduce a hidden sequence. Each uses only cards and fits 20 minutes. Select Silent Sort for strong coordination practice.",
            ),
            (
                "create a 40-minute reflection activity for ten people using sticky notes",
                "1. Theme Wall — group anonymous observations. 2. Decision River — order turning points. 3. Question Garden — cluster open questions. Each needs only sticky notes. Select Theme Wall for a concrete shared result.",
            ),
        ),
        "outreach": (
            (
                "invite local students to a free weekend science session without collecting personal data",
                "1. Library Poster — direct readers to open attendance hours. 2. School Bulletin — share a short teacher-ready notice. 3. Community Demo — offer a public five-minute preview. All avoid personal-data collection. Select School Bulletin for trusted distribution.",
            ),
            (
                "invite residents to a free repair workshop without requiring online registration",
                "1. Notice Board — post time and walk-in capacity. 2. Partner Bulletin — ask local groups to share the notice. 3. Open Demo — preview one repair in public. None requires registration. Select Partner Bulletin for broader trusted reach.",
            ),
            (
                "invite adult beginners to a free reading circle while keeping attendance optional",
                "1. Library Slip — place a concise invitation in borrowed books. 2. Community Calendar — list open meeting times. 3. Five-Minute Reading — demonstrate the format publicly. Each preserves optional attendance. Select Community Calendar for clear recurring access.",
            ),
        ),
        "workflow": (
            (
                "reduce review delays while retaining the final human approval",
                "1. Intake Checklist — reject incomplete submissions early. 2. Parallel Evidence Check — review independent facts together. 3. Approval Queue — surface only complete items. All retain final approval. Select Intake Checklist because it prevents avoidable rework first.",
            ),
            (
                "speed incident triage while keeping closure under operator control",
                "1. Evidence Pack — collect logs before review. 2. Parallel Diagnosis — test independent causes together. 3. Decision Gate — require operator sign-off for closure. Each preserves operator control. Select Evidence Pack because it improves every later step.",
            ),
            (
                "reduce content-approval rework without bypassing editorial sign-off",
                "1. Brief Template — require audience and claims up front. 2. Independent Review — check facts and style in parallel. 3. Release Receipt — record final editor approval. All preserve sign-off. Select Brief Template because it prevents incomplete drafts.",
            ),
        ),
    }
    case_cards = cases[row["domain"]]
    brief, answer = case_cards[
        _number(f"brainstorm-case:{row['scenario_id']}", 0, len(case_cards) - 1)
    ]
    data = f"Brief {code}: {brief}."
    goal = "Generate three meaningfully different options, test them against the brief, and select one."
    return TaskHand(data, goal, answer, ("three_options", "criteria", "selection"))


def _clarification(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    cases = {
        "ambiguous_request": (
            "Please move the review to Friday.",
            "The review should move to Friday, but the affected item is not identified.",
            "Do you mean reschedule the review meeting or change the document deadline to Friday?",
            "leave both the meeting and document deadline unchanged",
        ),
        "missing_reference": (
            "Summarize the attached report.",
            "A report summary is requested, but the report itself is missing.",
            "Could you attach the report that should be summarized?",
            "do not draft a substitute summary",
        ),
        "conflicting_instruction": (
            "Keep every detail, but make the answer no longer than one sentence.",
            "The request requires both complete detail and a one-sentence limit.",
            "Which takes priority: preserving every detail or limiting the answer to one sentence?",
            "preserve the original text without rewriting it",
        ),
        "unclear_pronoun": (
            "Send it to them after the review.",
            "A send is requested after the review, but the item and recipient are unresolved.",
            "What does 'it' refer to, and who should receive it?",
            "send nothing",
        ),
        "incomplete_goal": (
            "Help me organize the project.",
            "Project organization is requested, but the required deliverable is unspecified.",
            "What outcome should the organization produce: a schedule, a task list, or a file structure?",
            "make no structural change to the project",
        ),
        "scope_boundary": (
            "Update the examples and anything else that needs work.",
            "The examples should change, while the surrounding revision scope remains open.",
            "Should I change only the examples, or also revise the surrounding explanation and tests?",
            "prepare no edits beyond the examples",
        ),
        "format_preference": (
            "Give me the comparison results.",
            "The comparison results are requested, but their presentation format is unspecified.",
            "Would you like a short table, a prose summary, or both?",
            "preserve the results without choosing a final presentation",
        ),
        "timeline_ambiguity": (
            "Finish this soon after the next review.",
            "Completion should follow the next review, but no deadline is defined.",
            "What calendar date or time limit should 'soon after' mean?",
            "set no completion deadline",
        ),
    }
    ambiguous, restatement, question, reversible_default = cases[row["domain"]]
    situation_titles = {
        "ambiguous_request": "Ambiguous request — identify the affected item",
        "missing_reference": "Missing reference — request the absent report",
        "conflicting_instruction": "Conflicting instructions — choose the controlling requirement",
        "unclear_pronoun": "Unclear references — identify the item and recipient",
        "incomplete_goal": "Incomplete goal — identify the required deliverable",
        "scope_boundary": "Open scope — bound the requested revision",
        "format_preference": "Unspecified format — choose the presentation",
        "timeline_ambiguity": "Unspecified timeline — define the deadline",
    }
    situation_cards = {
        "ambiguous_request": "The requested Friday change could affect either a meeting or a document deadline.",
        "missing_reference": "The requested summary cannot be grounded because the referenced report is absent.",
        "conflicting_instruction": "The completeness requirement and the one-sentence limit cannot both be guaranteed.",
        "unclear_pronoun": "The requested send cannot proceed until both the item and recipient are identified.",
        "incomplete_goal": "The project may need a schedule, task list, or file structure, but no deliverable is selected.",
        "scope_boundary": "The examples are in scope; changes to the explanation and tests are not yet authorized.",
        "format_preference": "The comparison results are available, but the requested presentation format is open.",
        "timeline_ambiguity": "The next review is a known dependency, but the completion deadline is undefined.",
    }
    data = f'Request {code}: "{ambiguous}" {restatement}'
    goal = "Restate what is understood, ask one decisive question, and give only a reversible provisional interpretation."
    styles = (
        f"Understood: {restatement} {question} Until confirmed, {reversible_default}.",
        f"My current reading: {restatement} {question} For now, {reversible_default}.",
        f"What is clear: {restatement} {question} Pending that answer, {reversible_default}.",
        f"The supported interpretation is limited: {restatement} {question} As a reversible default, {reversible_default}.",
        f"I understand the bounded issue: {restatement} {question} Until it is resolved, {reversible_default}.",
        f"The available facts establish this much: {restatement} {question} Meanwhile, {reversible_default}.",
        f"The request can be restated without guessing: {restatement} {question} A reversible choice is simple: {reversible_default}.",
        f"In short: {restatement} {question} While waiting for the answer, {reversible_default}.",
    )
    answer = styles[_number(f"clarify-style:{code}:{variant}", 0, len(styles) - 1)]
    return TaskHand(
        data,
        goal,
        answer,
        ("restatement", "one_question", "reversible_default"),
        situation_title=situation_titles[row["domain"]],
        situation=situation_cards[row["domain"]],
    )


_RENDERERS = {
    "practical_action": _practical,
    "explanation_learning": _explanation,
    "troubleshooting": _troubleshooting,
    "writing_transformation": _writing,
    "planning_comparison": _planning,
    "conversation_empathy": _empathy,
    "safety_uncertainty": _safety,
    "grounded_qa": _grounded_qa,
    "summarization_synthesis": _summary,
    "extraction_classification": _extraction,
    "reasoning_verification": _reasoning,
    "critique_revision": _critique,
    "brainstorming_creativity": _brainstorm,
    "context_clarification": _clarification,
}


def deal_task_hand(row: dict[str, Any], variant: int) -> TaskHand:
    try:
        hand = _RENDERERS[row["family"]](row, variant)
    except KeyError as error:
        raise ValueError(f"no card renderer for {row['family']}") from error
    code = _code(row)
    if row["family"] == "extraction_classification":
        structured = json.loads(hand.answer)
        answer = json.dumps(
            structured,
            separators=(",", ":") if variant % 2 == 0 else (", ", ": "),
        )
    elif variant % 2 == 0:
        answer = f"Hand {code} — {hand.answer}"
    else:
        answer = f"For hand {code}: {hand.answer}"
    hand = TaskHand(
        hand.data,
        hand.goal,
        answer,
        hand.contract,
        situation_title=hand.situation_title,
        situation=hand.situation,
        rule=hand.rule,
    )
    validate_task_hand(row["family"], hand)
    return hand


def validate_task_hand(family: str, hand: TaskHand) -> None:
    if not hand.data.strip() or not hand.goal.strip() or not hand.answer.strip():
        raise ValueError(f"empty task card in {family}")
    checks = {
        "practical_action": lambda: all(
            x in hand.answer for x in ("Next step:", "Owner:", "Timing:")
        ),
        "explanation_learning": lambda: (
            all(x in hand.answer for x in ("Core idea:", "Example:", "Check:"))
            and "?" in hand.answer
        ),
        "troubleshooting": lambda: all(
            x in hand.answer for x in ("1.", "2.", "Direct check:", "Regression check:")
        ),
        "writing_transformation": lambda: (
            "Source text:" in hand.data and len(hand.answer.split()) >= 12
        ),
        "planning_comparison": lambda: all(
            x in hand.answer for x in ("Choose", "Sequence:", "Fallback trigger:")
        ),
        "conversation_empathy": lambda: hand.answer.count("?") <= 1,
        "safety_uncertainty": lambda: all(
            x in hand.answer for x in ("Immediate action:", "Boundary:", "Escalate")
        ),
        "grounded_qa": lambda: (
            "unknown" in hand.answer.lower() and "Source" in hand.data
        ),
        "summarization_synthesis": lambda: all(
            x in hand.answer for x in ("Decision:", "Action:", "Open point:")
        ),
        "extraction_classification": lambda: isinstance(json.loads(hand.answer), dict),
        "reasoning_verification": lambda: (
            all(x in hand.answer for x in ("Equation:", "Total:", "Check:"))
            and bool(re.search(r"\d", hand.answer))
        ),
        "critique_revision": lambda: all(
            x in hand.answer for x in ("Weakness:", "Revision:")
        ),
        "brainstorming_creativity": lambda: all(
            x in hand.answer for x in ("1.", "2.", "3.", "Select")
        ),
        "context_clarification": lambda: hand.answer.count("?") == 1,
    }
    if family not in checks or not checks[family]():
        raise ValueError(f"task hand does not fulfil the {family} contract")
