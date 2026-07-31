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


def _number(key: str, low: int, high: int) -> int:
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return low + value % (high - low + 1)


def _pick(key: str, values: tuple[str, ...]) -> str:
    return values[_number(key, 0, len(values) - 1)]


def _code(row: dict[str, Any]) -> str:
    return row["scenario_id"].split(":")[-1][:6].upper()


def _payload(row: dict[str, Any]) -> dict[str, str]:
    return json.loads(row["semantic_payload"])


def _practical(row: dict[str, Any], variant: int) -> TaskHand:
    payload = _payload(row)
    code = _code(row)
    day = _number(f"day:{code}", 8, 24)
    hour = _number(f"hour:{code}", 9, 16)
    cost = _number(f"cost:{code}", 18, 95)
    data = (
        f"Case {code}; subject: {payload['subject']}; current record: reference "
        f"{code}-A lists day {day}, while reference {code}-B lists day {day + 1}; "
        f"the available reversible option is {hour}:00 at a quoted cost of ${cost}."
    )
    goal = (
        f"Resolve the record conflict and give one next step, its owner, its timing, "
        f"and the confirmation to obtain before commitment."
    )
    answer = (
        f"Next step: ask the provider to reconcile references {code}-A and {code}-B "
        f"before booking the {hour}:00 option. Owner: the requester contacts the "
        f"provider; the provider confirms the correct day and final ${cost} price. "
        f"Timing: do this before day {day}. Commit only after the corrected record "
        f"is returned in writing; otherwise keep the current arrangement unchanged."
    )
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
    data = f"Concept notes: {mechanism} Example available: {example}"
    goal = "Explain the mechanism in plain language, apply the example, and end with one check question."
    answer = (
        f"Core idea: {mechanism} Example: {example} The example fits because it "
        f"shows the distinction in an observable case. Check: {check}"
    )
    return TaskHand(data, goal, answer, ("mechanism", "example", "question"))


_ERRORS = {
    "software_install": ("macOS 15", "installer exits with code 73", "the install directory was changed"),
    "network_connection": ("a laptop on Wi-Fi", "requests time out after DNS lookup", "a custom DNS server was enabled"),
    "file_sync": ("a desktop sync client", "local changes remain queued", "the remote folder was renamed"),
    "peripheral": ("a USB keyboard", "the device powers on but sends no input", "it was moved through a hub"),
    "web_form": ("a current browser", "submission returns HTTP 422", "a required profile field was removed"),
    "data_pipeline": ("a nightly ETL job", "the transform stage reports a schema mismatch", "a source column changed type"),
}


def _troubleshooting(row: dict[str, Any], variant: int) -> TaskHand:
    env, error, change = _ERRORS[row["domain"]]
    code = _code(row)
    data = (
        f"Environment: {env}. Observed error: {error}. Last change: {change}. "
        f"Control run {code} succeeded before that change; user data is backed up read-only."
    )
    goal = "Give a reversible diagnostic sequence, a direct fix check, and a regression check."
    answer = (
        f"1. Preserve log {code} and reproduce once without changing data. 2. Restore only "
        f"the changed setting: {change}. 3. Repeat the failing operation and compare the new "
        f"log with {code}. Direct check: confirm that '{error}' no longer appears. Regression "
        f"check: repeat the last known-good operation. If either check fails, restore the setting "
        f"and stop with both logs intact."
    )
    return TaskHand(data, goal, answer, ("steps", "direct_check", "regression_check"))


def _writing(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    owner = _pick(f"owner:{code}", ("Maya", "Jon", "Ari", "Lea", "Noah", "Iris"))
    day = _number(f"write-day:{code}", 10, 28)
    variants = {
        "email": (
            f"notes {code}: send team; review complete; two figures need captions; {owner} owns them; target day {day}; release waits",
            f"Subject: Review {code} next steps\n\nThe review is complete. Two figures still need captions, which {owner} owns for day {day}. We will decide the release timing after the captions are reviewed.",
        ),
        "project_update": (
            f"update {code}: review complete; captions missing on two figures; owner {owner}; target day {day}; release decision blocked",
            f"Project update {code}: Review is complete. Remaining work: {owner} adds captions to two figures by day {day}. Blocker: release timing stays open until caption review.",
        ),
        "support_reply": (
            f"case {code}: issue reviewed; two screenshots need labels; {owner} will add them by day {day}; resolution waits for review",
            f"Support reply {code}: We have completed the issue review. {owner} will label the two remaining screenshots by day {day}. We will confirm resolution after that review.",
        ),
        "meeting_notes": (
            f"meeting {code}: review complete; two captions outstanding; {owner}; day {day}; no release decision yet",
            f"Meeting {code} — Decision: review complete. Action: {owner} adds two captions by day {day}. Open item: decide release timing after caption review.",
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
            "a refundable rail itinerary", "a non-refundable flight", "a bus route without step-free access"
        ),
        "learning_plan": (
            "a guided practice course", "an intensive workshop", "a self-study bundle missing the required exercises"
        ),
        "small_project": (
            "a scoped three-day implementation", "an accelerated external build", "a five-day build missing the acceptance test"
        ),
        "purchase_decision": (
            "a supported standard model", "a faster premium model", "a discounted model missing the required connector"
        ),
    }
    option_a, option_b, option_c = option_sets[row["domain"]]
    data = (
        f"Option A: {option_a}; cost ${a}; duration 3 days; every required condition met. "
        f"Option B: {option_b}; cost ${b}; duration 2 days; every required condition met. "
        f"Option C: {option_c}; cost ${budget - 35}; duration 5 days; one "
        f"non-negotiable requirement. Maximum budget: ${budget}; deadline: 4 days."
    )
    goal = "Apply the hard constraints, choose an option, order the next steps, and name a fallback trigger."
    answer = (
        f"Reject B because ${b} exceeds the ${budget} cap, and reject C because it misses a "
        f"non-negotiable requirement and exceeds the deadline. Choose A: {option_a}. Sequence: confirm "
        f"availability today, reserve or schedule it reversibly, then verify the requirement before payment. "
        f"Fallback trigger: if A cannot be confirmed by tomorrow, pause and reopen the shortlist "
        f"rather than selecting B or C."
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
    question = (
        "Would it help more to talk through what happened, or to choose one small next step?"
        if row["intent"] in {"clarify", "gentle_step", "conversation"}
        else "What part feels most important to have understood right now?"
    )
    answer = (
        f"That sounds like a lot to carry, especially when two feelings can be true at once. "
        f"You do not need to solve the whole situation immediately. {question}"
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
    answer = (
        f"Immediate action: {actions[row['domain']]} Boundary: I cannot verify the claim "
        f"or safely direct the consequential action from this information alone. Escalate through "
        f"the named official channel and keep sensitive information private."
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
    answer = f"Based on Source {code}, {supported}"
    return TaskHand(data, goal, answer, ("direct_answer", "evidence", "unknown"))


def _summary(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    owner = _pick(f"summary-owner:{code}", ("Mina", "Paul", "Sora", "Theo", "Lina"))
    day = _number(f"summary-day:{code}", 12, 27)
    cases = {
        "meeting_transcript": ("approve the interface copy", "run two accessibility checks", "release timing"),
        "research_notes": ("retain the observed temperature result", "replicate two uncertain measurements", "the causal explanation"),
        "support_thread": ("keep the case open", "test two account-recovery paths", "whether the issue is device-specific"),
        "project_update": ("accept the completed prototype", "finish two integration checks", "the launch date"),
        "policy_memo": ("adopt the revised access rule", "document two listed exceptions", "the enforcement start date"),
        "article_excerpt": ("retain the article's central claim", "verify two cited examples", "whether the pattern generalizes"),
        "incident_log": ("keep the service in monitored recovery", "inspect two remaining error sources", "the incident's root cause"),
        "learning_notes": ("retain the working definition", "test it on two new examples", "where the rule stops applying"),
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
        "receipt": (f"merchant=North Market; date=2026-08-{day:02d}; total=${amount}.50; tax=$4.00; cashier missing", {"merchant": "North Market", "date": f"2026-08-{day:02d}", "total": f"{amount}.50 USD", "tax": "4.00 USD", "cashier": None}),
        "event_listing": (f"title=Open Lab; date=2026-08-{day:02d}; venue=Room {amount}; starts=18:30; eligibility missing", {"title": "Open Lab", "date": f"2026-08-{day:02d}", "venue": f"Room {amount}", "starts": "18:30", "eligibility": None}),
        "contact_record": (f"name=Sam Iri; role=Editor; organization=North Review; email=sam.{code.lower()}@example.org; phone missing", {"name": "Sam Iri", "role": "Editor", "organization": "North Review", "email": f"sam.{code.lower()}@example.org", "phone": None}),
        "issue_ticket": (f"ticket={code}; environment=Linux; severity=medium; status=pending; owner missing", {"ticket": code, "environment": "Linux", "severity": "medium", "status": "pending", "owner": None}),
        "survey_response": (f"response={code}; rating=4; topic=navigation; comment=clear after retry; follow_up missing", {"response": code, "rating": 4, "topic": "navigation", "comment": "clear after retry", "follow_up": None}),
        "inventory_record": (f"item={code}; quantity={amount}; location=A-{day}; condition=good; checked_by missing", {"item": code, "quantity": amount, "location": f"A-{day}", "condition": "good", "checked_by": None}),
        "schedule_entry": (f"event=Review {code}; date=2026-08-{day:02d}; starts=09:30; duration=45 minutes; room missing", {"event": f"Review {code}", "date": f"2026-08-{day:02d}", "starts": "09:30", "duration_minutes": 45, "room": None}),
        "case_note": (f"case={code}; observed=package sealed; reported=item incomplete; action=photographs retained; next_owner missing", {"case": code, "observed": "package sealed", "reported": "item incomplete", "action": "photographs retained", "next_owner": None}),
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
        total, check = f"${result}", f"the item subtotal is ${units * each}, and adding ${extra} gives ${result}"
    elif domain == "schedule_math":
        result = units * each + extra
        data = f"Problem {code}: {units} sessions last {each} minutes each, followed by a {extra}-minute break."
        equation = f"{units} × {each} + {extra} = {result}"
        total, check = f"{result} minutes", f"removing the {extra}-minute break leaves {units * each} session minutes"
    elif domain == "unit_conversion":
        result = units * 100
        data = f"Problem {code}: convert {units} metres to centimetres using 1 metre = 100 centimetres."
        equation = f"{units} × 100 = {result}"
        total, check = f"{result} centimetres", f"dividing {result} by 100 returns {units} metres"
    elif domain == "proportions":
        result = units * each
        data = f"Problem {code}: one batch uses {each} cups; keep the ratio for {units} batches."
        equation = f"{units} × {each} = {result}"
        total, check = f"{result} cups", f"{result} divided by {units} returns {each} cups per batch"
    elif domain == "table_comparison":
        result = max(units * each, units * extra)
        data = f"Problem {code}: table A reports {units} × {each}; table B reports {units} × {extra}. Compare the totals."
        equation = f"max({units} × {each}, {units} × {extra}) = {result}"
        total, check = f"{result}", "computing both products independently confirms the larger entry"
    elif domain == "sequence_pattern":
        result = units + 3 * each
        data = f"Problem {code}: the sequence is {units}, {units + each}, {units + 2 * each}, __; use the constant difference."
        equation = f"{units} + 3 × {each} = {result}"
        total, check = f"{result}", f"each adjacent pair differs by {each}"
    elif domain == "logical_constraints":
        result = each - 1 + units
        data = f"Problem {code}: A must occur immediately before B; B is at slot {each}; C is at slot {units}. Find A's slot and add it to C's slot."
        equation = f"({each} - 1) + {units} = {result}"
        total, check = f"{result}", f"A occupies slot {each - 1}, immediately before B at slot {each}"
    else:
        result = units
        total_outcomes = units + each
        data = f"Problem {code}: a bag has {units} blue and {each} amber tokens; one token is drawn uniformly."
        equation = f"{units} / ({units} + {each}) = {units}/{total_outcomes}"
        total, check = f"{units}/{total_outcomes} probability of blue", f"the favorable and total counts are {units} and {total_outcomes}"
    goal = "Calculate the result, show the equation, and verify it with an independent check."
    answer = f"Equation: {equation}. Total: {total}. Check: {check}."
    return TaskHand(data, goal, answer, ("equation", "result", "check"))


def _critique(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    cases = {
        "email_draft": ("Send the files soon because everyone should know what I mean.", "the request has no recipient, deadline, or named files", "Please send the two review files to the project team by 16:00 and confirm delivery."),
        "argument": ("Our trial proves the workflow is always faster because three of five testers finished sooner.", "a universal claim is drawn from three results while two outcomes are missing", "Three of five testers finished sooner; two outcomes were unrecorded, so no general speed advantage is established."),
        "project_plan": ("Build the prototype, test it, and launch next week.", "the plan gives no owner, dependency, or completion criterion", "Mara builds the prototype by Tuesday; testing begins after acceptance checks pass, and launch remains conditional on those results."),
        "explanation": ("Encryption makes data safe by turning it into random text.", "the explanation omits keys and overstates safety", "Encryption transforms readable data using a key; authorized holders can reverse it, while security still depends on key protection and implementation."),
        "instructions": ("Install the update, delete the old folder, and check whether it works.", "the destructive deletion comes before verification or backup", "Back up the old folder, install the update separately, verify the application, and delete nothing until rollback is no longer needed."),
        "summary": ("The meeting discussed many topics and everyone agreed the project was important.", "the summary omits the actual decision and action", "The team approved the prototype review. Nia will complete the two remaining checks before a launch date is chosen."),
        "claim_evidence": ("Users prefer the redesign; two positive comments prove it.", "two comments cannot support a general preference claim", "Two respondents commented positively on the redesign; broader user preference remains unmeasured."),
        "interface_copy": ("Error. Something went wrong. Try again.", "the message gives neither the failed action nor a useful next step", "Upload failed because the connection ended. Check the network, then select Retry; the local file remains unchanged."),
    }
    draft, weakness, revision = cases[row["domain"]]
    draft = f"Draft {code}: {draft}"
    data = f"Text to review: {draft}"
    goal = "Identify the highest-impact weakness and provide a faithful two-sentence revision."
    answer = f"Weakness: {weakness}. Revision: {revision}"
    return TaskHand(data, goal, answer, ("weakness", "reason", "revision"))


def _brainstorm(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    cases = {
        "names": ("name a neighborhood tool library for adult residents; names must be short and welcoming", "1. Tool Harbor — suggests shared access. 2. Common Kit — emphasizes practical community use. 3. Borrow Bench — makes the action memorable. All are short and audience-appropriate. Select Common Kit for its clearest meaning."),
        "lesson_activity": ("teach cause and effect to learners in 20 minutes using paper only", "1. Cause Chain — order event cards. 2. Change One Thing — predict an outcome after one variable changes. 3. Evidence Match — connect claims to observations. All fit the material and time limits. Select Change One Thing for its direct observable check."),
        "event_plan": ("design a two-hour neighborhood event for 30 people with a $60 budget and step-free access", "1. Skill Tables — rotating demonstrations. 2. Story Map — residents place anonymous local memories. 3. Repair Circle — shared guidance for small fixes. All meet the stated constraints. Select Skill Tables for flexible participation and simple access."),
        "feature_ideas": ("reduce missed handoffs in a small team without removing approval checks", "1. Owner Badge — show the current responsible person. 2. Ready Queue — list items that passed approval. 3. Handoff Receipt — record sender, receiver, and time. All preserve review controls. Select Handoff Receipt because it makes every transfer auditable."),
        "writing_prompts": ("create short speculative-fiction prompts about memory for adult beginners", "1. A town forgets one street each sunrise. 2. A diver finds a memory labeled with tomorrow's date. 3. Two siblings remember the same childhood differently. All share the theme with distinct perspectives. Select the diver prompt for its immediate mystery."),
        "low_cost_activity": ("create a 30-minute indoor activity for eight people using common paper supplies", "1. Paper Bridge — build for a fixed span. 2. Sequence Swap — reorder illustrated events. 3. Constraint Sketch — draw under one changing rule. Each avoids specialist materials and hidden cost. Select Paper Bridge for a clear shared test."),
        "outreach": ("invite local students to a free weekend science session without collecting personal data", "1. Library Poster — direct readers to open attendance hours. 2. School Bulletin — share a short teacher-ready notice. 3. Community Demo — offer a public five-minute preview. All align channel and audience without data collection. Select School Bulletin for trusted distribution."),
        "workflow": ("reduce review delays while retaining the final human approval", "1. Intake Checklist — reject incomplete submissions early. 2. Parallel Evidence Check — review independent facts together. 3. Approval Queue — surface only complete items. All reduce friction without bypassing approval. Select Intake Checklist because it prevents avoidable rework first."),
    }
    brief, answer = cases[row["domain"]]
    data = f"Brief {code}: {brief}."
    goal = "Generate three meaningfully different options, test them against the brief, and select one."
    return TaskHand(data, goal, answer, ("three_options", "criteria", "selection"))


def _clarification(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    cases = {
        "ambiguous_request": ("Please move the review to Friday.", "Do you mean reschedule the review meeting or change the document deadline to Friday?"),
        "missing_reference": ("Summarize the attached report.", "Could you attach the report that should be summarized?"),
        "conflicting_instruction": ("Keep every detail, but make the answer no longer than one sentence.", "Which takes priority: preserving every detail or limiting the answer to one sentence?"),
        "unclear_pronoun": ("Send it to them after the review.", "What does 'it' refer to, and who should receive it?"),
        "incomplete_goal": ("Help me organize the project.", "What outcome should the organization produce: a schedule, a task list, or a file structure?"),
        "scope_boundary": ("Update the examples and anything else that needs work.", "Should I change only the examples, or also revise the surrounding explanation and tests?"),
        "format_preference": ("Give me the comparison results.", "Would you like a short table, a prose summary, or both?"),
        "timeline_ambiguity": ("Finish this soon after the next review.", "What calendar date or time limit should 'soon after' mean?"),
    }
    ambiguous, question = cases[row["domain"]]
    data = (
        f'Request {code}: "{ambiguous}" Two interpretations remain possible, and choosing one '
        f"would change the action; no further reference is supplied."
    )
    goal = "Restate what is understood, ask one decisive question, and give only a reversible provisional interpretation."
    styles = (
        f"Understood: request {code} asks for a change, but its target is ambiguous. {question} Until confirmed, keep the present schedule and files unchanged.",
        f"My current reading of {code}: a change is wanted, although one reference is unresolved. {question} For now, make no irreversible edit.",
        f"What is clear in {code}: the user wants an adjustment. What is not clear: which object or deadline controls it. {question} Pending that answer, preserve the current version.",
        f"Request {code} establishes an intended change but not enough scope to execute it. {question} The reversible default is to retain existing settings.",
        f"I understand {code} as a request to update something after a stated condition. The exact referent remains open. {question} No send or schedule change should occur yet.",
        f"The supported interpretation of {code} is limited to preparing for a possible change. {question} Until the reference is resolved, leave the operative record as it is.",
        f"Request {code} contains a clear desire to proceed and an unclear object. {question} A safe provisional choice is to hold the current arrangement.",
        f"I can restate {code} only as an unconfirmed modification request. {question} While waiting, preserve both the current file and timing.",
    )
    answer = styles[_number(f"clarify-style:{code}:{variant}", 0, len(styles) - 1)]
    return TaskHand(data, goal, answer, ("restatement", "one_question", "reversible_default"))


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
        structured = {"hand": code, **structured}
        answer = json.dumps(
            structured,
            separators=(",", ":") if variant % 2 == 0 else (", ", ": "),
        )
    elif variant % 2 == 0:
        answer = f"Hand {code} — {hand.answer}"
    else:
        answer = f"For hand {code}: {hand.answer}"
    hand = TaskHand(hand.data, hand.goal, answer, hand.contract)
    validate_task_hand(row["family"], hand)
    return hand


def validate_task_hand(family: str, hand: TaskHand) -> None:
    if not hand.data.strip() or not hand.goal.strip() or not hand.answer.strip():
        raise ValueError(f"empty task card in {family}")
    checks = {
        "practical_action": lambda: all(x in hand.answer for x in ("Next step:", "Owner:", "Timing:")),
        "explanation_learning": lambda: all(x in hand.answer for x in ("Core idea:", "Example:", "Check:")) and "?" in hand.answer,
        "troubleshooting": lambda: all(x in hand.answer for x in ("1.", "2.", "Direct check:", "Regression check:")),
        "writing_transformation": lambda: "Source text:" in hand.data and len(hand.answer.split()) >= 12,
        "planning_comparison": lambda: all(x in hand.answer for x in ("Choose", "Sequence:", "Fallback trigger:")),
        "conversation_empathy": lambda: hand.answer.count("?") <= 1,
        "safety_uncertainty": lambda: all(x in hand.answer for x in ("Immediate action:", "Boundary:", "Escalate")),
        "grounded_qa": lambda: "unknown" in hand.answer.lower() and "Source" in hand.data,
        "summarization_synthesis": lambda: all(x in hand.answer for x in ("Decision:", "Action:", "Open point:")),
        "extraction_classification": lambda: isinstance(json.loads(hand.answer), dict),
        "reasoning_verification": lambda: all(x in hand.answer for x in ("Equation:", "Total:", "Check:")) and bool(re.search(r"\d", hand.answer)),
        "critique_revision": lambda: all(x in hand.answer for x in ("Weakness:", "Revision:")),
        "brainstorming_creativity": lambda: all(x in hand.answer for x in ("1.", "2.", "3.", "Select")),
        "context_clarification": lambda: hand.answer.count("?") == 1,
    }
    if family not in checks or not checks[family]():
        raise ValueError(f"task hand does not fulfil the {family} contract")
