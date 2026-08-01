from __future__ import annotations

from typing import Any

from .core import (
    TaskHand,
    _code,
    _compose_subcards,
    _deal_task_frames,
    _number,
)


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
            f"slot {each - 1} is occupied by A, immediately before B at slot {each}",
        )
    elif domain == "work_allocation":
        data = (
            f"Problem {code}: distribute 24 items equally among 3 people, "
            "then divide each person's share equally across two rounds."
        )
        equation = "24 / 3 = 8; 8 / 2 = 4"
        total, check = (
            "8 items per person and 4 items per person per round",
            "3 people × 2 rounds × 4 items = 24 items",
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
    data, goal = _deal_task_frames(
        row,
        variant,
        "reasoning",
        (
            data,
            f"Calculation card {code}: {data.split(': ', 1)[-1]}",
            f"Use the supplied values only — {data}",
        ),
        (
            "Calculate the result, show the equation, and verify it with an independent check.",
            "Give the equation and total, then confirm them through a second calculation.",
            "Solve the supplied problem and include one independent numerical check.",
        ),
    )
    answer = _compose_subcards(
        row,
        variant,
        "reasoning-answer",
        (
            (
                f"Equation: {equation}.",
                f"Equation: using the supplied values, {equation}.",
                f"Equation: the direct calculation is {equation}.",
                f"Equation: represent the required operation as {equation}.",
                f"Equation: evaluating the quantities gives {equation}.",
                f"Equation: the numerical relation is {equation}.",
            ),
            (
                f"Total: {total}.",
                f"Total: this gives {total}.",
                f"Total: the result is {total}.",
                f"Total: the computed value is {total}.",
                f"Total: therefore, {total}.",
                f"Total: the supplied values produce {total}.",
            ),
            (
                f"Check: {check}.",
                f"Check: independently, {check}.",
                f"Check: a second view confirms that {check}.",
                f"Check: verify the result by noting that {check}.",
            ),
        ),
        pool_names=("equation", "result", "verification"),
    )
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
        "status_update": (
            "Everything is on track, although integration is blocked and the delivery date is no longer known.",
            "the opening claim conflicts with the stated blocker and missing delivery date",
            "Core work is progressing, but integration remains blocked. Reassess the delivery date after that blocker is resolved.",
        ),
        "survey_report": (
            "Most users prefer the new layout because six of twelve participants selected it.",
            "six responses in a twelve-person sample do not establish a majority or a broader user preference",
            "Six of twelve participants selected the new layout. This sample does not establish a broader user preference.",
        ),
        "policy_notice": (
            "Access after 18:00 is prohibited unless approved, and exceptions may be available.",
            "the notice gives no approval authority or exception process",
            "Access after 18:00 requires prior approval. Name the approving authority and exception process before publishing the notice.",
        ),
        "data_caption": (
            "The results improved significantly after the change.",
            "the caption names no metric, comparator, magnitude, or uncertainty",
            "The figure compares results before and after the change. Add the metric, magnitude, comparator, and uncertainty before claiming an improvement.",
        ),
        "release_note": (
            "This update fixes all synchronization problems and works on every supported system.",
            "the universal reliability and compatibility claims exceed the stated evidence",
            "This update fixes the synchronization cases verified in the release tests. List the tested systems and retain any known limitations.",
        ),
        "support_macro": (
            "We resolved your issue. Please repeat the failed action to confirm that it now works.",
            "the reply claims resolution before the requested verification is complete",
            "We applied a possible fix for the reported issue. Please repeat the failed action so we can verify whether it is resolved.",
        ),
        "risk_assessment": (
            "The risk is low because no incident occurred last month.",
            "one incident-free month does not establish low likelihood or low impact",
            "No incident was recorded last month. Assess likelihood, impact, exposure, and mitigation evidence before assigning a risk level.",
        ),
    }
    draft, weakness, revision = cases[row["domain"]]
    draft = f"Draft {code}: {draft}"
    data, goal = _deal_task_frames(
        row,
        variant,
        "critique",
        (
            f"Text to review: {draft}",
            f"Review candidate {code}: {draft}",
            f"Editing input — {draft}",
        ),
        (
            "Identify the highest-impact weakness and provide a faithful two-sentence revision.",
            "Name the main evidence or clarity problem, then rewrite the text in exactly two sentences.",
            "Diagnose the most consequential flaw and revise it without inventing any fact.",
        ),
    )
    answer = _compose_subcards(
        row,
        variant,
        "critique-answer",
        (
            (
                f"Weakness: {weakness}.",
                f"Weakness: {weakness}; the wording exceeds the supplied evidence.",
                f"Weakness: {weakness}. The correction must remain within the recorded facts.",
                f"Weakness: {weakness}, making the original difficult to verify.",
            ),
            (
                f"Revision: {revision}",
                f"Faithful Revision: {revision}",
                f"Bounded Revision: {revision}",
            ),
        ),
        pool_names=("weakness", "revision"),
    )
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
    constraint_checks = {
        "Make the options meaningfully different rather than cosmetic rewrites.": (
            "The options differ in mechanism rather than wording alone."
        ),
        "Avoid ideas that create unnecessary safety, privacy, or exclusion risks.": (
            "None of the options requires sensitive personal data or an avoidable safety risk."
        ),
        "Explain briefly how each retained option meets the named criteria.": (
            "Each description states how its option fits the brief."
        ),
        "Keep every option feasible within the stated resources.": (
            "All three options stay within the resources named in the brief."
        ),
        "Keep the intended audience visible in each option.": (
            "Each option remains directed to the audience named in the brief."
        ),
        "Keep the proposal small enough to test and revise.": (
            "The selected option can be tested at the stated scale before expansion."
        ),
    }
    outcome_checks = {
        "The remaining ideas are feasible within the available resources.": (
            "The three retained ideas remain feasible under the stated limits."
        ),
        "The main trade-off of each leading option is visible.": (
            "The alternatives emphasize different strengths, making the choice explicit."
        ),
        "Each retained option satisfies the stated criteria.": (
            "Each retained option satisfies the stated criteria."
        ),
        "One idea is developed into a small testable proposal.": (
            "The selected idea is the smallest concrete proposal to test first."
        ),
        "Compatible strengths are combined without preserving their conflicts.": (
            "The selection keeps the most compatible strengths without combining conflicting requirements."
        ),
        "The candidate options differ in a meaningful and useful way.": (
            "The candidate options differ in a way that changes how the brief would be carried out."
        ),
    }
    constraint_check = constraint_checks.get(
        row.get("constraint", ""),
        "The options remain bounded by the explicit brief.",
    )
    outcome_check = outcome_checks.get(
        row.get("desired_outcome", ""),
        "The selected option is concrete enough to test first.",
    )
    options, selection = answer.rsplit(" Select ", 1)
    answer = _compose_subcards(
        row,
        variant,
        "brainstorm-answer",
        (
            (
                options,
                f"Options: {options}",
                f"Candidate set: {options}",
            ),
            (
                f"Criteria review: {constraint_check}",
                f"Constraint review: {constraint_check}",
                f"Fit with the brief: {constraint_check}",
            ),
            (
                f"Outcome review: {outcome_check}",
                f"Comparison result: {outcome_check}",
                f"Practical result: {outcome_check}",
            ),
            (
                f"Select {selection}",
                f"Select this option: {selection}",
                f"Select the strongest fit: {selection}",
            ),
        ),
        pool_names=("options", "criteria", "outcome", "selection"),
    )
    data = _compose_subcards(
        row,
        variant,
        "brainstorm-input",
        (
            (f"Brief {code}:", f"Idea brief {code} —", f"Creative constraint card {code}:"),
            (f"{brief}.",),
        ),
        pool_names=("brief_label", "brief"),
    )
    goal = _compose_subcards(
        row,
        variant,
        "brainstorm-objective",
        (
            (
                "Generate three meaningfully different options.",
                "Propose three distinct approaches.",
                "Create three feasible alternatives.",
            ),
            (
                "Test each against the brief.",
                "Compare their fit with the stated limits.",
                "Check each option against the named criteria.",
            ),
            (
                "Select the strongest one.",
                "Recommend one option and explain the choice.",
                "Choose the best bounded proposal to test first.",
            ),
        ),
        pool_names=("generation", "criteria", "selection"),
    )
    return TaskHand(data, goal, answer, ("three_options", "criteria", "selection"))
