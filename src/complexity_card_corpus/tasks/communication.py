from __future__ import annotations

from typing import Any

from .core import (
    TaskHand,
    _card_pick,
    _code,
    _lower_sentence_initial,
    _number,
    _pick,
)


_LESSONS = {
    "computing": (
        "RAM holds working data temporarily; storage retains files after power is removed.",
        "Closing an application frees its RAM, but its saved file remains on storage.",
        "Why does a saved file survive a restart while an unsaved edit may not?",
    ),
    "software_resilience": (
        "A backup is a separate restorable copy that preserves a known-good state if an update corrupts or removes working files.",
        "Before updating an application, copy its data and configuration to another location and confirm that the copy can be restored.",
        "Why should restore capability be checked before the update begins?",
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
