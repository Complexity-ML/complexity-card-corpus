from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClarificationFacts:
    weekday: str
    report_id: int
    word_limit: int
    lead_event: str
    project_name: str
    doc_count: int
    option_count: int
    dependency: str
    update_topic: str
    dashboard_name: str
    destination: str
    product: str


def clarification_question_cards(
    domain: str,
    facts: ClarificationFacts,
) -> tuple[str, ...]:
    """Return grammatically complete questions with the same information need."""

    f = facts
    cards = {
        "ambiguous_request": (
            f"Do you mean reschedule the review meeting or change the document deadline to {f.weekday}?",
            f"Which item should move to {f.weekday}: the review meeting or the document due date?",
            f"Is {f.weekday} the new meeting date, or is it the revised deadline for the document?",
            f"Should the calendar review or the document deadline be changed to {f.weekday}?",
        ),
        "missing_reference": (
            f"Could you attach report {f.report_id} so it can be summarized?",
            f"Where can I access report {f.report_id}, the source that needs summarizing?",
            f"Please provide the contents of report {f.report_id}; is the file available to attach?",
            f"Can you share report {f.report_id} before I prepare its summary?",
        ),
        "conflicting_instruction": (
            f"Which takes priority: preserving every detail or limiting the answer to {f.word_limit} words?",
            f"Should completeness override the {f.word_limit}-word cap, or should the cap control?",
            f"If all details cannot fit within {f.word_limit} words, which requirement may be relaxed?",
            f"Do you want an exhaustive answer or a strict maximum of {f.word_limit} words?",
        ),
        "unclear_pronoun": (
            f"What does 'it' refer to, and who should receive it after {f.lead_event}?",
            f"Which item should be sent after {f.lead_event}, and to which recipient?",
            f"Can you identify both the thing to send and the people meant by 'them' after {f.lead_event}?",
            f"After {f.lead_event}, what exactly is being sent and who is the destination?",
        ),
        "incomplete_goal": (
            f"What outcome should organizing {f.project_name} produce: a schedule, a task list, or a file structure?",
            f"For {f.project_name}, is the required deliverable a timeline, an action list, or an organized folder layout?",
            f"Which concrete result would count as organized for {f.project_name}?",
            f"Should I structure {f.project_name} as dates, tasks, or files?",
        ),
        "scope_boundary": (
            f"Should I change only the {f.doc_count} examples, or also revise the surrounding explanation and tests?",
            f"Is the authorized scope limited to {f.doc_count} examples, or does it include their explanation and tests?",
            f"Beyond those {f.doc_count} examples, may I update the related prose and test coverage?",
            f"Do the explanation and tests belong in scope with the {f.doc_count} example edits?",
        ),
        "format_preference": (
            f"Would you like a short table, a prose summary, or both for the {f.option_count} options?",
            f"How should the {f.option_count} results be presented: in prose, as a compact table, or in both forms?",
            f"Should I format the comparison of {f.option_count} options as a table or a written summary?",
            f"For these {f.option_count} alternatives, do you prefer tabular results, narrative results, or both?",
        ),
        "timeline_ambiguity": (
            f"What calendar date or time limit should 'soon after {f.dependency}' mean?",
            f"How much time after {f.dependency} is available before completion is due?",
            f"Which exact deadline follows {f.dependency}?",
            f"Can you translate 'soon' into a date or interval measured from {f.dependency}?",
        ),
        "team_request": (
            f"Which team group should receive the {f.update_topic} update, and when is the review scheduled?",
            f"Who is the intended team audience for the {f.update_topic}, and what is the review time?",
            f"Before sending the {f.update_topic} note, can you identify the group and the review date?",
            f"Where should the {f.update_topic} update go, and by what review deadline?",
        ),
        "data_request": (
            f"Which dataset and exact date range should the {f.dashboard_name} export contain?",
            f"What source table and start-to-end dates define the requested {f.dashboard_name} records?",
            f"Can you identify both the dataset and the reporting interval for the {f.dashboard_name} export?",
            f"From the {f.dashboard_name}, which records count as recent and which dataset should supply them?",
        ),
        "travel_request": (
            f"What are the travel dates and departure city for the {f.destination} trip?",
            f"When should the journey to {f.destination} occur, and where does it begin?",
            f"Which origin city and outbound and return dates apply to the {f.destination} workshop trip?",
            f"Can you provide the departure point and travel window for {f.destination}?",
        ),
        "purchasing_request": (
            f"How many {f.product} are needed, and what is the maximum budget per unit?",
            f"What quantity of {f.product} should be ordered, at what per-item spending cap?",
            f"Can you set both the unit count and maximum unit price for the {f.product} purchase?",
            f"How large is the {f.product} order, and how much may each item cost?",
        ),
    }
    return cards[domain]


def clarification_restatement_cards(
    *,
    requester: str,
    stakeholder_group: str,
    work_context: str,
    restatement: str,
) -> tuple[str, ...]:
    """Place one known-state clause in varied natural conversation frames."""

    lower = restatement[:1].lower() + restatement[1:]
    return (
        f"Context for {requester} and {stakeholder_group} {work_context}: {lower}",
        f"{requester}'s request for {stakeholder_group} {work_context} establishes that {lower}",
        f"For {stakeholder_group} {work_context}, the known part of {requester}'s request is this: {lower}",
        f"What is already clear for {requester} {work_context} is that {lower}",
        f"Within {stakeholder_group}, {requester}'s current request can be bounded as follows: {lower}",
        f"The available context from {requester} {work_context} confirms that {lower}",
        f"As {stakeholder_group} works {work_context}, one point is established: {lower}",
        f"The stated request from {requester} gives {stakeholder_group} this much certainty: {lower}",
        f"For this {stakeholder_group} task {work_context}, the supplied information shows that {lower}",
        f"{requester} has already specified one part of the work for {stakeholder_group}: {lower}",
        f"The settled portion of the request involving {stakeholder_group} is that {lower}",
        f"Before {stakeholder_group} proceeds {work_context}, the record supports this reading: {lower}",
        f"{requester} and {stakeholder_group} share this established starting point {work_context}: {lower}",
        f"One part of the task is not in dispute for {stakeholder_group}: {lower}",
        f"The information supplied by {requester} settles the following point: {lower}",
        f"Here is the confirmed portion of {requester}'s request {work_context}: {lower}",
        f"For the current work with {stakeholder_group}, the record is definite on one point: {lower}",
        f"{stakeholder_group} can rely on this part of the request from {requester}: {lower}",
        f"Before anything else proceeds, one firm limit applies — {lower}",
        f"As a starting point {work_context}, {requester} has made this clear: {lower}",
        f"The known state for {stakeholder_group} can be stated directly: {lower}",
        f"Only one portion is settled so far in {requester}'s request: {lower}",
        f"The current record gives {stakeholder_group} this confirmed information: {lower}",
        f"Before deciding what follows, this much is supported: {lower}",
        f"The source establishes a limited fact for {requester}: {lower}",
        f"This is the portion {stakeholder_group} can treat as confirmed: {lower}",
        f"One boundary is already explicit in the request: {lower}",
        f"At present, the supported interpretation goes only this far: {lower}",
        f"The shared record fixes one detail before any choice is made: {lower}",
        f"This much of the requested work is grounded in the supplied context: {lower}",
        f"There is a clear starting condition for the task: {lower}",
        f"The non-ambiguous part can be stated without inference: {lower}",
    )


def clarification_restatement_meaning_cards(
    domain: str,
    facts: ClarificationFacts,
) -> tuple[str, ...]:
    """Express the known portion without reusing one domain sentence."""

    f = facts
    return {
        "ambiguous_request": (
            f"The requested move to {f.weekday} is clear, but the meeting and document deadline are both possible targets.",
            f"A date change to {f.weekday} is wanted; the request does not say whether it applies to the review or its paperwork.",
            f"{f.weekday} is the new requested date, while the item being rescheduled remains unidentified.",
        ),
        "missing_reference": (
            f"A summary is wanted from report {f.report_id}, whose contents have not been supplied.",
            f"Report {f.report_id} is named as the source, but no file or text is available to summarize.",
            f"The requested output is a summary of report {f.report_id}; the underlying report is absent.",
        ),
        "conflicting_instruction": (
            f"Retaining all information while staying under {f.word_limit} words creates two requirements that may conflict.",
            f"Completeness and a strict {f.word_limit}-word ceiling are both requested without a stated priority.",
            f"The source asks for every detail and no more than {f.word_limit} words, but does not resolve the trade-off.",
        ),
        "unclear_pronoun": (
            f"Something should be sent after {f.lead_event}, though neither the item nor its recipients are named.",
            f"The timing after {f.lead_event} is known, but 'it' and 'them' lack clear referents.",
            f"A later send is requested following {f.lead_event}; its content and destination remain ambiguous.",
        ),
        "incomplete_goal": (
            f"Help is requested for organizing {f.project_name}, without defining the artifact to produce.",
            f"{f.project_name} needs organization, but the request leaves its desired output open.",
            f"The task concerns structuring {f.project_name}; no schedule, task list, or file layout has been selected.",
        ),
        "scope_boundary": (
            f"Edits to {f.doc_count} examples are requested, while authorization for related explanation and test changes is unclear.",
            f"The {f.doc_count} examples are definitely included, but the surrounding material has no confirmed status.",
            f"A revision must cover {f.doc_count} examples; whether that scope extends beyond them is unresolved.",
        ),
        "format_preference": (
            f"Results exist for {f.option_count} options, but no output format has been chosen.",
            f"The request covers a {f.option_count}-option comparison without saying how readers should receive it.",
            f"All {f.option_count} comparison results are needed; table and narrative presentation remain open alternatives.",
        ),
        "timeline_ambiguity": (
            f"Work should finish after {f.dependency}, but 'soon' supplies no measurable deadline.",
            f"{f.dependency.capitalize()} controls when completion can begin, while the allowed interval afterward is undefined.",
            f"The sequence relative to {f.dependency} is known; the actual delivery date is not.",
        ),
        "team_request": (
            f"The {f.update_topic} update must precede a review, though audience and review time are missing.",
            f"A team communication about the {f.update_topic} is requested without a named group or deadline.",
            f"The message topic is the {f.update_topic}; who receives it and when it is due remain unknown.",
        ),
        "data_request": (
            f"Records are wanted from the {f.dashboard_name}, but 'recent' does not identify a dataset or date interval.",
            f"The {f.dashboard_name} is the source area; the exact table and reporting period are unspecified.",
            f"An export is requested from the {f.dashboard_name} without boundaries for source data or time.",
        ),
        "travel_request": (
            f"The workshop destination is {f.destination}, while origin and travel dates are absent.",
            f"A trip around the {f.destination} workshop is needed, but its departure point and travel window are open.",
            f"{f.destination} and the workshop anchor the plan; the journey's starting city and dates do not.",
        ),
        "purchasing_request": (
            f"The design team needs {f.product}, with no stated quantity or unit-price ceiling.",
            f"A purchase of {f.product} is requested, but order size and budget remain unset.",
            f"The item category and recipient team are known; the number of {f.product} and permitted cost are not.",
        ),
    }[domain]


def clarification_default_cards(
    domain: str,
    facts: ClarificationFacts,
) -> tuple[str, ...]:
    """Return reversible defaults that preserve the same unresolved boundary."""

    f = facts
    return {
        "ambiguous_request": (
            f"leave both possible {f.weekday} changes untouched",
            "keep the meeting and document dates as they are",
            "make no calendar or deadline edit before the target is named",
        ),
        "missing_reference": (
            "do not draft a substitute summary",
            f"produce no summary until report {f.report_id} is available",
            "leave the requested summary empty rather than inventing source content",
        ),
        "conflicting_instruction": (
            "preserve the original text without rewriting it",
            "leave the source unchanged until the controlling limit is chosen",
            "make no edit while completeness and length still conflict",
            "retain the current wording pending a priority decision",
            "hold the document in its present form until one requirement controls",
            "avoid shortening or expanding the text before the conflict is resolved",
        ),
        "unclear_pronoun": (
            "send nothing until item and recipient are known",
            "leave the material unsent while both references remain unresolved",
            "make no delivery based on the ambiguous pronouns",
        ),
        "incomplete_goal": (
            f"make no structural change to {f.project_name}",
            "avoid choosing a deliverable on the requester's behalf",
            "leave the project organization unchanged pending an outcome choice",
        ),
        "scope_boundary": (
            f"prepare no edits beyond the {f.doc_count} examples",
            "hold related explanations and tests outside the working scope",
            f"limit any proposed revision to the named {f.doc_count} examples",
        ),
        "format_preference": (
            f"preserve all {f.option_count} results without choosing a presentation",
            "make no table-versus-prose decision for the requester",
            "retain the comparison content in its current unformatted state",
        ),
        "timeline_ambiguity": (
            "set no completion deadline",
            f"record {f.dependency} only as a dependency, not as a due date",
            "leave delivery timing open until a measurable interval is supplied",
        ),
        "team_request": (
            "send no team update",
            f"keep the {f.update_topic} message in draft form",
            "make no distribution choice before audience and timing are known",
        ),
        "data_request": (
            "export no records",
            f"leave the {f.dashboard_name} data untouched",
            "create no dataset extract without a source and date boundary",
        ),
        "travel_request": (
            "make no booking or itinerary commitment",
            f"hold the {f.destination} trip at the planning stage",
            "reserve no travel before origin and dates are confirmed",
        ),
        "purchasing_request": (
            "place no order",
            f"keep the {f.product} purchase unsubmitted",
            "make no spending commitment without quantity and budget",
        ),
    }[domain]
