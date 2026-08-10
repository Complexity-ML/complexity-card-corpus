from __future__ import annotations

from typing import Any

from ..variable_by import empathy_variable_by
from ..variable_by.reservoirs import ExplanationFacts, explanation_reservoir
from ..variable_by.templates import EMPATHY_TEMPLATES
from .core import (
    TaskHand,
    _code,
    _compose_subcards,
    _deal_task_frames,
    _lower_sentence_initial,
    _number,
    _pick,
    _render_domain,
)


def _explanation(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    rendered_domain = _render_domain(row)
    domain_label = rendered_domain.replace("_", " ")
    app_name = _pick(
        f"explain-app:{code}",
        (
            "a text editor", "a photo app", "a spreadsheet app", "a code editor",
            "a music player", "a note-taking app", "a video editor", "a slideshow app",
        ),
    )
    object_name = _pick(
        f"explain-object:{code}",
        (
            "a rock", "a backpack", "a bicycle", "a laptop",
            "a book", "a hammer", "a suitcase", "a chair",
        ),
    )
    cell_pair = _pick(
        f"explain-cells:{code}",
        (
            "a skin cell and a muscle cell",
            "a liver cell and a neuron",
            "a blood cell and a bone cell",
            "a fat cell and a nerve cell",
            "a kidney cell and a heart cell",
            "a lung cell and a stomach cell",
        ),
    )
    rect_w = _number(f"explain-rectw:{code}", 100, 999)
    rect_h = _number(f"explain-recth:{code}", 100, 999)
    principal = _number(f"explain-principal:{code}", 50, 950)
    interest = _number(f"explain-interest:{code}", 5, 95)
    bill_type = _pick(
        f"explain-bill:{code}",
        (
            "a zoning bill", "a budget bill", "a transportation bill", "an education bill",
            "a healthcare bill", "an energy bill", "a housing bill", "an agriculture bill",
        ),
    )
    topic = _pick(
        f"explain-topic:{code}",
        (
            "a product launch", "a policy change", "an election result", "a scientific study",
            "a merger announcement", "a court ruling", "a sports upset", "a weather event",
        ),
    )
    toss_count = _number(f"explain-tosses:{code}", 100, 999)
    heads_count = _number(
        f"explain-heads:{code}", toss_count // 3, (toss_count * 2) // 3
    )
    producer = _pick(
        f"explain-producer:{code}",
        ("A plant", "Algae", "A tree", "Phytoplankton", "Moss", "Seagrass"),
    )
    consumer = _pick(
        f"explain-consumer:{code}",
        ("an herbivore", "a grazing animal", "an insect", "a browsing deer", "a grazing rabbit"),
    )
    habitat_area = _number(f"explain-habitatarea:{code}", 100, 999)
    kw = _number(f"explain-kw:{code}", 100, 999)
    hours = _number(f"explain-hours:{code}", 100, 999)
    grammar_subject = _pick(
        f"explain-grammarsubj:{code}",
        ("Mira", "Omar", "Lea", "Theo", "Nadia", "Sam", "Priya", "Iris", "Kofi", "Yara"),
    )
    grammar_verb_phrase = _pick(
        f"explain-grammarverb:{code}",
        (
            "opens the window", "closes the door", "reads the letter", "paints the fence",
            "locks the gate", "carries the box", "waters the garden", "folds the map",
            "repairs the bicycle", "cleans the workshop",
        ),
    )
    grammar_adverb = _pick(
        f"explain-grammaradverb:{code}",
        (
            "carefully", "quietly", "quickly", "patiently", "eagerly",
            "slowly", "deliberately", "gracefully", "calmly", "confidently",
        ),
    )
    grammar_sentence = f"{grammar_subject} {grammar_adverb} {grammar_verb_phrase}"
    grammar_object = grammar_verb_phrase.rsplit(" ", 1)[-1]
    site = _pick(
        f"explain-site:{code}",
        (
            "example.org", "example.com", "openweb.dev", "docsite.io",
            "notesapp.net", "libraryhub.org", "mailrelay.io", "cloudpad.dev",
        ),
    )
    dns_ttl_seconds = _number(f"explain-dnsttl:{code}", 100, 999)
    correlated_pair = _pick(
        f"explain-correlated:{code}",
        (
            "ice-cream sales and sunburns",
            "umbrella sales and traffic accidents",
            "hot chocolate sales and heating bills",
            "sunglasses sales and lawn-mower rentals",
            "fan sales and swimming-pool visits",
        ),
    )
    mean_low = _number(f"explain-meanlow:{code}", 1, 4)
    mean_mid = _number(f"explain-meanmid:{code}", 5, 8)
    mean_outlier = _number(f"explain-meanoutlier:{code}", 40, 300)
    dataset = (mean_low, mean_mid, mean_outlier)
    dataset_mean = round(sum(dataset) / 3, 1)
    minutes_open = _number(f"explain-minutesopen:{code}", 100, 999)
    backup_version = _number(f"explain-backupversion:{code}", 100, 999)
    weight_kg = _number(f"explain-weightkg:{code}", 1, 48)
    gene_count = _number(f"explain-genecount:{code}", 100, 999)
    chamber_size = _number(f"explain-chambersize:{code}", 100, 999)
    vote_margin = _number(f"explain-votemargin:{code}", 10, 99)
    interview_number = _number(f"explain-interviewnum:{code}", 1_000, 9_999)
    port_number = _number(f"explain-port:{code}", 10_000, 65_535)
    sample_days = _number(f"explain-sampledays:{code}", 100, 999)
    lessons = explanation_reservoir(
        ExplanationFacts(
            app_name=app_name,
            backup_version=backup_version,
            bill_type=bill_type,
            cell_pair=cell_pair,
            chamber_size=chamber_size,
            consumer=consumer,
            correlated_pair=correlated_pair,
            dataset_mean=dataset_mean,
            dns_ttl_seconds=dns_ttl_seconds,
            gene_count=gene_count,
            grammar_object=grammar_object,
            grammar_sentence=grammar_sentence,
            grammar_subject=grammar_subject,
            habitat_area=habitat_area,
            heads_count=heads_count,
            hours=hours,
            interest=interest,
            interview_number=interview_number,
            kw=kw,
            mean_low=mean_low,
            mean_mid=mean_mid,
            mean_outlier=mean_outlier,
            minutes_open=minutes_open,
            object_name=object_name,
            port_number=port_number,
            principal=principal,
            producer=producer,
            rect_h=rect_h,
            rect_w=rect_w,
            sample_days=sample_days,
            site=site,
            topic=topic,
            toss_count=toss_count,
            vote_margin=vote_margin,
            weight_kg=weight_kg,
        )
    )
    mechanism, example, check, transfer_pool = lessons[rendered_domain]
    embedded_mechanism = _lower_sentence_initial(mechanism)
    data, goal = _deal_task_frames(
        row,
        variant,
        "explanation",
        (
            f"Concept notes: {mechanism} Example available: {example}",
            f"Learning card: {mechanism} Worked example: {example}",
            f"Teach from these supplied notes — mechanism: {mechanism} Example: {example}",
        ),
        (
            f"Explain the {domain_label} mechanism '{mechanism}', apply '{example}', and finish with the check '{check}'.",
            f"Connect the {domain_label} idea '{mechanism}' to the worked case '{example}', then use '{check}' as the transfer question.",
            f"Give a concise {domain_label} explanation of '{mechanism}', demonstrate it through '{example}', and end by asking '{check}'.",
        ),
    )
    answer = _compose_subcards(
        row,
        variant,
        "explanation-answer",
        (
            (
                f"Core idea: {mechanism}",
                f"Core idea: in plain terms, {embedded_mechanism}",
                f"Core idea: the key distinction is that {embedded_mechanism}",
            ),
            (
                f"Example: {example} This applies the distinction directly.",
                f"Example: {example} This turns the definition into a checkable case.",
                f"Example: {example} The example makes the mechanism visible.",
            ),
            transfer_pool,
            (
                f"Check: {check}",
                f"Check: As a transfer test, {_lower_sentence_initial(check)}",
                f"Check: To verify the idea, {_lower_sentence_initial(check)}",
            ),
        ),
        pool_names=("mechanism", "example", "related_case", "transfer_check"),
    )
    return TaskHand(data, goal, answer, ("mechanism", "example", "question"))


def _writing(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    writing_domain = _render_domain(row)
    writing_label = writing_domain.replace("_", " ")
    owner = _pick(f"owner:{code}", ("Maya", "Jon", "Ari", "Lea", "Noah", "Iris"))
    day = _number(f"write-day:{code}", 10, 28)
    content_cards = {
        "email": (
            f"notes {code}: send team; review complete; two figures need captions; {owner} owns them; target day {day}; release waits",
            f"Subject: Review {code} next steps\n\nThe review is complete. Two figures still need captions, which {owner} owns for day {day}. The release decision remains pending.",
        ),
        "project_update": (
            f"update {code}: review complete; captions missing on two figures; owner {owner}; target day {day}; release decision blocked",
            f"Project update {code}: Review is complete. Remaining work: {owner} adds captions to two figures, with completion expected day {day}. Blocker: the release decision remains pending.",
        ),
        "support_reply": (
            f"case {code}: issue reviewed; two screenshots need labels; {owner} will add them by day {day}; resolution waits for review",
            f"Support reply {code}: We have completed the issue review. {owner} will label the two remaining screenshots ahead of day {day}. We will confirm resolution after that review.",
        ),
        "meeting_notes": (
            f"meeting {code}: review complete; two captions outstanding; {owner}; day {day}; no release decision yet",
            f"Meeting {code} — Decision: review complete. Action: {owner} adds two captions, day {day} at the latest. Open item: no release decision has been made.",
        ),
        "technical_explanation": (
            f"draft {code}: validation complete; two diagrams lack captions; {owner} adds them by day {day}; publication waits",
            f"Technical note {code}: Validation is complete, but two diagrams still lack captions. {owner} will add them, deadline day {day}; publication timing remains undecided until they are reviewed.",
        ),
        "public_notice": (
            f"notice {code}: east entrance closed day {day}; inspection; use west entrance; {owner} posts signs; reopening not confirmed",
            f"Public notice {code}: The east entrance will be closed for inspection starting day {day}. Please use the west entrance. {owner} will post directions; the reopening time is not yet confirmed.",
        ),
        "handover_note": (
            f"handover {code}: source review done; two tables pending; {owner} owns checks day {day}; export not started",
            f"Handover {code}: Source review is complete. {owner} will check the two pending tables, cutting off at day {day}. The export has not started.",
        ),
        "schedule_change": (
            f"schedule {code}: review moved from day {day - 1} to day {day}; room unchanged; {owner} confirms attendees; reason not provided",
            f"Schedule change {code}: The review has moved from day {day - 1} to day {day}; the room is unchanged. {owner} will confirm attendance. No reason for the change was provided.",
        ),
        "feedback_message": (
            f"feedback {code}: summary accurate; main decision appears after background; ask {owner} to move it first by day {day}; no content change",
            f"Feedback {code}: The summary is accurate, but the main decision appears after the background. {owner}, please move the decision to the opening, aiming for day {day}, without changing the content.",
        ),
        "procedure_summary": (
            f"procedure {code}: preserve original; duplicate file; {owner} validates copy day {day}; publish only after match; fallback unspecified",
            f"Procedure {code}: Preserve the original, duplicate the file, and have {owner} validate the copy, day {day} being the last acceptable date. Publish only after both versions match; no fallback is specified.",
        ),
        "event_invitation": (
            f"invite {code}: open review session; team audience; room {day}; 15:00 day {day}; reply to {owner} by day {day - 2}; agenda pending",
            f"Invitation {code}: Join the team review session in Room {day} at 15:00 on day {day}. Please reply to {owner} by day {day - 2}. The agenda is still pending.",
        ),
        "progress_brief": (
            f"brief {code}: 8 of 10 records checked; two awaiting sources; {owner} requests them day {day}; final count pending",
            f"Progress brief {code}: Eight of ten records are checked. Two still await source documents, which {owner} will request, with day {day} marking the deadline. The final count remains pending.",
        ),
    }
    source, content = content_cards[writing_domain]
    data, goal = _deal_task_frames(
        row,
        variant,
        "writing",
        (
            f"Source text: {source}. Intended reader: the project team.",
            f"Notes to rewrite for the project team: {source}.",
            f"Project-team input {code}: {source}.",
        ),
        (
            f"Rewrite the {writing_label} source as a short update that keeps {owner}'s ownership and the day {day} timing without new commitments.",
            f"Produce a team-ready {writing_label} version preserving {owner}, day {day}, and every unresolved limit in the source.",
            f"Turn the {writing_label} notes into a direct update while retaining {owner}'s action, day {day}, and all unresolved points.",
        ),
    )
    faithful_cards = (
        content,
        f"Here is the revised text: {content}",
        f"The concise version is: {content}",
    )
    answer = _compose_subcards(
        row,
        variant,
        "writing-answer",
        (
            ("", "Clear rewrite:", "Short version:", "Team update:"),
            faithful_cards,
        ),
        pool_names=("layout", "faithful_content"),
        links=(
            (
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (2, 0),
                (3, 0),
            ),
        ),
    )
    return TaskHand(data, goal, answer, ("faithful_rewrite", "owner", "timing"))


def _empathy(row: dict[str, Any], variant: int) -> TaskHand:
    rendered_domain = _render_domain(row)
    variables = empathy_variable_by(
        rendered_domain,
        state=str(row.get("state", "")),
    )
    data = _compose_subcards(
        row,
        variant,
        "empathy-data",
        (EMPATHY_TEMPLATES["data"],),
        pool_names=("source_message",),
        variable_by=variables,
    )
    goal = _compose_subcards(
        row,
        variant,
        "empathy-goal",
        (EMPATHY_TEMPLATES["goal"],),
        pool_names=("response_goal",),
        variable_by=variables,
    )
    answer = _compose_subcards(
        row,
        variant,
        "empathy-answer",
        (
            EMPATHY_TEMPLATES["answer_grounding"],
            EMPATHY_TEMPLATES["answer_agency"],
        ),
        pool_names=("grounded_reflection", "agency_and_question"),
        variable_by=variables,
    )
    return TaskHand(
        data,
        goal,
        answer,
        ("acknowledgment", "state_reflection", "agency", "question"),
    )
def _clarification(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    weekday = _pick(
        f"clarify-day:{code}", ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
    )
    report_id = _number(f"clarify-report:{code}", 100, 999)
    word_limit = _number(f"clarify-wordlimit:{code}", 15, 60)
    lead_event = _pick(
        f"clarify-event:{code}", ("the review", "the meeting", "the call", "the demo")
    )
    project_name = _pick(
        f"clarify-project:{code}",
        (
            "the onboarding project",
            "the migration project",
            "the redesign project",
            "the launch project",
        ),
    )
    doc_count = _number(f"clarify-doccount:{code}", 2, 6)
    option_count = _number(f"clarify-optioncount:{code}", 3, 8)
    dependency = _pick(
        f"clarify-dependency:{code}",
        (
            "the next review",
            "the design sign-off",
            "the budget approval",
            "the next release",
        ),
    )
    update_topic = _pick(
        f"clarify-topic:{code}",
        (
            "the pricing change",
            "the outage postmortem",
            "the roadmap update",
            "the security patch",
        ),
    )
    dashboard_name = _pick(
        f"clarify-dashboard:{code}",
        ("sales dashboard", "support dashboard", "analytics dashboard", "billing dashboard"),
    )
    destination = _pick(f"clarify-destination:{code}", ("Porto", "Lisbon", "Faro", "Braga"))
    product = _pick(
        f"clarify-product:{code}",
        ("monitors", "laptops", "headsets", "keyboards", "docking stations"),
    )
    case_ref = _number(f"clarify-ref:{code}", 10_000, 99_999)
    cases = {
        "ambiguous_request": (
            f"Please move the review to {weekday}.",
            f"The review should move to {weekday}, but the affected item is not identified.",
            f"Do you mean reschedule the review meeting or change the document deadline to {weekday}?",
            f"leave both the meeting and document deadline unchanged until the {weekday} ambiguity is resolved",
        ),
        "missing_reference": (
            f"Summarize the attached report {report_id}.",
            f"A summary of report {report_id} is requested, but the report itself is missing.",
            f"Could you attach report {report_id} so it can be summarized?",
            "do not draft a substitute summary",
        ),
        "conflicting_instruction": (
            f"Keep every detail, but make the answer no longer than {word_limit} words.",
            f"The request requires both complete detail and a {word_limit}-word limit.",
            f"Which takes priority: preserving every detail or limiting the answer to {word_limit} words?",
            "preserve the original text without rewriting it",
        ),
        "unclear_pronoun": (
            f"Send it to them after {lead_event}.",
            f"A send is requested after {lead_event}, but the item and recipient are unresolved.",
            f"What does 'it' refer to, and who should receive it after {lead_event}?",
            "send nothing",
        ),
        "incomplete_goal": (
            f"Help me organize {project_name}.",
            f"Organizing {project_name} is requested, but the required deliverable is unspecified.",
            f"What outcome should organizing {project_name} produce: a schedule, a task list, or a file structure?",
            "make no structural change to the project",
        ),
        "scope_boundary": (
            f"Update the {doc_count} examples and anything else that needs work.",
            f"The {doc_count} examples should change, while the surrounding revision scope remains open.",
            f"Should I change only the {doc_count} examples, or also revise the surrounding explanation and tests?",
            f"prepare no edits beyond the {doc_count} examples",
        ),
        "format_preference": (
            f"Give me the comparison results for the {option_count} options.",
            f"The comparison results for {option_count} options are requested, but their presentation format is unspecified.",
            f"Would you like a short table, a prose summary, or both for the {option_count} options?",
            f"preserve all {option_count} results without choosing a final presentation",
        ),
        "timeline_ambiguity": (
            f"Finish this soon after {dependency}.",
            f"Completion should follow {dependency}, but no deadline is defined.",
            f"What calendar date or time limit should 'soon after {dependency}' mean?",
            "set no completion deadline",
        ),
        "team_request": (
            f"Share the {update_topic} update with the team before the review.",
            f"An update about {update_topic} should be shared before the review, but the team group and review time are not identified.",
            f"Which team group should receive the {update_topic} update, and when is the review scheduled?",
            "send no team update",
        ),
        "data_request": (
            f"Send me the recent records from the {dashboard_name}.",
            f"Recent {dashboard_name} records are requested, but the dataset and date range are not defined.",
            f"Which dataset and exact date range should the {dashboard_name} export contain?",
            "export no records",
        ),
        "travel_request": (
            f"Plan the trip to {destination} around the workshop.",
            f"{destination} and the workshop are known, but the travel dates and departure city are unresolved.",
            f"What are the travel dates and departure city for the {destination} trip?",
            "make no booking or itinerary commitment",
        ),
        "purchasing_request": (
            f"Order new {product} for the design team.",
            f"The design team needs {product}, but the quantity and spending limit are unspecified.",
            f"How many {product} are needed, and what is the maximum budget per unit?",
            "place no order",
        ),
    }
    rendered_domain = _render_domain(row)
    clarification_label = rendered_domain.replace("_", " ")
    ambiguous, restatement, question, reversible_default = cases[rendered_domain]
    restatement = f"{restatement} (request #{case_ref})"
    situation_titles = {
        "ambiguous_request": "Ambiguous request — identify the affected item",
        "missing_reference": "Missing reference — request the absent report",
        "conflicting_instruction": "Conflicting instructions — choose the controlling requirement",
        "unclear_pronoun": f"Unclear references — identify the item and recipient for request {case_ref}",
        "incomplete_goal": "Incomplete goal — identify the required deliverable",
        "scope_boundary": "Open scope — bound the requested revision",
        "format_preference": "Unspecified format — choose the presentation",
        "timeline_ambiguity": "Unspecified timeline — define the deadline",
        "team_request": f"Incomplete team request — identify audience and timing for request {case_ref}",
        "data_request": f"Incomplete data request — identify dataset and range for request {case_ref}",
        "travel_request": f"Incomplete travel request — identify dates and origin for request {case_ref}",
        "purchasing_request": f"Incomplete purchase request — identify quantity and budget for request {case_ref}",
    }
    situation_cards = {
        "ambiguous_request": f"The requested {weekday} change could affect either a meeting or a document deadline.",
        "missing_reference": f"The requested summary of report {report_id} cannot be grounded because the referenced report is absent.",
        "conflicting_instruction": f"The completeness requirement and the {word_limit}-word limit cannot both be guaranteed.",
        "unclear_pronoun": f"The requested send after {lead_event} cannot proceed until both the item and recipient are identified.",
        "incomplete_goal": f"Organizing {project_name} may need a schedule, task list, or file structure, but no deliverable is selected.",
        "scope_boundary": f"The {doc_count} examples are in scope; changes to the explanation and tests are not yet authorized.",
        "format_preference": f"The comparison results for {option_count} options are available, but the requested presentation format is open.",
        "timeline_ambiguity": f"{dependency.capitalize()} is a known dependency, but the completion deadline is undefined.",
        "team_request": f"The {update_topic} update exists, but the intended team audience and review time remain open.",
        "data_request": f"An export from the {dashboard_name} is requested, but neither the source dataset nor the requested interval is established.",
        "travel_request": f"The destination, {destination}, and the workshop are known, while the travel dates and departure point remain open.",
        "purchasing_request": f"The {product} category and intended team are known, but quantity and budget are unresolved.",
    }
    data, goal = _deal_task_frames(
        row,
        variant,
        "clarification",
        (
            f'Request {code}: "{ambiguous}" {restatement}',
            f'Clarification case {code}: "{ambiguous}" Known so far: {restatement}',
            f'Unresolved request {code} — "{ambiguous}" Supported reading: {restatement}',
        ),
        (
            f"Restate the {clarification_label} request '{ambiguous}', ask '{question}', and preserve the reversible default: {reversible_default}.",
            f"Identify what remains ambiguous in '{ambiguous}', ask exactly '{question}', and keep this default: {reversible_default}.",
            f"Bound the {clarification_label} interpretation of '{ambiguous}', request the missing choice through '{question}', and meanwhile {reversible_default}.",
        ),
    )
    answer = _compose_subcards(
        row,
        variant,
        "clarification-answer",
        (
            (
                f"Understood: {restatement}",
                f"My current reading: {restatement}",
                f"What is clear: {restatement}",
                f"The supported interpretation is limited: {restatement}",
            ),
            (
                question,
                f"One point to resolve: {question}",
                f"Before proceeding: {question}",
            ),
            (
                f"Until confirmed, {reversible_default}.",
                f"For now, {reversible_default}.",
                f"Pending that answer, {reversible_default}.",
                f"As a reversible default, {reversible_default}.",
            ),
        ),
        pool_names=(
            "restatement",
            "clarifying_question",
            "reversible_default",
        ),
    )
    return TaskHand(
        data,
        goal,
        answer,
        ("restatement", "one_question", "reversible_default"),
        situation_title=situation_titles[rendered_domain],
        situation=situation_cards[rendered_domain],
    )
