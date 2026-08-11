from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BrainstormFacts:
    activity_material: str
    activity_minutes: int
    activity_participants: int
    event_attendees: int
    event_budget: int
    event_group_size: int
    event_groups: int
    event_minutes: int
    feature_approval_steps: int
    feature_review_hours: int
    feature_team_size: int
    lesson_learners: int
    lesson_material: str
    lesson_minutes: int
    name_audience: str
    name_quality: str
    name_word_limit: int
    outreach_capacity: int
    outreach_days: int
    outreach_partner: str
    prompt_audience: str
    prompt_draft_minutes: int
    prompt_word_limit: int
    workflow_items: int
    workflow_reviewers: int
    workflow_target_hours: int


def brainstorm_cases(facts: BrainstormFacts) -> dict[str, tuple[Any, ...]]:
    """Localize brainstorm cases from already dealt semantic facts."""

    activity_material = facts.activity_material
    activity_minutes = facts.activity_minutes
    activity_participants = facts.activity_participants
    event_attendees = facts.event_attendees
    event_budget = facts.event_budget
    event_group_size = facts.event_group_size
    event_groups = facts.event_groups
    event_minutes = facts.event_minutes
    feature_approval_steps = facts.feature_approval_steps
    feature_review_hours = facts.feature_review_hours
    feature_team_size = facts.feature_team_size
    lesson_learners = facts.lesson_learners
    lesson_material = facts.lesson_material
    lesson_minutes = facts.lesson_minutes
    name_audience = facts.name_audience
    name_quality = facts.name_quality
    name_word_limit = facts.name_word_limit
    outreach_capacity = facts.outreach_capacity
    outreach_days = facts.outreach_days
    outreach_partner = facts.outreach_partner
    prompt_audience = facts.prompt_audience
    prompt_draft_minutes = facts.prompt_draft_minutes
    prompt_word_limit = facts.prompt_word_limit
    workflow_items = facts.workflow_items
    workflow_reviewers = facts.workflow_reviewers
    workflow_target_hours = facts.workflow_target_hours
    cases = {
        "names": (
            (
                f"name a neighborhood tool library for {name_audience}; use at most {name_word_limit} words and keep the result {name_quality}",
                f"1. Tool Harbor — suggests shared access to {name_audience}. 2. Common Kit — emphasizes practical community use. 3. Borrow Bench — makes the borrowing action memorable. All fit the {name_word_limit}-word limit and aim to sound {name_quality}. Select Common Kit for its clearest meaning.",
            ),
            (
                f"name a free weekend reading circle for {name_audience}; use at most {name_word_limit} words with a {name_quality} tone",
                f"1. Open Pages — signals easy entry for {name_audience}. 2. Story Neighbors — emphasizes community. 3. First Chapter — welcomes people starting out. Each stays within {name_word_limit} words and supports a {name_quality} tone. Select Open Pages for immediate clarity.",
            ),
            (
                f"name a community seed exchange for {name_audience}; keep it to {name_word_limit} words and make it {name_quality}",
                f"1. Seed Circle — conveys exchange among {name_audience}. 2. Common Ground — stresses shared participation. 3. Garden Share — states the activity directly. All meet the {name_word_limit}-word ceiling and the {name_quality} criterion. Select Seed Circle for its clearest action.",
            ),
            (
                f"name a free neighborhood repair shelf for {name_audience}; stay within {name_word_limit} words and sound {name_quality}",
                f"1. Fix Shelf — names the shared repair resource directly. 2. Mend Corner — gives {name_audience} a friendly place cue. 3. Repair Share — links fixing with common access. Each remains within {name_word_limit} words and supports a {name_quality} tone. Select Fix Shelf because its purpose is immediately recognizable.",
            ),
        ),
        "lesson_activity": (
            (
                f"teach cause and effect to {lesson_learners} learners in {lesson_minutes} minutes using only {lesson_material}",
                f"1. Cause Chain — groups order {lesson_material} event sequences. 2. Change One Thing — learners predict an outcome after one variable changes. 3. Evidence Match — pairs connect claims to observations. All fit {lesson_minutes} minutes and use only {lesson_material} for the {lesson_learners} learners. Select Change One Thing for its direct observable check.",
            ),
            (
                f"teach equivalent fractions to {lesson_learners} learners in {lesson_minutes} minutes using {lesson_material}",
                f"1. Fold and Compare — align sections made from {lesson_material}. 2. Fraction Match — pair equal diagrams. 3. Missing Piece — complete one whole from partial representations. Each fits the {lesson_minutes}-minute limit and scales to {lesson_learners} learners. Select Fold and Compare because equality is directly visible.",
            ),
            (
                f"teach claims and evidence to {lesson_learners} learners in {lesson_minutes} minutes with {lesson_material}",
                f"1. Claim Sort — separate claims from observations on {lesson_material}. 2. Evidence Trail — link each claim to supporting material. 3. Source Ladder — rank support strength. All serve {lesson_learners} learners within {lesson_minutes} minutes using the supplied {lesson_material}. Select Evidence Trail for its explicit reasoning step.",
            ),
            (
                f"teach comparison and classification to {lesson_learners} learners in {lesson_minutes} minutes using only {lesson_material}",
                f"1. Attribute Grid — record visible properties on {lesson_material}. 2. Rule Switch — regroup examples after one criterion changes. 3. Odd One Out — defend which example does not fit. All three formats engage {lesson_learners} learners within {lesson_minutes} minutes using only {lesson_material}. Select Rule Switch because the changed criterion makes classification logic observable.",
            ),
        ),
        "event_plan": (
            (
                f"design a {event_minutes}-minute neighborhood event for {event_attendees} people with a ${event_budget} budget and step-free access",
                f"1. Skill Tables — {event_groups} rotating demonstrations. 2. Story Map — residents place anonymous local memories. 3. Repair Circle — shared guidance for small fixes. Each fits {event_minutes} minutes, accommodates all {event_attendees} people in {event_groups} step-free groups of at most {event_group_size}, and uses no more than ${event_budget} in common supplies. Select Skill Tables for flexible participation.",
            ),
            (
                f"design a quiet {event_minutes}-minute library event for {event_attendees} people with a ${event_budget} budget and step-free access",
                f"1. Mini Talks — {event_groups} short resident presentations. 2. Swap Shelf — exchange labeled recommendations. 3. Local Puzzle — solve a seated team challenge in groups of at most {event_group_size}. Each fits {event_minutes} minutes, seats {event_attendees} people with step-free access, remains quiet, and uses no more than ${event_budget}. Select Local Puzzle for shared participation.",
            ),
            (
                f"design a {event_minutes}-minute intergenerational event for {event_attendees} people within ${event_budget} without collecting participant data",
                f"1. Story Stations — share optional memories across {event_groups} tables. 2. Skill Exchange — demonstrate simple techniques. 3. Object Stories — discuss an everyday object. Each serves all {event_attendees} people within {event_minutes} minutes as rotations for {event_groups} groups of at most {event_group_size}, stays within ${event_budget}, and requires neither registration nor personal records. Select Skill Exchange for active participation.",
            ),
            (
                f"design a drop-in {event_minutes}-minute making session for {event_attendees} people within ${event_budget}, with step-free circulation",
                f"1. Shared Collage — contribute one piece at any time. 2. Pattern Tables — assemble reusable shapes in {event_groups} small groups. 3. Build-and-Pass — extend another participant's simple model. All accommodate {event_attendees} people through {event_groups} step-free stations of at most {event_group_size}, fit the {event_minutes}-minute window, and keep shared materials within ${event_budget}. Select Shared Collage because drop-in participation does not interrupt the activity.",
            ),
        ),
        "feature_ideas": (
            (
                f"reduce missed handoffs in a {feature_team_size}-person team without removing its {feature_approval_steps} approval checks",
                f"1. Owner Badge — show the responsible person across the {feature_team_size}-person team. 2. Ready Queue — list items that passed all {feature_approval_steps} checks. 3. Handoff Receipt — record sender, receiver, and time for each transfer. All preserve the review controls and can be evaluated within {feature_review_hours} hours. Select Handoff Receipt because it makes every transfer auditable.",
            ),
            (
                f"reduce forgotten approvals across {feature_approval_steps} review stages for a {feature_team_size}-person team while keeping the final decision human",
                f"1. Approval Timer — flag requests waiting longer than {feature_review_hours} hours. 2. Ready Signal — mark evidence packs complete across {feature_approval_steps} stages. 3. Decision Receipt — record the human reviewer and outcome. Each supports the {feature_team_size}-person team while retaining human authority. Select Ready Signal because it removes avoidable review starts.",
            ),
            (
                f"improve incident follow-up for a {feature_team_size}-person team with {feature_approval_steps} checkpoints and no automatic closure",
                f"1. Recovery Owner — show one accountable person for the {feature_team_size}-person team. 2. Checkpoint List — expose all {feature_approval_steps} unresolved checks. 3. Closure Note — require evidence and a human decision within the {feature_review_hours}-hour review window. All prevent silent automatic closure. Select Checkpoint List for continuous visibility.",
            ),
            (
                f"shorten review delays for a {feature_team_size}-person team while preserving {feature_approval_steps} mandatory approvals and a human release decision",
                f"1. Evidence Readiness — show which required attachments are missing before review. 2. Parallel Review Map — separate independent checks among the {feature_team_size} people. 3. Release Handoff — notify the named human approver only after all {feature_approval_steps} checks pass. Each keeps the release decision human and can be measured across the {feature_review_hours}-hour review window. Select Evidence Readiness because it prevents incomplete work from entering the queue.",
            ),
        ),
        "writing_prompts": (
            (
                f"create speculative-fiction prompts about memory for {prompt_audience}; each draft should stay below {prompt_word_limit} words and begin within {prompt_draft_minutes} minutes",
                f"1. A town forgets one street each sunrise. 2. A diver finds a memory labeled with tomorrow's date. 3. Two siblings remember the same childhood differently. All give {prompt_audience} a clear memory premise that can produce a draft under {prompt_word_limit} words within {prompt_draft_minutes} minutes. Select the diver prompt for its immediate mystery.",
            ),
            (
                f"create speculative-fiction prompts about unusual weather for {prompt_audience}; target drafts under {prompt_word_limit} words in {prompt_draft_minutes} minutes",
                f"1. Rain begins falling upward. 2. A storm calls residents by name. 3. Tomorrow's forecast describes yesterday. Each gives {prompt_audience} one accessible twist and supports a {prompt_word_limit}-word draft started within {prompt_draft_minutes} minutes. Select the named storm for its personal tension.",
            ),
            (
                f"create speculative-fiction prompts about ordinary objects for {prompt_audience}; keep drafts below {prompt_word_limit} words with {prompt_draft_minutes} minutes to begin",
                f"1. A key refuses every lock except one. 2. A chair remembers each person who sat in it. 3. A clock offers to trade an hour. All center one familiar object so {prompt_audience} can begin within {prompt_draft_minutes} minutes and remain below {prompt_word_limit} words. Select the clock for its immediate choice.",
            ),
            (
                f"create speculative-fiction prompts about mistaken identity for {prompt_audience}; support a first draft below {prompt_word_limit} words begun within {prompt_draft_minutes} minutes",
                f"1. A courier receives messages addressed to a future self. 2. A town mistakes a visitor for its missing cartographer. 3. Two strangers discover that every record has exchanged their names. Each gives {prompt_audience} a specific identity conflict that can begin within {prompt_draft_minutes} minutes and stay under {prompt_word_limit} words. Select the exchanged records because the consequence appears immediately.",
            ),
        ),
        "low_cost_activity": (
            (
                f"create a {activity_minutes}-minute indoor activity for {activity_participants} people using only {activity_material}",
                f"1. Paper Bridge — build for a fixed span with {activity_material}. 2. Sequence Swap — reorder illustrated events. 3. Constraint Sketch — draw under one changing rule. Each serves {activity_participants} people within {activity_minutes} minutes without specialist materials or hidden cost. Select Paper Bridge for a clear shared test.",
            ),
            (
                f"create a {activity_minutes}-minute teamwork activity for {activity_participants} people using {activity_material}",
                f"1. Silent Sort — arrange {activity_material} without speech. 2. Priority Relay — revise a shared ranking. 3. Pattern Build — reproduce a hidden sequence. Each uses only {activity_material}, includes {activity_participants} people, and fits {activity_minutes} minutes. Select Silent Sort for strong coordination practice.",
            ),
            (
                f"create a {activity_minutes}-minute reflection activity for {activity_participants} people using {activity_material}",
                f"1. Theme Wall — group anonymous observations on {activity_material}. 2. Decision River — order turning points. 3. Question Garden — cluster open questions. Each includes {activity_participants} people, needs only {activity_material}, and ends within {activity_minutes} minutes. Select Theme Wall for a concrete shared result.",
            ),
            (
                f"create a {activity_minutes}-minute problem-solving activity for {activity_participants} people with only {activity_material}",
                f"1. Constraint Tower — build while one rule changes each round. 2. Route Cards — connect destinations under a shared limit. 3. Error Hunt — find and repair a flawed sequence. Every option uses only {activity_material}, gives all {activity_participants} people an active role, and finishes within {activity_minutes} minutes. Select Error Hunt because success has a visible check.",
            ),
        ),
        "outreach": (
            (
                f"invite up to {outreach_capacity} local students to a free weekend science session through {outreach_partner} over {outreach_days} days without collecting personal data",
                f"1. Library Poster — direct readers to open attendance hours through {outreach_partner}. 2. School Bulletin — share a short teacher-ready notice. 3. Community Demo — offer a public preview. All can reach up to {outreach_capacity} students over {outreach_days} days without personal-data collection. Select School Bulletin for trusted distribution.",
            ),
            (
                f"invite up to {outreach_capacity} residents to a free repair workshop through {outreach_partner} over {outreach_days} days without online registration",
                f"1. Notice Board — post time and the {outreach_capacity}-person walk-in capacity. 2. Partner Bulletin — ask {outreach_partner} to share the notice. 3. Open Demo — preview one repair in public. All work across the {outreach_days}-day outreach window without registration. Select Partner Bulletin for broader trusted reach.",
            ),
            (
                f"invite up to {outreach_capacity} adult beginners to a free reading circle through {outreach_partner} over {outreach_days} days while keeping attendance optional",
                f"1. Library Slip — place a concise invitation through {outreach_partner}. 2. Community Calendar — list open meeting times. 3. Public Reading — demonstrate the format openly. Each preserves optional attendance while reaching toward {outreach_capacity} people during {outreach_days} days. Select Community Calendar for clear recurring access.",
            ),
            (
                f"invite up to {outreach_capacity} residents to a free skills exchange through {outreach_partner} during {outreach_days} days without building a contact list",
                f"1. Partner Poster — publish the walk-in format through {outreach_partner}. 2. Open Demonstration — show one sample activity in a public space. 3. Printed Bulletin — distribute the schedule without response tracking. All can reach toward {outreach_capacity} residents during {outreach_days} days without storing contact details. Select Partner Poster because the trusted channel and privacy boundary are both explicit.",
            ),
        ),
        "workflow": (
            (
                f"reduce delays across {workflow_items} review items handled by {workflow_reviewers} reviewers while retaining final human approval within {workflow_target_hours} hours",
                f"1. Intake Checklist — reject incomplete submissions among the {workflow_items} items early. 2. Parallel Evidence Check — let {workflow_reviewers} reviewers check independent facts together. 3. Approval Queue — surface only complete items for the final decision. All retain human approval and target {workflow_target_hours} hours. Select Intake Checklist because it prevents avoidable rework first.",
            ),
            (
                f"triage {workflow_items} incident records with {workflow_reviewers} reviewers inside {workflow_target_hours} hours while keeping closure under operator control",
                f"1. Evidence Pack — collect logs for all {workflow_items} records before review. 2. Parallel Diagnosis — distribute independent causes across {workflow_reviewers} reviewers. 3. Decision Gate — require operator sign-off before closure. Each preserves operator control while targeting {workflow_target_hours} hours. Select Evidence Pack because it improves every later step.",
            ),
            (
                f"reduce rework across {workflow_items} content items reviewed by {workflow_reviewers} editors within {workflow_target_hours} hours without bypassing sign-off",
                f"1. Brief Template — require audience and claims on all {workflow_items} items up front. 2. Independent Review — distribute fact and style checks across {workflow_reviewers} editors. 3. Release Receipt — record final editor approval. All preserve sign-off and the {workflow_target_hours}-hour target. Select Brief Template because it prevents incomplete drafts.",
            ),
            (
                f"clear a queue of {workflow_items} evidence requests with {workflow_reviewers} reviewers inside {workflow_target_hours} hours while reserving acceptance for a human approver",
                f"1. Missing-Evidence Gate — return requests lacking a required source before assignment. 2. Review Lanes — divide independent checks among {workflow_reviewers} reviewers. 3. Acceptance Record — attach the human approver's decision to every completed request. Together they cover all {workflow_items} requests without automating acceptance and keep the {workflow_target_hours}-hour service target visible. Select Missing-Evidence Gate because it removes non-reviewable work before scarce reviewer time is used.",
            ),
        ),
    }
    return cases
