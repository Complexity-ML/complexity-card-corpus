from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CritiqueFacts:
    action_name: str
    affected_user_count: int
    arg_success: int
    arg_total: int
    artifact_type: str
    attendee_count: int
    blocked_days: int
    budget_amount: int
    caption_product: str
    change_type: str
    comment_count: int
    cutoff_hour: int
    data_category: str
    data_noun: str
    doc_qualifier: str
    doc_type: str
    error_code: int
    exception_count: int
    exposure_amount: int
    feature_name: str
    feature_qualifier: str
    file_count: int
    incident_count: int
    initiative_area: str
    initiative_noun: str
    key_bits: int
    location_name: str
    metric_delta: int
    page_count: int
    percent_complete: int
    plan_days: int
    product_area: str
    quiet_weeks: int
    record_count: int
    release_component: str
    release_ver: int
    retry_limit: int
    risk_system: str
    support_feature: str
    support_issue: str
    survey_selected: int
    survey_total: int
    surveyed_feature: str
    surveyed_qualifier: str
    system_area: str
    system_noun: str
    team_name: str
    team_noun: str
    tested_system_count: int
    topic_count: int
    update_minutes: int
    venue_noun: str
    venue_qualifier: str
    view_count: int


def critique_cases(facts: CritiqueFacts) -> dict[str, tuple[Any, ...]]:
    """Localize critique cases from already dealt semantic facts."""

    action_name = facts.action_name
    affected_user_count = facts.affected_user_count
    arg_success = facts.arg_success
    arg_total = facts.arg_total
    artifact_type = facts.artifact_type
    attendee_count = facts.attendee_count
    blocked_days = facts.blocked_days
    budget_amount = facts.budget_amount
    caption_product = facts.caption_product
    change_type = facts.change_type
    comment_count = facts.comment_count
    cutoff_hour = facts.cutoff_hour
    data_category = facts.data_category
    data_noun = facts.data_noun
    doc_qualifier = facts.doc_qualifier
    doc_type = facts.doc_type
    error_code = facts.error_code
    exception_count = facts.exception_count
    exposure_amount = facts.exposure_amount
    feature_name = facts.feature_name
    feature_qualifier = facts.feature_qualifier
    file_count = facts.file_count
    incident_count = facts.incident_count
    initiative_area = facts.initiative_area
    initiative_noun = facts.initiative_noun
    key_bits = facts.key_bits
    location_name = facts.location_name
    metric_delta = facts.metric_delta
    page_count = facts.page_count
    percent_complete = facts.percent_complete
    plan_days = facts.plan_days
    product_area = facts.product_area
    quiet_weeks = facts.quiet_weeks
    record_count = facts.record_count
    release_component = facts.release_component
    release_ver = facts.release_ver
    retry_limit = facts.retry_limit
    risk_system = facts.risk_system
    support_feature = facts.support_feature
    support_issue = facts.support_issue
    survey_selected = facts.survey_selected
    survey_total = facts.survey_total
    surveyed_feature = facts.surveyed_feature
    surveyed_qualifier = facts.surveyed_qualifier
    system_area = facts.system_area
    system_noun = facts.system_noun
    team_name = facts.team_name
    team_noun = facts.team_noun
    tested_system_count = facts.tested_system_count
    topic_count = facts.topic_count
    update_minutes = facts.update_minutes
    venue_noun = facts.venue_noun
    venue_qualifier = facts.venue_qualifier
    view_count = facts.view_count
    cases = {
        "email_draft": (
            f"Send the {page_count}-page {doc_qualifier} {doc_type} and its {file_count} attachments soon because everyone should know what I mean.",
            f"the request has no recipient, deadline, or names for the {page_count}-page {doc_qualifier} {doc_type} or its {file_count} attachments",
            (
                f"Please send the {page_count}-page {doc_qualifier} {doc_type} and its {file_count} attachments. First confirm the recipient, deadline, and file names.",
                f"Confirm who should receive the {page_count}-page {doc_qualifier} {doc_type}, when it is due, and which {file_count} files belong with it. Send the document and attachments only after those details are supplied.",
                f"The {page_count}-page {doc_qualifier} {doc_type} has {file_count} attachments awaiting delivery. Obtain the recipient, deadline, and attachment names before requesting that anyone send them.",
                f"Do not dispatch the {page_count}-page {doc_qualifier} {doc_type} or its {file_count} attachments yet. Ask for the missing recipient, due date, and file names first.",
            ),
            (
                f"Without those three details, no one receiving the {page_count}-page {doc_qualifier} {doc_type} could act on sending it and its {file_count} attachments correctly.",
                f"The message asks for the {page_count}-page {doc_qualifier} {doc_type} and {file_count} attachments to be sent but supplies none of the specific information a recipient would need to do that.",
                f"No recipient, deadline, or file names for the {page_count}-page {doc_qualifier} {doc_type} means it and its {file_count} attachments cannot actually be sent as requested.",
            ),
        ),
        "argument": (
            f"Our trial proves the {feature_qualifier} {feature_name} is always faster because {arg_success} of {arg_total} testers finished sooner.",
            f"a universal claim about the {feature_qualifier} {feature_name} is not supported by {arg_success} successes among {arg_total} testers across {tested_system_count} configurations",
            (
                f"{arg_success} of {arg_total} testers finished sooner with the {feature_qualifier} {feature_name} in this trial. That result does not establish that it is always faster.",
                f"In this trial, the {feature_qualifier} {feature_name} produced faster completion for {arg_success} of {arg_total} testers across {tested_system_count} configurations. The evidence supports only that observed result, not a universal speed claim.",
                f"The {feature_qualifier} {feature_name} was faster for {arg_success} participants in the {arg_total}-person trial. Performance on these {tested_system_count} configurations cannot establish what will happen in every use.",
                f"Report the trial result as {arg_success} faster completions among {arg_total} testers using the {feature_qualifier} {feature_name}. Avoid claiming it is always faster because the evidence covers only {tested_system_count} tested configurations.",
            ),
            (
                f"A result from {arg_success} of {arg_total} testers across {tested_system_count} configurations describes only this one trial of the {feature_qualifier} {feature_name}, not every future run.",
                f"Generalizing from {arg_success} of {arg_total} testers on the {feature_qualifier} {feature_name} to an always-faster claim requires evidence the draft does not provide.",
                f"The {arg_success}-of-{arg_total} result for the {feature_qualifier} {feature_name}, gathered across {tested_system_count} configurations, is evidence about this trial alone.",
            ),
        ),
        "project_plan": (
            f"Build the {product_area} {artifact_type} on a ${budget_amount} budget, test it, and launch in {plan_days} days.",
            f"the ${budget_amount} plan for the {product_area} {artifact_type} gives no owner, dependency, or completion criterion",
            (
                f"Build and test the {product_area} {artifact_type} on its ${budget_amount} budget before the {plan_days}-day launch. Assign an owner, dependencies, completion criteria, and a launch date before execution.",
                f"The {product_area} {artifact_type} has a ${budget_amount} budget and a {plan_days}-day target. Name the owner, dependencies, completion test, and dated launch milestone before work begins.",
                f"Keep the {product_area} {artifact_type} within ${budget_amount} and test it before launch. Execution should wait until ownership, dependencies, completion criteria, and the date inside the {plan_days}-day window are explicit.",
                f"Plan the {product_area} {artifact_type} against its ${budget_amount} limit and {plan_days}-day window. Record responsibility, prerequisites, a definition of done, and the launch date before starting.",
            ),
            (
                f"Without a named owner or completion criterion, no one can confirm when the ${budget_amount} {product_area} {artifact_type} is actually done, only that {plan_days} days passed.",
                f"This {plan_days}-day, ${budget_amount} schedule for the {product_area} {artifact_type} alone cannot be executed without knowing who is responsible for each step.",
                f"A {plan_days}-day timeline for the ${budget_amount} {product_area} {artifact_type} without an owner or completion criterion cannot be tracked to completion.",
            ),
        ),
        "explanation": (
            f"Encryption makes {record_count} {data_category} {data_noun} safe by turning them into random text using a {key_bits}-bit key.",
            f"the explanation of encrypting {record_count} {data_category} {data_noun} omits how the key is protected and overstates safety",
            (
                f"Encryption transforms {record_count} {data_category} {data_noun} using a {key_bits}-bit key. Authorized holders can reverse it, while security still depends on key protection and implementation.",
                f"A {key_bits}-bit key transforms the {record_count} {data_category} {data_noun} into ciphertext that an authorized key holder can reverse. Protection still depends on how that key and the implementation are secured.",
                f"The {record_count} {data_category} {data_noun} are encrypted with a {key_bits}-bit key rather than made inherently safe. Anyone with authorized key access can decrypt them, so key handling and implementation remain part of the security boundary.",
                f"Encryption uses a {key_bits}-bit key to convert the {record_count} {data_category} {data_noun} into recoverable ciphertext. Its security is limited by both the protection of the key and the correctness of the implementation.",
            ),
            (
                f"Calling the result merely random text hides the specific role the {key_bits}-bit key plays in protecting {record_count} {data_category} {data_noun}.",
                f"This explanation overstates the safety of {record_count} {data_category} {data_noun} without mentioning how the {key_bits}-bit key is protected.",
                f"The {key_bits}-bit key, not just the transformation itself, is what makes {record_count} {data_category} {data_noun} recoverable.",
            ),
        ),
        "instructions": (
            f"Install the {update_minutes}-minute update for the {system_area} {system_noun} affecting {affected_user_count} users, delete the old folder, and check whether it works.",
            f"the destructive deletion in the {system_area} {system_noun} update comes before verification or backup",
            (
                f"Back up the old folder and install the {update_minutes}-minute {system_area} {system_noun} update separately. Verify the application before deleting anything, and retain rollback until the checks pass.",
                f"Preserve a restorable copy of the old folder before starting the {update_minutes}-minute {system_area} {system_noun} update. Test the separate installation for all {affected_user_count} users and remove nothing until verification succeeds.",
                f"Install the {system_area} {system_noun} update in isolation after backing up the existing folder. Keep that rollback path intact through the {update_minutes}-minute update and its validation checks.",
                f"Create a verified backup, then apply the {update_minutes}-minute update to a separate {system_area} {system_noun} location. Deletion should occur only after the application passes its checks and rollback is no longer needed.",
            ),
            (
                f"Deleting before verifying removes the only available fallback if the {system_area} {system_noun} update, which reaches {affected_user_count} users, turns out to be broken.",
                f"Without a separate backup step first, an unnoticed failure in the {system_area} {system_noun} update becomes unrecoverable for {affected_user_count} users.",
                f"Verifying only after deletion leaves no way back if the {update_minutes}-minute {system_area} {system_noun} update fails for its {affected_user_count} users.",
            ),
        ),
        "summary": (
            f"The {team_name} {team_noun}'s {attendee_count}-person meeting discussed {topic_count} topics and everyone agreed the project was important.",
            f"the summary of the {team_name} {team_noun}'s {attendee_count}-person meeting omits the actual decision and action",
            (
                f"The notes from the {team_name} {team_noun}'s {attendee_count}-person meeting record only that {topic_count} topics were discussed and the project was considered important. Add the actual decision and assigned action before using this as a complete summary.",
                f"This record says the {team_name} {team_noun}'s {attendee_count} participants covered {topic_count} topics but identifies no outcome. Include the decision and its assigned follow-up before presenting the notes as a summary.",
                f"The {attendee_count}-person {team_name} {team_noun} meeting covered {topic_count} topics, yet the notes preserve neither a decision nor an owner. Supply those operational details before calling the record complete.",
                f"Discussion of {topic_count} topics by the {team_name} {team_noun} does not reveal what its {attendee_count} participants decided. Revise the summary to state the decision and assigned action.",
            ),
            (
                f"A summary of the {team_name} {team_noun}'s {attendee_count}-person meeting that omits the decision and action leaves nothing concrete for a reader to follow up on.",
                f"Recording only that {attendee_count} people on the {team_name} {team_noun} discussed {topic_count} topics provides no operational detail to act on.",
                f"Recording {topic_count} topics from the {team_name} {team_noun}'s {attendee_count}-person meeting as merely important gives a reader nothing concrete to follow up on.",
            ),
        ),
        "claim_evidence": (
            f"Users prefer the redesign; {comment_count} positive comments on the {venue_qualifier} {venue_noun} ({view_count} views) prove it.",
            f"{comment_count} comments among {view_count} views on the {venue_qualifier} {venue_noun} cannot support a general preference claim",
            (
                f"{comment_count} respondents commented positively on the {venue_qualifier} {venue_noun}, which recorded {view_count} views, about the redesign. Broader user preference remains unmeasured.",
                f"The redesign received {comment_count} positive comments on a {venue_qualifier} {venue_noun} with {view_count} views. Those comments describe their authors and do not measure preference across all viewers or users.",
                f"Among {view_count} views of the {venue_qualifier} {venue_noun}, {comment_count} people left favorable redesign comments. No representative sample is provided for a general user-preference conclusion.",
                f"Report the observed evidence as {comment_count} positive comments attached to {view_count} views of the {venue_qualifier} {venue_noun}. Leave the wider population's preference unresolved.",
            ),
            (
                f"{comment_count} comments out of {view_count} views on the {venue_qualifier} {venue_noun} describe only {comment_count} people's individual reactions, not the broader user base.",
                f"Treating {comment_count} comments among {view_count} views on the {venue_qualifier} {venue_noun} as proof skips the sampling that a general claim would need.",
                f"Extrapolating from {comment_count} comments on {view_count} views of the {venue_qualifier} {venue_noun} to the whole user base skips the sampling that claim would require.",
            ),
        ),
        "interface_copy": (
            f"Error {error_code}. The {action_name} could not be completed after {retry_limit} attempts. Try again.",
            f"the message about the {action_name} failing after {retry_limit} attempts gives neither the failed action nor a useful next step",
            (
                f"The requested {action_name} could not be completed after {retry_limit} attempts (error {error_code}). Review the available error details before trying again.",
                f"Error {error_code} occurred while attempting the {action_name}, which has now failed {retry_limit} times. Inspect the recorded details before deciding whether another attempt is useful.",
                f"The {action_name} remains incomplete after {retry_limit} tries and produced error {error_code}. Check the error information for a specific next step instead of repeating the same action blindly.",
                f"After {retry_limit} unsuccessful attempts, the {action_name} returned error {error_code}. Use the accompanying diagnostics to determine the next action before retrying.",
            ),
            (
                f"Without naming what failed beyond code {error_code}, a user retrying the {action_name} {retry_limit} times has no real way to know what it accomplished.",
                f"A vague error message about the {action_name} failing {retry_limit} times leaves the user unable to distinguish a worthwhile retry from a dead end.",
                f"Code {error_code} alone does not tell the user what part of the {action_name} failed across {retry_limit} attempts or what a retry would change.",
            ),
        ),
        "status_update": (
            f"The {initiative_area} {initiative_noun}, {percent_complete}% complete, is on track, although integration has been blocked for {blocked_days} days and the delivery date is no longer known.",
            f"the opening claim about the {percent_complete}%-complete {initiative_area} {initiative_noun} conflicts with the stated blocker and missing delivery date",
            (
                f"Core work on the {initiative_area} {initiative_noun} is {percent_complete}% complete and progressing, but integration has been blocked for {blocked_days} days. Reassess the delivery date after that blocker is resolved.",
                f"The {initiative_area} {initiative_noun} has reached {percent_complete}% completion, while its integration work has remained blocked for {blocked_days} days. Delivery timing is therefore unknown until the blocker clears.",
                f"Report the {initiative_area} {initiative_noun} as {percent_complete}% complete rather than on track. A {blocked_days}-day integration delay prevents confirmation of the delivery date.",
                f"Work completed on the {initiative_area} {initiative_noun} stands at {percent_complete}%, but integration is still blocked after {blocked_days} days. Set a new delivery estimate only once that dependency is resolved.",
            ),
            (
                f"Calling the {percent_complete}%-complete {initiative_area} {initiative_noun} on track while integration has been blocked for {blocked_days} days misrepresents the actual status.",
                f"The opening claim about the {initiative_area} {initiative_noun} at {percent_complete}% and the separately stated blocker cannot both be accurate as written.",
                f"{blocked_days} days of blocked integration on the {percent_complete}%-complete {initiative_area} {initiative_noun} is inconsistent with describing it as on track.",
            ),
        ),
        "survey_report": (
            f"Most users prefer the {surveyed_qualifier} {surveyed_feature} because {survey_selected} of {survey_total} participants selected it, a {metric_delta}-point margin.",
            f"{survey_selected} responses in a {survey_total}-person sample about the {surveyed_qualifier} {surveyed_feature} (a {metric_delta}-point margin) do not establish a majority or a broader user preference",
            (
                f"{survey_selected} of {survey_total} participants selected the {surveyed_qualifier} {surveyed_feature}, a {metric_delta}-point margin. This sample does not establish a broader user preference.",
                f"In this {survey_total}-person sample, {survey_selected} participants chose the {surveyed_qualifier} {surveyed_feature} by a {metric_delta}-point margin. The result describes the sample rather than users generally.",
                f"The {surveyed_qualifier} {surveyed_feature} was selected by {survey_selected} of the {survey_total} respondents, with a {metric_delta}-point margin. Evidence about preference outside those respondents is not provided.",
                f"Report that {survey_selected} participants in the {survey_total}-person survey chose the {surveyed_qualifier} {surveyed_feature} by {metric_delta} points. Do not generalize that observation to the wider user population.",
            ),
            (
                f"{survey_selected} of {survey_total} choosing the {surveyed_qualifier} {surveyed_feature} by a {metric_delta}-point margin is not a majority on its own.",
                f"This {survey_total}-person sample about the {surveyed_qualifier} {surveyed_feature}, with its {metric_delta}-point margin, describes only those specific people.",
                f"A survey where {survey_selected} of {survey_total} chose the {surveyed_qualifier} {surveyed_feature} by {metric_delta} points cannot speak for users outside that sample.",
            ),
        ),
        "policy_notice": (
            f"Access to the {location_name} after {cutoff_hour}:00 is prohibited unless approved, and {exception_count} listed exceptions may be available.",
            f"the notice about the {location_name} gives no approval authority for its {exception_count} listed exceptions",
            (
                f"Access to the {location_name} after {cutoff_hour}:00 requires prior approval. Name the approving authority and the process for the {exception_count} listed exceptions before publishing the notice.",
                f"Prior approval is required to enter the {location_name} after {cutoff_hour}:00. Identify who grants approval and how each of the {exception_count} exceptions is requested before the notice is issued.",
                f"The {location_name} closes to unapproved access at {cutoff_hour}:00. Publication should wait until the responsible approver and procedure for the {exception_count} exceptions are stated.",
                f"State that after-hours entry to the {location_name} depends on advance approval after {cutoff_hour}:00. Add the authority and request path governing all {exception_count} exceptions.",
            ),
            (
                f"Without a named approver, no one can actually request or grant any of the {exception_count} exceptions to the {location_name}'s {cutoff_hour}:00 cutoff.",
                f"The notice about the {location_name} lacks a named enforcement contact for its {exception_count} listed exceptions, so it cannot be followed consistently.",
                f"The {location_name}'s {cutoff_hour}:00 cutoff and its {exception_count} exceptions cannot be granted by anyone until an approving authority is named.",
            ),
        ),
        "data_caption": (
            f"The results for the {caption_product} improved by {metric_delta}% after the {change_type}.",
            f"the caption citing a {metric_delta}% change for the {caption_product} names no metric, comparator, magnitude, or uncertainty",
            (
                f"The figure compares {caption_product} results before and after the {change_type}, showing a {metric_delta}% shift. Add the metric, magnitude, comparator, and uncertainty before claiming an improvement.",
                f"Results for the {caption_product} differ by {metric_delta}% before and after the {change_type}. The caption needs the measured quantity, comparison baseline, magnitude, and uncertainty before interpreting that difference as better.",
                f"The available figure shows a {metric_delta}% change in the {caption_product} after the {change_type}. Name the metric and comparator and report uncertainty before assigning a positive direction to the result.",
                f"Describe the {caption_product} figure as a {metric_delta}% before-and-after shift associated with the {change_type}. Withhold the improvement claim until the metric, baseline, magnitude, and uncertainty are specified.",
            ),
            (
                f"A {metric_delta}% change for the {caption_product} after the {change_type}, without a stated metric or magnitude, cannot be checked against the data.",
                f"The caption citing {metric_delta}% for the {caption_product} is missing its comparator, leaving the reader unable to judge what actually changed.",
                f"A {metric_delta}% claim about the {caption_product}'s {change_type} with no metric or magnitude cannot be verified.",
            ),
        ),
        "release_note": (
            f"This update to the {release_component} (version {release_ver}.0) fixes all synchronization problems across {tested_system_count} systems and works on every supported system.",
            f"the universal reliability and compatibility claims for {tested_system_count} tested systems exceed the stated evidence",
            (
                f"Version {release_ver}.0 fixes the {release_component} synchronization cases verified across {tested_system_count} tested systems. List the tested systems and retain any known limitations.",
                f"Tests on {tested_system_count} systems verified specific {release_component} synchronization fixes in version {release_ver}.0. Name those systems and document remaining limitations instead of claiming universal compatibility.",
                f"Release {release_ver}.0 resolved the tested {release_component} synchronization cases on {tested_system_count} systems. Its note should identify that test set and avoid extending the result to untested environments.",
                f"The evidence for version {release_ver}.0 covers {release_component} synchronization checks across {tested_system_count} named systems. Report those verified cases together with any known limits rather than saying every system works.",
            ),
            (
                f"Claiming every system is fixed in the {release_component} update extends well past the {tested_system_count} systems actually covered by the release tests.",
                f"Universal wording about the {release_component}, tested on only {tested_system_count} systems, invites an untested user to expect an unverified fix.",
                f"Version {release_ver}.0's {release_component} release tests cover {tested_system_count} specific systems, not the universal claim the note currently makes.",
            ),
        ),
        "support_macro": (
            f"We resolved the {support_issue} affecting {support_feature} for {affected_user_count} users. Please repeat the failed action to confirm that it now works.",
            f"the reply claims resolution for {affected_user_count} affected users before the requested verification is complete",
            (
                f"We applied a possible fix for the {support_issue} affecting {support_feature} for {affected_user_count} users. Please repeat the failed action so we can verify whether it is resolved.",
                f"A potential correction for the {support_issue} in {support_feature} has been applied for the {affected_user_count} affected users. Repeat the failed action and report the result before we mark the case resolved.",
                f"The change may address the {support_issue} affecting {support_feature} across {affected_user_count} users, but resolution is not confirmed. Rerun the failed action so its outcome can be checked.",
                f"We have made a candidate adjustment to {support_feature} for the {affected_user_count} users reporting the {support_issue}. Confirm its effect by repeating the action that previously failed.",
            ),
            (
                f"Declaring the {support_issue} affecting {support_feature} resolved for {affected_user_count} users overstates what has actually been confirmed so far.",
                f"The verification step exists precisely because the possible fix for {affected_user_count} affected users has not yet been confirmed to actually work.",
                f"The fix for {support_feature}, affecting {affected_user_count} users, is only possibly working until the requested verification confirms the {support_issue} are gone.",
            ),
        ),
        "risk_assessment": (
            f"The risk to the {risk_system}, with ${exposure_amount} of exposure, is low because only {incident_count} incidents occurred in the last {quiet_weeks} weeks.",
            f"{incident_count} incidents across {quiet_weeks} weeks against ${exposure_amount} of exposure for the {risk_system} do not establish low likelihood or low impact",
            (
                f"Only {incident_count} incidents were recorded for the {risk_system} in the last {quiet_weeks} weeks, against ${exposure_amount} of exposure. Assess likelihood, impact, exposure, and mitigation evidence before assigning a risk level.",
                f"The {risk_system} recorded {incident_count} incidents across {quiet_weeks} weeks while ${exposure_amount} remained exposed. A risk rating requires evidence about likelihood, impact, and mitigation beyond that short count.",
                f"Over {quiet_weeks} weeks, the {risk_system} had {incident_count} recorded incidents and ${exposure_amount} of exposure. Evaluate the severity and probability of loss together with existing controls before classifying the risk.",
                f"The available record covers {incident_count} incidents, a {quiet_weeks}-week window, and ${exposure_amount} exposed on the {risk_system}. Keep the risk level open until likelihood, consequence, and mitigation evidence are assessed.",
            ),
            (
                f"{incident_count} incidents across {quiet_weeks} weeks against ${exposure_amount} of exposure for the {risk_system} describe only a short observation window, not the likelihood or impact of a rare event.",
                f"{incident_count} recorded incidents across {quiet_weeks} weeks against ${exposure_amount} of exposure on the {risk_system} is too little evidence to set a risk level.",
                f"A {quiet_weeks}-week window with {incident_count} incidents against ${exposure_amount} of exposure for the {risk_system} is too short to characterize the likelihood or impact of a rare event.",
            ),
        ),
    }
    return cases
