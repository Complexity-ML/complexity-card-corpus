from __future__ import annotations


def summary_decision_surfaces(decision: str) -> tuple[str, ...]:
    """Return natural model-facing openings for one supported decision."""

    return (
        f"The decision is to {decision}", f"They agreed to {decision}",
        f"Proceed by choosing to {decision}", f"The selected direction is to {decision}",
        decision, f"The record supports a decision to {decision}",
        f"The agreed course is to {decision}", f"Participants chose to {decision}",
        f"The retained outcome is to {decision}", f"The documented choice is to {decision}",
        f"Agreement centers on the step to {decision}", f"The notes establish a plan to {decision}",
        f"The group settled on the option to {decision}", f"The resulting decision calls for the team to {decision}",
        f"The supported path forward is to {decision}", f"The meeting outcome is to {decision}",
        f"The source records a commitment to {decision}", f"The chosen course requires the group to {decision}",
        f"The recorded resolution is to {decision}", f"The final supported direction is to {decision}",
        f"The agreed result has the team {decision}", f"The documented preference became a choice to {decision}",
        f"The accountable outcome is to {decision}", f"The source-backed conclusion is to {decision}",
    )


def summary_case_variants(
    domain: str,
    *,
    default_decision: str,
    default_action: str,
    default_open_point: str,
    contrast_ratio: int,
    sample_count: int,
    case_count: int,
    test_coverage: int,
    employee_count: int,
    citation_count: int,
    downtime_minutes: int,
    example_count: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Expand one summary fact triple without changing its supported meaning."""

    alternatives = {
        "meeting_transcript": (
            (
                f"accept new settings-page wording once it satisfies the {contrast_ratio}:1 contrast target",
                f"use the redesigned settings copy subject to confirming contrast at {contrast_ratio}:1",
                f"approve the settings wording only after its measured contrast reaches {contrast_ratio}:1",
                f"adopt the revised interface copy when accessibility review verifies a {contrast_ratio}:1 ratio",
            ),
            (
                f"verify screen-reader navigation and measure color contrast against the {contrast_ratio}:1 target",
                f"complete both the screen-reader flow test and the {contrast_ratio}:1 visual-contrast check",
                f"test keyboard and screen-reader flow, then confirm the interface reaches {contrast_ratio}:1 contrast",
                f"run the assistive-navigation check and record whether visual contrast meets {contrast_ratio}:1",
                f"validate the screen-reader path separately from the color measurement at the {contrast_ratio}:1 threshold",
                f"have accessibility review cover spoken navigation plus a measured {contrast_ratio}:1 contrast result",
                f"check that assistive technology can traverse the settings page and that its colors clear {contrast_ratio}:1",
                f"document both navigation with a screen reader and compliance with the {contrast_ratio}:1 visual target",
            ),
            (
                f"when compliant wording will ship and which rollout group will receive it first after the {contrast_ratio}:1 check",
                f"the deployment sequence and confirmed launch date following accessibility approval at {contrast_ratio}:1",
                f"which users receive the accessible copy first and when rollout starts after the {contrast_ratio}:1 result",
                "the launch order and release timing that remain unset after accessibility verification",
            ),
        ),
        "research_notes": (
            (
                f"keep the initial trial's measured thermal result in the {sample_count}-run record",
                f"preserve the temperature observation obtained in run one of the {sample_count} trials",
                f"retain the measured heating result as an observation across the {sample_count}-trial record",
                f"record the first run's thermal change without extending it beyond the {sample_count} tested samples",
            ),
            (
                f"repeat measurement of peak heat and cooling speed throughout the {sample_count}-sample set",
                f"check both maximum temperature and subsequent cooling rate over all {sample_count} samples",
                f"measure peak temperature separately from cooldown behavior in each of the {sample_count} runs",
                f"repeat the heat-rise and cooling observations throughout the full {sample_count}-sample series",
            ),
            (
                f"what mechanism caused the thermal pattern seen in the {sample_count} observations",
                f"the supported cause, if any, behind the temperature effect recorded across {sample_count} samples",
                f"whether the {sample_count} observations establish a mechanism rather than only a thermal pattern",
                "the causal explanation that the recorded temperature measurements do not yet supply",
            ),
        ),
        "support_thread": (
            (
                f"leave this case unresolved while diagnosis continues across {case_count} related reports",
                f"maintain the open status until the issue shared by {case_count} cases is diagnosed",
                f"keep all {case_count} related recovery reports open while the shared failure is investigated",
                f"avoid closing the case until diagnosis accounts for the pattern across {case_count} reports",
            ),
            (
                f"compare email reset with device verification throughout the {case_count} recovery cases",
                f"run both recovery checks—reset by email and verification by device—against the {case_count} reports",
                f"compare the email-reset path with device verification across each of the {case_count} cases",
                f"test both recovery methods against the evidence collected in the {case_count}-case group",
            ),
            (
                f"whether a single device category accounts for the problem across the {case_count} reports",
                f"the unresolved scope of the issue among device types represented in {case_count} related cases",
                f"whether failures across the {case_count} reports cluster around one kind of device",
                "the device-level boundary of the problem, which the current support evidence has not established",
            ),
        ),
        "project_update": (
            (
                f"treat the prototype as complete at {test_coverage}% coverage while reserving acceptance of the final integrations",
                f"retain the finished prototype after reaching {test_coverage}% coverage, subject to two integration rounds",
                f"recognize prototype completion at {test_coverage}% coverage while keeping integration acceptance separate",
                f"preserve the completed prototype result but defer release approval until both integrations pass",
            ),
            (
                f"validate payment processing and notification delivery after the suite reaches {test_coverage}% coverage",
                f"complete the gateway and notification integration tests against the {test_coverage}%-covered prototype",
                f"run separate payment and notification checks on the prototype with {test_coverage}% test coverage",
                "verify the two outstanding integrations before treating the prototype as release-ready",
            ),
            (
                f"when the {test_coverage}%-covered prototype will receive a confirmed public-release date",
                "the launch timing for wider availability after the remaining integration work is complete",
                f"when a prototype at {test_coverage}% coverage can move from completed build to public release",
                "the public launch date, which remains unset pending the final integration evidence",
            ),
        ),
        "policy_memo": (
            (
                f"approve the updated rule governing late access to the workspace used by {employee_count} employees",
                f"put the revised out-of-hours entry policy in place for the {employee_count}-person site",
                f"adopt the updated after-hours workspace rule covering all {employee_count} employees",
                f"approve the new access requirements for late entry at the {employee_count}-employee workplace",
            ),
            (
                f"record how emergency entry and preapproved contractors are handled for the {employee_count}-employee workspace",
                "define both exception paths—emergencies and approved contractors—under the shared-workspace rule",
                f"document emergency and authorized-contractor exceptions for the site serving {employee_count} employees",
                "write down how the revised rule handles urgent access and contractors approved in advance",
            ),
            (
                f"when enforcement of the new access policy begins for the {employee_count}-person workforce",
                "the effective date on which the revised after-hours requirements become enforceable",
                f"the first day on which the {employee_count} employees are expected to follow the revised access rule",
                "whether approval takes effect immediately or after a separate implementation notice",
                "the still-unconfirmed start date for applying the updated out-of-hours policy",
                f"when the approved workspace rule becomes operational across the {employee_count}-person site",
                "the transition date between the existing access practice and the newly approved one",
                "which communicated date will activate enforcement of the revised entry requirements",
            ),
        ),
        "article_excerpt": (
            (
                f"keep the main reported pattern while preserving its basis in {citation_count} cited sources",
                f"carry forward the excerpt's core observation and its {citation_count}-citation support",
                f"retain the reported pattern while tying it explicitly to the {citation_count} sources in the excerpt",
                f"preserve the observed result without claiming support beyond its {citation_count} cited references",
            ),
            (
                f"check the pilot study and later survey within the set of {citation_count} references",
                f"validate both named examples against their sources in the {citation_count}-item citation record",
                f"trace the pilot and follow-up survey separately through the {citation_count} listed references",
                "check that each example says only what its cited study or survey actually establishes",
            ),
            (
                "whether evidence outside the cited pilot and survey supports the observed pattern more broadly",
                f"the unresolved reach of the claim beyond the examples contained in {citation_count} citations",
                f"whether the pattern generalizes outside the studies represented by the {citation_count} references",
                "the breadth of support beyond the pilot and survey, which the excerpt leaves open",
            ),
        ),
        "incident_log": (
            (
                f"maintain monitored recovery for the service following its {downtime_minutes}-minute interruption",
                f"leave the restored service under observation after downtime lasting {downtime_minutes} minutes",
                f"continue supervised recovery after restoring service from the {downtime_minutes}-minute disruption",
                "keep the service available in a monitored state while post-incident checks continue",
                f"treat restoration after {downtime_minutes} minutes as provisional until recovery signals remain stable",
                "operate the recovered service under observation rather than declaring the incident fully closed",
                f"preserve service availability with active monitoring after the interruption lasted {downtime_minutes} minutes",
                "maintain the recovered path while telemetry is watched for any recurrence",
            ),
            (
                f"check the load balancer and caching layer as the two remaining leads from the {downtime_minutes}-minute outage",
                "investigate both unresolved components—the balancer and cache—after service restoration",
                f"compare the load-balancer and cache evidence collected after the {downtime_minutes}-minute interruption",
                "test the balancer hypothesis separately from the caching hypothesis during monitored recovery",
                f"review both infrastructure leads without treating either as the cause of the {downtime_minutes}-minute outage",
                "gather discriminating evidence for the two open component hypotheses while the service remains monitored",
                "assign separate follow-ups for load distribution and cached responses after restoration",
                f"preserve both component theories as unresolved until post-recovery checks explain the {downtime_minutes}-minute interruption",
            ),
            (
                f"which mechanism actually produced the service interruption lasting {downtime_minutes} minutes",
                "the verified root cause of the outage rather than the remaining component hypotheses",
                f"what ultimately caused the {downtime_minutes}-minute interruption, beyond the two current leads",
                "which of the infrastructure hypotheses, if either, is supported as the outage cause",
                "the causal finding that post-recovery evidence has not yet established",
                f"whether load distribution, caching, or another mechanism explains the {downtime_minutes}-minute outage",
                "the unresolved distinction between suspected components and a verified initiating cause",
                "what evidence will identify the actual failure mechanism instead of merely naming candidates",
            ),
        ),
        "learning_notes": (
            (
                f"keep the provisional rule definition supported by its check across {example_count} examples",
                f"preserve the working formulation after validation on the {example_count}-example set",
                f"retain the provisional definition that currently fits all {example_count} checked examples",
                f"keep the tested rule as provisional evidence rather than a universal result after {example_count} cases",
            ),
            (
                f"add one edge example and one counterexample to the existing {example_count} checks",
                f"test the rule beyond its {example_count} examples using both a boundary instance and a negative instance",
                f"add one edge case and one clear failure case to the existing set of {example_count} checks",
                "probe the working definition with examples chosen specifically to expose its limit",
            ),
            (
                f"which boundary marks the failure of the current rule beyond the {example_count} checked examples",
                "the still-unknown limit of applicability for the working definition",
                f"where the rule stops working outside the {example_count} examples already examined",
                "the boundary condition that would separate valid use of the definition from a counterexample",
            ),
        ),
    }
    decision_alternatives, action_alternatives, open_alternatives = alternatives[domain]
    return (
        (default_decision, *decision_alternatives),
        (default_action, *action_alternatives),
        (default_open_point, *open_alternatives),
    )


def meeting_summary_cards(
    contrast_ratio: int,
    *,
    default_decision: str,
    default_open_point: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return compatible decision and open-point variants for meeting summaries."""

    decisions = (
        default_decision,
        f"use the revised settings-page wording once it meets the {contrast_ratio}:1 contrast target",
        f"move forward with the settings-page copy revision subject to a {contrast_ratio}:1 contrast result",
        f"accept the settings-page wording change after confirming its {contrast_ratio}:1 contrast ratio",
        f"adopt the new settings copy when accessibility testing verifies contrast of at least {contrast_ratio}:1",
        f"approve the rewritten settings text conditional on clearing the {contrast_ratio}:1 visual threshold",
        f"retain the proposed wording for release after its measured contrast reaches {contrast_ratio}:1",
        f"proceed with the copy update only when the final accessibility record confirms {contrast_ratio}:1 contrast",
    )
    open_points = (
        default_open_point,
        f"when the copy cleared at {contrast_ratio}:1 will be released and in which rollout sequence",
        f"the release timing and deployment order after the {contrast_ratio}:1 accessibility check",
        f"which rollout stage will receive the {contrast_ratio}:1-compliant copy first and on what date",
        f"the date for publishing the revised copy and the order in which audiences receive the {contrast_ratio}:1 version",
        f"how deployment will be sequenced after accessibility confirms the {contrast_ratio}:1 target",
        f"which group receives the approved settings wording first and when that release begins",
        f"the launch calendar and staged distribution plan following the final contrast check at {contrast_ratio}:1",
    )
    return decisions, open_points


def summary_answer_cards(
    *,
    decision: str,
    action: str,
    open_point: str,
    owner: str,
    day: int,
) -> tuple[tuple[str, ...], ...]:
    """Return deep, compatible decision/action/open-point response decks."""

    decisions = (
        f"Decision: {decision}.",
        f"Decision: the record is to {decision}.",
        f"Decision: proceed by choosing to {decision}.",
        f"Decision: the agreed direction is to {decision}.",
        f"Decision: the documented outcome is to {decision}.",
        f"Decision: participants settled on the choice to {decision}.",
        f"Decision: the source records agreement to {decision}.",
        f"Decision: the selected course is to {decision}.",
        f"Decision: the final direction in the record is to {decision}.",
        f"Decision: the group resolved to {decision}.",
        f"Decision: the supported choice is to {decision}.",
        f"Decision: the recorded conclusion calls for the team to {decision}.",
        f"Decision: the notes establish an intention to {decision}.",
        f"Decision: the agreed result requires the group to {decision}.",
        f"Decision: the confirmed path is to {decision}.",
        f"Decision: the source identifies the chosen direction as the need to {decision}.",
        f"Decision: agreement was reached to {decision}.",
        f"Decision: the record supports moving ahead to {decision}.",
        f"Decision: the chosen outcome is that the team will {decision}.",
        f"Decision: the meeting concluded with a commitment to {decision}.",
        f"Decision: the stated resolution is to {decision}.",
        f"Decision: the documented preference became a decision to {decision}.",
        f"Decision: the accountable direction is to {decision}.",
        f"Decision: the retained conclusion is to {decision}.",
    )
    actions = (
        f"Action: due day {day}, {owner} will {action}.",
        f"Action: no later than day {day}, {owner} will {action}.",
        f"Action: {action}, owned by {owner}, closing out on day {day}, once confirmed.",
        f"Action: {owner} is assigned to {action}; day {day} is the outside limit.",
        f"Action: {owner} owns the work to {action}; day {day} is the deadline.",
        f"Action: the step belongs to {owner}: {action}. Its deadline is day {day}.",
        f"Action: the timed assignment gives {owner} until day {day} to {action}.",
        f"Action: responsibility sits with {owner}, who will {action}; the final date is day {day}.",
        f"Action: {owner} will complete the work to {action}, with day {day} as the deadline.",
        f"Action: the record assigns {action} to {owner} for completion before the day {day} cutoff.",
        f"Action: ownership belongs to {owner}; the required step is due on or before day {day}.",
        f"Action: day {day} is the due date for {owner} to {action}.",
        f"Action: {owner} takes the follow-up to {action} and must finish before day {day} closes.",
        f"Action: before the day {day} cutoff, {owner} is to {action}.",
        f"Action: {action} is {owner}'s assigned work, due on day {day}.",
        f"Action: {owner}'s owned next move is to {action}, to be finished by the close of day {day}.",
        f"Action: {owner} carries responsibility for {action} through day {day}.",
        f"Action: completion no later than day {day} requires {owner} to {action}.",
        f"Action: the notes assign both the work and its due date — {owner} will {action} by day {day}.",
        f"Action: {owner}'s deadline is day {day} for the task to {action}.",
        f"Action: the follow-up remains with {owner} until {action} is completed, with day {day} as the deadline.",
        f"Action: {owner} is accountable for {action} within the period ending on day {day}.",
        f"Action: by the close of day {day}, {owner} must {action}.",
        f"Action: the assigned owner is {owner}, with completion of {action} required on day {day}.",
    )
    open_points = (
        f"Open point: {open_point} remains unresolved.",
        f"Open point: nothing in the source resolves {open_point}.",
        f"Open point: {open_point} is still unresolved.",
        f"Open point: no resolution is recorded for {open_point}.",
        f"Open point: the source does not establish {open_point}.",
        f"Open point: the record leaves {open_point} unanswered.",
        f"Open point: available notes do not settle {open_point}.",
        f"Open point: no documented conclusion addresses {open_point}.",
        f"Open point: the evidence remains silent on {open_point}.",
        f"Open point: {open_point} stays outside the recorded decision.",
        f"Open point: the summary must preserve uncertainty about {open_point}.",
        f"Open point: the material supplies no answer regarding {open_point}.",
        f"Open point: the record contains no basis for resolving {open_point}.",
        f"Open point: {open_point} remains an explicit unknown.",
        f"Open point: no source detail closes the question of {open_point}.",
        f"Open point: the documented outcome does not determine {open_point}.",
        f"Open point: uncertainty continues around {open_point}.",
        f"Open point: the notes stop short of answering {open_point}.",
        f"Open point: the supplied record cannot resolve {open_point}.",
        f"Open point: {open_point} was not decided in the source.",
        f"Open point: the available evidence leaves {open_point} open.",
        f"Open point: no recorded fact determines {open_point}.",
        f"Open point: the question of {open_point} has no documented answer.",
        f"Open point: preserve {open_point} as unresolved rather than inferring an answer.",
    )
    return decisions, actions, open_points
