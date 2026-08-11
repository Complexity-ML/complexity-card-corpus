from __future__ import annotations

from typing import Any

from .core import (
    TaskHand,
    _code,
    _compose_subcards,
    _number,
    _pick,
    _render_domain,
)
from ..variable_by import (
    brainstorming_variable_by,
    critique_variable_by,
    reasoning_variable_by,
)
from ..variable_by.templates import CRITIQUE_TEMPLATES, REASONING_TEMPLATES
from ..variable_by.brainstorm_templates import (
    BRAINSTORM_GOAL_TEMPLATES,
    _BRAINSTORM_SCALE_CLOSINGS,
)
from ..variable_by.reservoirs import (
    BrainstormFacts,
    CritiqueFacts,
    brainstorm_cases,
    brainstorm_checks,
    brainstorm_pilot_cards,
    critique_cases,
    reasoning_case,
)


def _reasoning(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    domain = _render_domain(row)
    ranges = {
        "shopping_arithmetic": ((2, 80), (4, 250), (3, 95)),
        "schedule_math": ((2, 48), (10, 180), (5, 90)),
        "unit_conversion": ((10, 999), (3, 99), (2, 99)),
        "proportions": ((2, 120), (2, 80), (2, 99)),
        "table_comparison": ((10, 999), (10, 999), (10, 999)),
        "sequence_pattern": ((10, 999), (2, 120), (2, 99)),
        "logical_constraints": ((2, 60), (2, 60), (2, 60)),
        "simple_probability": ((10, 999), (10, 999), (2, 99)),
        "work_allocation": ((10, 999), (10, 999), (10, 999)),
    }
    unit_range, each_range, extra_range = ranges[domain]
    units = _number(f"units:{code}", *unit_range)
    each = _number(f"each:{code}", *each_range)
    extra = _number(f"extra:{code}", *extra_range)
    data, equation, total, check, components = reasoning_case(
        domain,
        code,
        units,
        each,
        extra,
        number=_number,
    )
    answer_variables = reasoning_variable_by(
        equation=equation,
        total=total,
        check=check,
        quantity_roles=components,
        domain=domain,
        code=code,
        data=data,
    )
    data = _compose_subcards(
        row,
        variant,
        "reasoning-data",
        (REASONING_TEMPLATES["data"],),
        pool_names=("problem",),
        variable_by=answer_variables,
    )
    goal = _compose_subcards(
        row,
        variant,
        "reasoning-goal",
        (REASONING_TEMPLATES["goal"],),
        pool_names=("calculation_instruction",),
        variable_by=answer_variables,
    )
    answer = _compose_subcards(
        row,
        variant,
        "reasoning-answer",
        (
            REASONING_TEMPLATES["calculation"],
            REASONING_TEMPLATES["verification"],
        ),
        pool_names=("calculation", "verification"),
        variable_by=answer_variables,
    )
    subject = domain.replace("_", " ").title()
    situation = _compose_subcards(
        row,
        variant,
        "reasoning-situation",
        (REASONING_TEMPLATES["situation"],),
        pool_names=("calculation_context",),
        variable_by=answer_variables,
    )
    return TaskHand(
        data,
        goal,
        answer,
        ("equation", "result", "check"),
        situation_title=f"{subject} — calculate and verify",
        situation=situation,
    )


# Reusable vocabulary reservoirs for _critique, grouped like a dictionary of
# card decks so a new word or a wider numeric range can be added in one place
# instead of hunting through the case bodies below. Add entries here, not as
# anonymous inline tuples, to keep the family easy to enrich later.
_CRITIQUE_WORD_RESERVOIRS: dict[str, tuple[str, ...]] = {
    "doc_type": ("report", "spreadsheet", "summary", "audit", "proposal"),
    "doc_qualifier": ("quarterly", "final", "draft", "revised", "updated"),
    "feature_qualifier": ("new", "updated", "redesigned", "streamlined", "automated"),
    "feature_name": (
        "workflow", "checkout flow", "onboarding process", "search pipeline", "sync engine",
    ),
    "product_area": ("payments", "search", "onboarding", "notifications", "analytics"),
    "artifact_type": ("prototype", "pilot", "beta", "proof of concept"),
    "key_bits": ("128", "192", "256"),
    "data_category": ("payment", "medical", "HR", "authentication", "financial"),
    "data_noun": ("records", "files", "logs", "forms"),
    "system_area": ("accounting", "inventory", "scheduling", "customer", "reporting"),
    "system_noun": ("app", "system", "portal", "dashboard", "tool"),
    "team_name": ("platform", "growth", "mobile", "data", "security"),
    "team_noun": ("team", "group", "squad"),
    "venue_qualifier": ("internal", "public", "beta", "official"),
    "venue_noun": ("forum", "review site", "feedback channel", "community board"),
    "action_name": ("upload", "sync", "payment", "export", "login"),
    "initiative_area": ("payments", "search", "onboarding", "checkout", "notifications"),
    "initiative_noun": ("migration", "relaunch", "redesign", "revamp"),
    "surveyed_qualifier": ("new", "redesigned", "updated", "simplified"),
    "surveyed_feature": (
        "navigation", "homepage", "pricing page", "checkout flow", "search bar",
    ),
    "location_name": (
        "building", "VPN", "office network", "shared workspace", "data center",
    ),
    "change_type": (
        "interface redesign", "pricing update", "onboarding flow", "checkout process",
    ),
    "caption_product": ("mobile app", "desktop client", "API", "website", "dashboard"),
    "release_component": (
        "sync engine", "notification service", "billing pipeline", "search index", "cache layer",
    ),
    "support_feature": ("login", "checkout", "file upload", "export", "password reset"),
    "support_issue": (
        "intermittent failures", "slow response times", "occasional errors", "unexpected timeouts",
    ),
    "risk_system": (
        "payment processing", "data pipeline", "authentication service",
        "customer database", "search infrastructure",
    ),
}

_CRITIQUE_COUNT_RESERVOIRS: dict[str, tuple[int, int]] = {
    "files": (2, 5),
    "arg_success": (2, 4),
    "arg_gap": (1, 3),
    "plan_days": (5, 14),
    "update_minutes": (5, 30),
    "topics": (3, 8),
    "comments": (2, 5),
    "error_code": (400, 599),
    "blocked_days": (2, 10),
    "survey_total": (10, 16),
    "cutoff_hour": (17, 22),
    "release_ver": (2, 9),
    "quiet_weeks": (2, 6),
    "page_count": (3, 48),
    "budget_amount": (5_000, 95_000),
    "record_count": (100, 9_999),
    "attendee_count": (4, 22),
    "view_count": (200, 9_800),
    "retry_limit": (1, 9),
    "percent_complete": (10, 95),
    "exception_count": (2, 40),
    "metric_delta": (3, 68),
    "tested_system_count": (2, 24),
    "affected_user_count": (50, 4_800),
    "incident_count": (0, 3),
    "exposure_amount": (5_000, 250_000),
}


def _critique(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)

    def word(name: str) -> str:
        return _pick(f"critique-{name}:{code}", _CRITIQUE_WORD_RESERVOIRS[name])

    def count(name: str) -> int:
        low, high = _CRITIQUE_COUNT_RESERVOIRS[name]
        return _number(f"critique-{name}:{code}", low, high)

    file_count = count("files")
    doc_type = word("doc_type")
    doc_qualifier = word("doc_qualifier")
    arg_success = count("arg_success")
    arg_total = arg_success + count("arg_gap")
    feature_qualifier = word("feature_qualifier")
    feature_name = word("feature_name")
    plan_days = count("plan_days")
    product_area = word("product_area")
    artifact_type = word("artifact_type")
    key_bits = word("key_bits")
    data_category = word("data_category")
    data_noun = word("data_noun")
    update_minutes = count("update_minutes")
    system_area = word("system_area")
    system_noun = word("system_noun")
    topic_count = count("topics")
    team_name = word("team_name")
    team_noun = word("team_noun")
    comment_count = count("comments")
    venue_qualifier = word("venue_qualifier")
    venue_noun = word("venue_noun")
    error_code = count("error_code")
    action_name = word("action_name")
    blocked_days = count("blocked_days")
    initiative_area = word("initiative_area")
    initiative_noun = word("initiative_noun")
    survey_total = count("survey_total")
    survey_selected = survey_total // 2
    surveyed_qualifier = word("surveyed_qualifier")
    surveyed_feature = word("surveyed_feature")
    cutoff_hour = count("cutoff_hour")
    location_name = word("location_name")
    change_type = word("change_type")
    caption_product = word("caption_product")
    release_ver = count("release_ver")
    release_component = word("release_component")
    support_feature = word("support_feature")
    support_issue = word("support_issue")
    quiet_weeks = count("quiet_weeks")
    risk_system = word("risk_system")

    page_count = count("page_count")
    budget_amount = count("budget_amount")
    record_count = count("record_count")
    attendee_count = count("attendee_count")
    view_count = count("view_count")
    retry_limit = count("retry_limit")
    percent_complete = count("percent_complete")
    exception_count = count("exception_count")
    metric_delta = count("metric_delta")
    tested_system_count = count("tested_system_count")
    affected_user_count = count("affected_user_count")
    incident_count = count("incident_count")
    exposure_amount = count("exposure_amount")

    cases = critique_cases(
        CritiqueFacts(
            action_name=action_name,
            affected_user_count=affected_user_count,
            arg_success=arg_success,
            arg_total=arg_total,
            artifact_type=artifact_type,
            attendee_count=attendee_count,
            blocked_days=blocked_days,
            budget_amount=budget_amount,
            caption_product=caption_product,
            change_type=change_type,
            comment_count=comment_count,
            cutoff_hour=cutoff_hour,
            data_category=data_category,
            data_noun=data_noun,
            doc_qualifier=doc_qualifier,
            doc_type=doc_type,
            error_code=error_code,
            exception_count=exception_count,
            exposure_amount=exposure_amount,
            feature_name=feature_name,
            feature_qualifier=feature_qualifier,
            file_count=file_count,
            incident_count=incident_count,
            initiative_area=initiative_area,
            initiative_noun=initiative_noun,
            key_bits=key_bits,
            location_name=location_name,
            metric_delta=metric_delta,
            page_count=page_count,
            percent_complete=percent_complete,
            plan_days=plan_days,
            product_area=product_area,
            quiet_weeks=quiet_weeks,
            record_count=record_count,
            release_component=release_component,
            release_ver=release_ver,
            retry_limit=retry_limit,
            risk_system=risk_system,
            support_feature=support_feature,
            support_issue=support_issue,
            survey_selected=survey_selected,
            survey_total=survey_total,
            surveyed_feature=surveyed_feature,
            surveyed_qualifier=surveyed_qualifier,
            system_area=system_area,
            system_noun=system_noun,
            team_name=team_name,
            team_noun=team_noun,
            tested_system_count=tested_system_count,
            topic_count=topic_count,
            update_minutes=update_minutes,
            venue_noun=venue_noun,
            venue_qualifier=venue_qualifier,
            view_count=view_count,
        )
    )
    draft, weakness, revision, consequence_pool = cases[_render_domain(row)]
    draft = f"Draft {code}: {draft}"
    critique_variables = critique_variable_by(
        code,
        weakness=weakness,
        revision=revision,
        consequences=consequence_pool,
    )
    data = _compose_subcards(
        row,
        variant,
        "critique-data",
        ((
            f"Text to review: {draft}",
            f"Review candidate {code}: {draft}",
            f"Editing input — {draft}",
        ),),
        pool_names=("draft",),
    )
    goal = _compose_subcards(
        row,
        variant,
        "critique-goal",
        (CRITIQUE_TEMPLATES["goal"],),
        pool_names=("critique_instruction",),
        variable_by=critique_variables,
    )
    answer = _compose_subcards(
        row,
        variant,
        "critique-answer",
        (CRITIQUE_TEMPLATES["answer"],),
        pool_names=("critique_response",),
        variable_by=critique_variables,
    )
    return TaskHand(data, goal, answer, ("weakness", "reason", "revision"))



_BRAINSTORM_SCALE_RANGES: dict[str, tuple[int, int]] = {
    "names": (100, 999),
    "lesson_activity": (20, 90),
    "event_plan": (20, 30),
    "feature_ideas": (100, 999),
    "writing_prompts": (20, 99),
    "low_cost_activity": (6, 10),
    "outreach": (100, 999),
    "workflow": (20, 99),
}

_BRAINSTORM_DAY_RANGES: dict[str, tuple[int, int]] = {
    "names": (14, 90),
    "lesson_activity": (7, 60),
    "event_plan": (30, 180),
    "feature_ideas": (14, 90),
    "writing_prompts": (7, 60),
    "low_cost_activity": (7, 60),
    "outreach": (7, 60),
    "workflow": (14, 90),
}



def _brainstorm(row: dict[str, Any], variant: int) -> TaskHand:
    code = _code(row)
    domain = _render_domain(row)
    domain_label = domain.replace("_", " ")
    name_audience = _pick(
        f"brainstorm-nameaudience:{code}",
        (
            "adult residents", "new neighbors", "local volunteers",
            "first-time visitors", "multilingual households", "community members",
        ),
    )
    name_quality = _pick(
        f"brainstorm-namequality:{code}",
        (
            "welcoming", "easy to pronounce", "memorable",
            "inclusive", "plain-language", "clear in conversation",
        ),
    )
    name_word_limit = _number(f"brainstorm-namewords:{code}", 2, 4)
    lesson_minutes = _number(f"brainstorm-lessonminutes:{code}", 25, 90)
    lesson_learners = _number(f"brainstorm-lessonlearners:{code}", 12, 48)
    lesson_material = _pick(
        f"brainstorm-lessonmaterial:{code}",
        (
            "paper strips", "index cards", "printed diagrams",
            "folded worksheets", "sticky notes", "cardstock tiles",
        ),
    )
    event_minutes = _number(f"brainstorm-eventminutes:{code}", 90, 240)
    event_attendees = _number(f"brainstorm-eventattendees:{code}", 20, 60)
    event_budget = _number(f"brainstorm-eventbudget:{code}", 100, 999)
    event_groups = _number(f"brainstorm-eventgroups:{code}", 3, 6)
    event_group_size = (event_attendees + event_groups - 1) // event_groups
    feature_team_size = _number(f"brainstorm-featureteam:{code}", 10, 99)
    feature_approval_steps = _number(f"brainstorm-approvalsteps:{code}", 2, 12)
    feature_review_hours = _number(f"brainstorm-reviewhours:{code}", 12, 96)
    prompt_word_limit = _number(f"brainstorm-promptwords:{code}", 100, 999)
    prompt_draft_minutes = _number(f"brainstorm-draftminutes:{code}", 20, 180)
    prompt_audience = _pick(
        f"brainstorm-promptaudience:{code}",
        (
            "adult beginners", "first-time fiction writers", "community writers",
            "returning learners", "library workshop participants", "peer-writing groups",
        ),
    )
    activity_minutes = _number(f"brainstorm-activityminutes:{code}", 30, 180)
    activity_participants = _number(f"brainstorm-activitypeople:{code}", 6, 40)
    activity_material = _pick(
        f"brainstorm-activitymaterial:{code}",
        (
            "recycled paper", "index cards", "sticky notes",
            "paper strips", "cardboard pieces", "plain worksheets",
        ),
    )
    outreach_capacity = _number(f"brainstorm-outreachcapacity:{code}", 100, 999)
    outreach_days = _number(f"brainstorm-outreachdays:{code}", 7, 90)
    outreach_partner = _pick(
        f"brainstorm-outreachpartner:{code}",
        (
            "public libraries", "local schools", "community centers",
            "repair groups", "adult-learning programs", "neighborhood associations",
        ),
    )
    workflow_items = _number(f"brainstorm-workitems:{code}", 100, 999)
    workflow_reviewers = _number(f"brainstorm-reviewers:{code}", 3, 20)
    workflow_target_hours = _number(f"brainstorm-targethours:{code}", 12, 96)
    cases = brainstorm_cases(
        BrainstormFacts(
            activity_material=activity_material,
            activity_minutes=activity_minutes,
            activity_participants=activity_participants,
            event_attendees=event_attendees,
            event_budget=event_budget,
            event_group_size=event_group_size,
            event_groups=event_groups,
            event_minutes=event_minutes,
            feature_approval_steps=feature_approval_steps,
            feature_review_hours=feature_review_hours,
            feature_team_size=feature_team_size,
            lesson_learners=lesson_learners,
            lesson_material=lesson_material,
            lesson_minutes=lesson_minutes,
            name_audience=name_audience,
            name_quality=name_quality,
            name_word_limit=name_word_limit,
            outreach_capacity=outreach_capacity,
            outreach_days=outreach_days,
            outreach_partner=outreach_partner,
            prompt_audience=prompt_audience,
            prompt_draft_minutes=prompt_draft_minutes,
            prompt_word_limit=prompt_word_limit,
            workflow_items=workflow_items,
            workflow_reviewers=workflow_reviewers,
            workflow_target_hours=workflow_target_hours,
        )
    )
    case_cards = cases[domain]
    brief, answer = case_cards[
        _number(f"brainstorm-case:{row['scenario_id']}", 0, len(case_cards) - 1)
    ]
    (
        constraint_checks,
        outcome_checks,
        default_constraint,
        default_outcome,
    ) = brainstorm_checks(domain_label)
    constraint_check = constraint_checks.get(
        row.get("constraint", ""),
        default_constraint,
    )
    outcome_check = outcome_checks.get(
        row.get("desired_outcome", ""),
        default_outcome,
    )
    options, selection = answer.rsplit(" Select ", 1)
    scale_low, scale_high = _BRAINSTORM_SCALE_RANGES[domain]
    day_low, day_high = _BRAINSTORM_DAY_RANGES[domain]
    generated_scale = _number(f"brainstorm-scale:{code}", scale_low, scale_high)
    scale_count = {
        "lesson_activity": lesson_learners,
        "event_plan": event_attendees,
        "feature_ideas": feature_team_size,
        "low_cost_activity": activity_participants,
        "outreach": outreach_capacity,
        "workflow": workflow_reviewers,
    }.get(domain, generated_scale)
    days_to_test = _number(f"brainstorm-testdays:{code}", day_low, day_high)
    pilot_rounds = _number(f"brainstorm-rounds:{code}", 3, 12)
    pilot_settings, pilot_signals = brainstorm_pilot_cards(domain)
    pilot_setting = _pick(f"brainstorm-setting:{code}", pilot_settings)
    pilot_signal = _pick(f"brainstorm-signal:{code}", pilot_signals)
    closing_templates = _BRAINSTORM_SCALE_CLOSINGS[domain]
    lexical_variables = brainstorming_variable_by(
        domain,
        scale=scale_count,
        days=days_to_test,
        rounds=pilot_rounds,
        setting=pilot_setting,
        signal=pilot_signal,
    )
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
                f"Select {selection} " + closing_templates[0],
                f"Select this option: {selection} " + closing_templates[1],
                f"Select the strongest fit: {selection} " + closing_templates[2],
            ),
        ),
        pool_names=("options", "criteria", "outcome", "selection"),
        variable_by=lexical_variables,
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
            BRAINSTORM_GOAL_TEMPLATES["request"],
            BRAINSTORM_GOAL_TEMPLATES["decision"],
        ),
        pool_names=("request", "decision"),
        variable_by=lexical_variables,
    )
    return TaskHand(data, goal, answer, ("three_options", "criteria", "selection"))
