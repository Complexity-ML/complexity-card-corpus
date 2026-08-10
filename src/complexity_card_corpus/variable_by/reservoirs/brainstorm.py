from __future__ import annotations

_BRAINSTORM_PILOT_SETTINGS: dict[str, tuple[str, ...]] = {
    "names": (
        "a library foyer", "a community hall", "a tool-lending desk",
        "a neighborhood workshop", "a residents' assembly", "a market stall",
    ),
    "lesson_activity": (
        "mixed-ability classrooms", "small table groups", "after-school sessions",
        "adult-learning workshops", "peer-learning circles", "library study groups",
    ),
    "event_plan": (
        "the accessible entrance area", "seated activity tables", "the quiet room",
        "the main community hall", "step-free breakout spaces", "the welcome desk",
    ),
    "feature_ideas": (
        "an opt-in user cohort", "a staging workspace", "one delivery team",
        "a support-team sandbox", "a limited beta group", "a training environment",
    ),
    "writing_prompts": (
        "a beginner workshop", "an online writing circle", "a library session",
        "a peer-review group", "an evening class", "a weekend drafting session",
    ),
    "low_cost_activity": (
        "a community room", "a classroom", "a library meeting space",
        "a shared workshop", "a youth-club room", "an indoor common area",
    ),
    "outreach": (
        "library notice boards", "school newsletters", "community calendars",
        "partner-group bulletins", "public demonstration tables", "local notice stands",
    ),
    "workflow": (
        "one delivery team", "a support rotation", "an editorial group",
        "a project squad", "an operations team", "a review committee",
    ),
}

_BRAINSTORM_PILOT_SIGNALS: dict[str, tuple[str, ...]] = {
    "names": (
        "unprompted name recall", "correct pronunciation", "first-choice preference",
        "perceived welcome", "clarity of purpose", "word-of-mouth recall",
    ),
    "lesson_activity": (
        "accurate transfer answers", "completed evidence links", "explained reasoning",
        "corrected misconceptions", "peer explanations", "independent solutions",
    ),
    "event_plan": (
        "steady participation", "step-free movement", "quiet-space use",
        "completed activities", "balanced group flow", "material availability",
    ),
    "feature_ideas": (
        "missed handoffs", "approval waiting time", "reopened incidents",
        "incomplete evidence packs", "manual follow-up effort", "review turnaround",
    ),
    "writing_prompts": (
        "completed first drafts", "distinct story premises", "voluntary revisions",
        "peer-readable openings", "sustained writing time", "clear narrative choices",
    ),
    "low_cost_activity": (
        "successful completion", "material reuse", "balanced participation",
        "setup effort", "shared problem solving", "clear group outcomes",
    ),
    "outreach": (
        "walk-in attendance", "questions at the venue", "notice-board reach",
        "partner referrals", "repeat attendance", "public-session participation",
    ),
    "workflow": (
        "queue age", "avoidable rework", "approval turnaround",
        "missing evidence", "handoff delays", "operator interventions",
    ),
}


def brainstorm_pilot_cards(domain: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return compatible pilot settings and success signals."""

    return _BRAINSTORM_PILOT_SETTINGS[domain], _BRAINSTORM_PILOT_SIGNALS[domain]
