from __future__ import annotations

from typing import Any

from .casual_semantics import (
    CASUAL_BALANCED_CLOSINGS,
    CASUAL_COMPACT_CLOSINGS,
    CASUAL_CONTEXT_CLOSINGS,
    CASUAL_SEMANTIC_CLOSINGS,
)


_TOPIC_FIELDS = (
    "opening",
    "acknowledgement",
    "question",
    "detail",
    "reply",
    "follow_up_question",
    "shift",
    "closing",
)
_CONTEXT_FIELDS = {
    "context_opening": "opening",
    "context_detail": "detail",
    "context_shift": "shift",
    "context_closing": "closing",
}

_TOPIC_COMPACT = {
    "casual:morning:quiet_start": "the quiet morning",
    "casual:cooking:improvised_soup": "the improvised soup",
    "casual:music:late_album": "the rediscovered album",
    "casual:reading:slow_novel": "the slowly read novel",
    "casual:film:comfort_rewatch": "the familiar film",
    "casual:walking:unplanned_route": "the unplanned walk",
    "casual:learning:language_phrase": "the new phrase",
    "casual:garden:balcony_herbs": "the balcony herbs",
    "casual:pet:window_watch": "the window-watching pet",
    "casual:travel:station_memory": "the station memory",
    "casual:photography:reflections": "the reflected photograph",
    "casual:exercise:evening_stretch": "the evening stretch",
    "casual:tea:afternoon_pot": "the afternoon tea",
    "casual:neighborhood:bakery_line": "the bakery queue",
    "casual:home:desk_reset": "the cleared desk",
    "casual:sleep:late_reading": "the late reading",
    "casual:writing:character_voice": "the character voice",
    "casual:games:cooperative_puzzle": "the shared puzzle",
    "casual:technology:phone_distance": "the phone-free distance",
    "casual:weather:rainy_window": "the rainy window",
    "casual:skill:small_repair": "the small repair",
}

_CONTEXT_COMPACT = {
    "context:curious": "open to curiosity",
    "context:light_exchange": "light",
    "context:recent_change": "experimental",
    "context:thinking_aloud": "unsettled",
    "context:first_impression": "provisional",
    "context:returning_topic": "worth revisiting",
    "context:personal_preference": "personal",
    "context:small_discovery": "simply noticed",
    "context:gentle_habit": "pressure-free",
    "context:compare_past": "open to change",
    "context:shared_later": "easy to share",
    "context:process_interest": "focused on process",
    "context:no_rush": "unhurried",
    "context:practical_note": "practical and enjoyable",
    "context:unexpected_pleasure": "pleasantly surprising",
    "context:one_detail": "centered on one detail",
    "context:easy_followup": "open to another question",
    "context:honest_reaction": "honest rather than polished",
    "context:quiet_mood": "quietly remembered",
    "context:open_ending": "open-ended",
}

_CONTEXT_CLOSING_VARIANTS = {
    "context:no_rush": (
        "Nothing else has to follow immediately.",
        "There is no need for an immediate next step.",
        "The thought can rest without leading anywhere yet.",
        "No quick conclusion or action has to come next.",
    ),
}


def _lower_first(value: str) -> str:
    value = value.strip()
    if value.startswith(("I ", "I'm ", "I've ", "I'd ")):
        return value
    return value[:1].lower() + value[1:] if value else value


def _clause(value: str) -> str:
    return value.strip().rstrip(".!?")


def _nested_surface(template: str) -> str:
    replacements = {
        **{
            f"{{{field}}}": f"{{topic[{field}]}}"
            for field in _TOPIC_FIELDS
        },
        **{
            f"{{{field}_lower}}": f"{{topic[{field}_lower]}}"
            for field in _TOPIC_FIELDS
        },
        **{
            f"{{{source}}}": f"{{context[{target}]}}"
            for source, target in _CONTEXT_FIELDS.items()
        },
        "{context_closing}": "{semantic[context_closing]}",
    }
    nested = template
    for placeholder, field in replacements.items():
        nested = nested.replace(placeholder, field)
    return nested


def casual_reservoir(
    topic: dict[str, Any],
    context: dict[str, Any],
    intent: dict[str, Any],
    arc: dict[str, Any],
    decks: dict[str, list[str]],
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return nested topic × context × intent × arc conversation cells."""

    subcards = topic["subcards"]
    topic_cells = {
        field: (subcards[field],)
        for field in _TOPIC_FIELDS
    }
    topic_cells.update(
        {
            f"{field}_lower": (_lower_first(subcards[field]),)
            for field in _TOPIC_FIELDS
        }
    )
    topic_cells["closing_clause"] = (_clause(subcards["closing"]),)
    topic_cells["compact"] = (_TOPIC_COMPACT[topic["topic_id"]],)
    context_closings = _CONTEXT_CLOSING_VARIANTS.get(
        context["context_id"],
        (context["closing_addition"],),
    )
    context_cells = {
        target: (
            context_closings
            if target == "closing"
            else (context[f"{target}_addition"],)
        )
        for target in _CONTEXT_FIELDS.values()
    }
    context_cells["compact"] = (_CONTEXT_COMPACT[context["context_id"]],)
    context_cells["closing_lower"] = tuple(_lower_first(value) for value in context_closings)
    context_cells["closing_clause"] = tuple(_clause(value) for value in context_closings)
    context_cells["closing_clause_lower"] = tuple(
        _lower_first(_clause(value)) for value in context_closings
    )
    intent_cells = {
        "user_opening": tuple(intent["user_opening"]),
        "assistant_entry": tuple(intent["assistant_entry"]),
        "closing_focus": tuple(intent["closing_focus"]),
    }
    arc_cells = {
        "user_follow_up": tuple(arc["user_follow_up"]),
        "user_shift": tuple(arc["user_shift"]),
        "closing_lens": tuple(arc["closing_lens"]),
    }
    stage_additions = {
        "user_opening": " {intent[user_opening]}",
        "assistant_entry": " {intent[assistant_entry]}",
        "user_follow_up": " {arc[user_follow_up]}",
        "user_shift": " {arc[user_shift]}",
        "assistant_closing": " {semantic[closing]}",
    }
    surface_cells = {}
    for stage, templates in decks.items():
        addition = stage_additions.get(stage, "")
        if stage == "user_shift":
            templates = [
                template.replace(
                    "I had not phrased it that way, but {shift_lower} {context_shift}",
                    "That gives me another way to put it. {shift} {context_shift}",
                ).replace(
                    "I think so. {shift} {context_shift}",
                    "That helps me answer more directly. {shift} {context_shift}",
                )
                for template in templates
            ]
        surface_cells[stage] = tuple(
            _nested_surface(template) + addition for template in templates
        )
    return {
        "topic": topic_cells,
        "context": context_cells,
        "intent": intent_cells,
        "arc": arc_cells,
        "semantic": {
            "closing": CASUAL_SEMANTIC_CLOSINGS,
            "closing_compact": CASUAL_COMPACT_CLOSINGS,
            "closing_balanced": CASUAL_BALANCED_CLOSINGS,
            "context_closing": CASUAL_CONTEXT_CLOSINGS,
        },
        "surface": surface_cells,
    }
