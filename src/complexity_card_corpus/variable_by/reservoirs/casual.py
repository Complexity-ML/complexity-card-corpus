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
    context_cells = {
        target: (context[f"{target}_addition"],)
        for target in _CONTEXT_FIELDS.values()
    }
    context_cells["closing_lower"] = (
        _lower_first(context["closing_addition"]),
    )
    context_cells["closing_clause"] = (
        _clause(context["closing_addition"]),
    )
    context_cells["closing_clause_lower"] = (
        _lower_first(_clause(context["closing_addition"])),
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
