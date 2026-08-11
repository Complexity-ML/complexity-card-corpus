from __future__ import annotations

from typing import Any


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
    }
    nested = template
    for placeholder, field in replacements.items():
        nested = nested.replace(placeholder, field)
    return nested


def casual_reservoir(
    topic: dict[str, Any],
    context: dict[str, Any],
    decks: dict[str, list[str]],
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return nested topic × context × conversational-function cells."""

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
    context_cells = {
        target: (context[f"{target}_addition"],)
        for target in _CONTEXT_FIELDS.values()
    }
    surface_cells = {
        stage: tuple(_nested_surface(template) for template in templates)
        for stage, templates in decks.items()
    }
    return {
        "topic": topic_cells,
        "context": context_cells,
        "surface": surface_cells,
    }
