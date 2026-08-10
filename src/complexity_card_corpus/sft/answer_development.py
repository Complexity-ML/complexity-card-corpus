from __future__ import annotations

from typing import Any


def development_card_count(task: str) -> int:
    """Return zero: generic answer-padding cards are intentionally disabled.

    Answer depth must be authored by the task-specific semantic decks.  A
    shared closing deck previously taught the model to discuss the response
    itself ("the supported takeaway", "the supplied material", and similar
    phrases) instead of completing the user's task.
    """

    del task
    return 0


def develop_answer(
    target: str,
    *,
    task: str,
    metadata: dict[str, Any],
    example_id: str,
) -> str:
    """Preserve an already-authored answer without generic padding.

    The parameters remain in the public helper signature for compatibility
    with existing builders.  Task-specific generators are responsible for
    adding explanations, checks, or next actions when those are semantically
    required.
    """

    del task, metadata, example_id
    return target
