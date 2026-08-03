from __future__ import annotations

import json

from ..english_morphology import correct_indefinite_articles
from ..training_cards import TrainingCards, deal_training_cards
from .dialogue_links import (
    preserve_linked_dialogue,
    render_dialogue_link,
    render_dialogue_opening,
)
from .language import (
    _card_sections,
    _final_assistant_target,
    _render_natural_instruction,
    _without_internal_card_labels,
)
from .target import _apply_semantic_resolution, _naturalize_assistant_target


def _project_sft_exchange(
    messages: list[dict[str, str]],
    *,
    example_id: str,
    task: str,
    answer_json: str,
) -> tuple[str, str, TrainingCards]:
    try:
        metadata = json.loads(answer_json) if answer_json else {}
    except json.JSONDecodeError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    cards = deal_training_cards(
        task=task,
        mode="chat" if len(messages) > 2 else "instruct",
        example_id=example_id,
        metadata=metadata,
    )
    prompt = _render_natural_instruction(messages, example_id, cards)
    if metadata.get("evaluation_source") == "separately_authored":
        target = _final_assistant_target(messages)
    else:
        target = _naturalize_assistant_target(
            messages,
            task=task,
            cards=cards,
            example_id=example_id,
        )
    if _card_sections(messages) is not None:
        target = _apply_semantic_resolution(
            target,
            task=task,
            metadata=metadata,
            example_id=example_id,
        )
    return (
        _without_internal_card_labels(prompt),
        correct_indefinite_articles(_without_internal_card_labels(target)),
        cards,
    )


def _project_sft_conversation(
    messages: list[dict[str, str]],
    *,
    example_id: str,
    task: str,
    answer_json: str,
) -> tuple[list[dict[str, str]], TrainingCards]:
    """Preserve real dialogue turns while removing card-storage syntax.

    Two-message examples remain direct instructions. A deterministic minority
    of four-message card hands become real linked dialogues: the first user
    supplies situation and data, the assistant makes one compatible
    clarification/objection/correction/follow-up/validation move, and the
    second user supplies goal and boundary. The other card hands become
    complete direct requests, while non-card conversations remain untouched.
    """

    prompt, target, cards = _project_sft_exchange(
        messages,
        example_id=example_id,
        task=task,
        answer_json=answer_json,
    )
    sections = _card_sections(messages)
    try:
        metadata = json.loads(answer_json) if answer_json else {}
    except json.JSONDecodeError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    if sections is None and len(messages) > 2:
        natural_messages = [
            {
                "role": message["role"],
                "content": correct_indefinite_articles(message["content"].strip()),
            }
            for message in messages
        ]
        natural_messages[-1]["content"] = target
        return natural_messages, cards
    if (
        len(messages) <= 2
        or sections is None
        or not preserve_linked_dialogue(
            example_id,
            natural_depth=cards.natural_depth,
        )
    ):
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ], cards

    situation = sections["situation"]
    first_user = render_dialogue_opening(
        example_id=example_id,
        situation=situation,
        data=sections["data"],
        task=task,
    )
    subject = str(metadata.get("subject", "the request")).strip().rstrip(".")
    source_state = str(metadata.get("source_state", "")).strip().rstrip(".")
    _move, first_assistant, second_user = render_dialogue_link(
        cards=cards,
        example_id=example_id,
        subject=subject,
        state=source_state,
        goal=sections["goal"],
        rule=sections["rule"],
        task=task,
    )
    return [
        {"role": "user", "content": correct_indefinite_articles(first_user)},
        {
            "role": "assistant",
            "content": correct_indefinite_articles(first_assistant),
        },
        {"role": "user", "content": correct_indefinite_articles(second_user)},
        {"role": "assistant", "content": target},
    ], cards
