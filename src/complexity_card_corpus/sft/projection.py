from __future__ import annotations

import json

from ..english_morphology import correct_indefinite_articles
from ..training_cards import TrainingCards, deal_training_cards
from .language import (
    _card_sections,
    _final_assistant_target,
    _inline_sentence,
    _render_natural_instruction,
    _sentence,
    _stable_index,
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
    target = _apply_semantic_resolution(
        target,
        task=task,
        metadata=metadata,
        example_id=example_id,
    )
    return prompt, correct_indefinite_articles(target), cards


def _project_sft_conversation(
    messages: list[dict[str, str]],
    *,
    example_id: str,
    task: str,
    answer_json: str,
) -> tuple[list[dict[str, str]], TrainingCards]:
    """Preserve real dialogue turns while removing card-storage syntax.

    Two-message examples remain direct instructions. Synthetic four-message
    card hands are flattened unless clarification itself is the task. Earlier
    releases preserved every generated clarification turn, which taught the
    model to ask for an outcome and constraint even when both were supplied.
    Non-card conversations remain untouched.
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
    if len(messages) <= 2 or sections is None or task != "context_clarification":
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ], cards

    openings = (
        "I need help with this situation: {situation}\n\nHere is what I know: {data}",
        "Here is the situation I am dealing with: {situation}\n\nThe available information is: {data}",
        "Can you help me think through this? {situation}\n\nThese are the relevant facts: {data}",
        "This is the current context: {situation}\n\nWhat I have confirmed so far: {data}",
    )
    acknowledgements = (
        "I understand the situation: {situation} What outcome and constraint should guide the answer?",
        "Thanks, that gives me the factual context. What result do you want, and what limit should I preserve?",
        "I have the context and the available facts. Tell me the intended outcome and any condition I must keep.",
        "Understood. Before I answer, what should the response accomplish and which constraint is non-negotiable?",
    )
    grounded_acknowledgements = (
        (
            "I understand that this concerns {subject}, and that {inline_state}. "
            "Before I suggest a solution, what exact result do you want, and which "
            "limit must remain unchanged? I will keep the supplied facts separate "
            "from any assumption until then."
            " Once those are clear, I can give a clear, direct answer instead of a "
            "generic one."
        ),
        (
            "Thanks, I have the context for {subject}: {state} To keep the response "
            "grounded, tell me the intended outcome and the constraint I must "
            "preserve. I will not fill either gap by guessing."
            " With those two details, I can respond directly and precisely without "
            "broadening the request."
        ),
        (
            "The key condition for {subject} is clear: {state} What should I help "
            "you accomplish, and where should the answer stop? Until you confirm "
            "that boundary, I will treat the facts as context rather than permission "
            "to act."
            " That will let me answer the actual case carefully rather than a broader "
            "version of it."
        ),
        (
            "I have the current context for {subject}. {state} Tell me the desired "
            "outcome and the one boundary I should not cross. I will preserve the "
            "recorded facts while waiting for that clarification."
            " After that, I can give a concrete, bounded answer without silently "
            "changing the request."
        ),
    )
    updates = (
        "The result I need is this: {goal}\n\nPlease keep this constraint: {rule}",
        "Please {lower_goal}\n\nOne condition matters: {rule}",
        "My goal is the following: {goal}\n\nThe answer must respect this limit: {rule}",
        "Here is the outcome I want: {goal}\n\nUse this boundary: {rule}",
    )
    variant = _stable_index(f"multi-turn:{example_id}:{cards.dialogue_state}", 4)
    situation = sections["situation"]
    first_user = openings[variant].format(
        situation=situation,
        data=sections["data"],
    )
    subject = str(metadata.get("subject", "the request")).strip().rstrip(".")
    source_state = str(metadata.get("source_state", "")).strip().rstrip(".")
    if source_state:
        first_assistant = grounded_acknowledgements[variant].format(
            subject=subject,
            state=_sentence(source_state),
            inline_state=_inline_sentence(source_state).rstrip(".!?"),
        )
    else:
        first_assistant = acknowledgements[variant].format(situation=situation)
    goal = sections["goal"]
    second_user = updates[variant].format(
        goal=goal,
        lower_goal=goal[:1].lower() + goal[1:],
        rule=sections["rule"],
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
