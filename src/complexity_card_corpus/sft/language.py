from __future__ import annotations

import hashlib
import re

from ..english_morphology import correct_indefinite_articles
from ..training_cards import TrainingCards, deal_training_cards


_CARD_SECTION = re.compile(r"(?m)^(SITUATION|DATA|RULE|GOAL) CARD\s*$")


_HAND_PREFIX = re.compile(
    r"^(?:For\s+hand|Hand)\s+[A-Za-z0-9]+\s*(?:—|:|-)\s*",
    re.IGNORECASE,
)


def _stable_index(value: str, size: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % size


def _pick(value: str, choices: tuple[str, ...]) -> str:
    return choices[_stable_index(value, len(choices))]


def _merged_user_content(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        message["content"].strip() for message in messages if message["role"] == "user"
    )


def _card_sections(messages: list[dict[str, str]]) -> dict[str, str] | None:
    """Extract authored cards without exposing their storage labels."""

    merged = _merged_user_content(messages)
    parts = _CARD_SECTION.split(merged)
    sections = {
        parts[index].lower(): parts[index + 1].strip()
        for index in range(1, len(parts) - 1, 2)
    }
    required = {"situation", "data", "rule", "goal"}
    if not sections:
        return None
    if set(sections) != required:
        raise ValueError("SFT card hand must contain situation, data, rule, and goal")
    return sections


def _render_natural_instruction(
    messages: list[dict[str, str]],
    example_id: str,
    cards: TrainingCards | None = None,
) -> str:
    sections = _card_sections(messages)
    if sections is None:
        return _merged_user_content(messages)
    cards = cards or deal_training_cards(
        task="unknown",
        mode="chat" if len(messages) > 2 else "instruct",
        example_id=example_id,
    )
    goal = sections["goal"]
    lower_goal = goal[:1].lower() + goal[1:]
    values = {**sections, "lower_goal": lower_goal}
    full = {
        "direct": "{goal}\n\nContext: {situation}\n\nUse these facts: {data}\n\nRequirement: {rule}",
        "polite": "Could you help with this request? {goal}\n\n{situation}\n\nThe available information is: {data}\n\nPlease keep this limit: {rule}",
        "compact": "{goal}\n\n{data}\n\nKeep to this condition: {rule}",
        "context_first": "{situation}\n\nGiven this context, {lower_goal}\n\nRelevant information: {data}\n\n{rule}",
        "conversational": "I need help with this: {lower_goal}\n\nHere is what happened: {situation}\n\nWhat we know: {data}\n\nPlease keep in mind that {rule}",
        "follow_up": "Following up on the context below, {lower_goal}\n\n{situation}\n\nUse this information: {data}\n\nThe remaining condition is: {rule}",
        "plain": "{goal}\n\n{situation}\n\n{data}\n\n{rule}",
    }
    rendered = full[cards.surface].format(**values)
    if cards.dialogue_state == "correction":
        rendered = (
            f"One detail in the earlier request conflicts with the record. "
            f"{sections['situation']}\n\n"
            f"{goal}\n\nUse only this recorded information: {sections['data']}\n\n"
            f"Keep this condition: {sections['rule']} Do not assume the conflict "
            "is resolved unless the information says so."
        )
    elif cards.dialogue_state == "constraint_update" and cards.surface != "compact":
        rendered = (
            f"There is one additional constraint: {sections['rule']}\n\n"
            f"{goal}\n\n{sections['situation']}\n\n"
            f"Use this information: {sections['data']}"
        )
    elif (
        cards.dialogue_state == "clarification_resolved" and cards.surface != "compact"
    ):
        rendered = (
            f"The scope is now clear. {goal}\n\n{sections['situation']}\n\n"
            f"Relevant information: {sections['data']}\n\n"
            f"Keep this requirement: {sections['rule']}"
        )

    if cards.context_density == "minimal" and cards.uncertainty == "answerable":
        rendered = f"{goal}\n\n{sections['data']}"
    elif cards.context_density == "focused":
        rendered = f"{goal}\n\n{sections['data']}\n\n{sections['rule']}"
    if cards.noise == "secondary_detail":
        notes = (
            "The record identifier is included only for traceability.",
            "The order in which the facts were copied does not determine the answer.",
            "Formatting differences in the source do not change the stated values.",
            "An administrative label in the record is not part of the requested result.",
        )
        rendered += (
            "\n\n" + notes[_stable_index(f"noise-note:{example_id}", len(notes))]
        )
    return correct_indefinite_articles(rendered)


def _final_assistant_target(messages: list[dict[str, str]]) -> str:
    response = next(
        message["content"].strip()
        for message in reversed(messages)
        if message["role"] == "assistant"
    )
    return _HAND_PREFIX.sub("", response, count=1)


def _labelled_fields(text: str, labels: tuple[str, ...]) -> dict[str, str]:
    """Read authored completion fields without retaining their storage labels."""

    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(label) for label in labels) + r"):\s*"
    )
    matches = list(pattern.finditer(text))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields[match.group(1)] = text[match.end() : end].strip().rstrip(" .")
    return fields


def _sentence(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    value = value[:1].upper() + value[1:]
    if value[-1] not in ".!?":
        value += "."
    return value


def _inline_sentence(value: str) -> str:
    """Make a complete clause safe to insert after an inline lead-in."""

    value = _sentence(value)
    initial = re.match(r"[A-Za-z]+", value)
    if initial is not None:
        word = initial.group(0)
        # Preserve a likely proper-name subject ("Mina will ..."). Blindly
        # lowercasing every initial word changed names into ordinary nouns in
        # otherwise natural summaries.
        proper_name_subject = re.match(
            r"^[A-Z][a-z]+\s+(?:will|can|must|owns|contacts|supplies)\b",
            value,
        )
        if not (len(word) > 1 and word.isupper()) and proper_name_subject is None:
            value = value[:1].lower() + value[1:]
    return value


def _render_messages(messages: list[dict[str, str]]) -> str:
    labels = {"user": "User", "assistant": "Assistant"}
    return "\n".join(
        f"{labels[message['role']]}: {message['content']}" for message in messages
    )
