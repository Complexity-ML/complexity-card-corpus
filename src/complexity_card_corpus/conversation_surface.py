from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import ceil, gcd
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

from .build import file_sha256
from .instruct import INSTRUCTION_SCHEMA


DATASET_ID = "complexity-original-conversation-v1"
SURFACE_VERSION = "conversation-surface-v1"
SURFACE_LICENSE = "CC BY-NC 4.0"
SURFACE_SOURCE = "Complexity original authored conversation cards"
_VARIANT_RADIX = 32

_PLACEHOLDER = re.compile(r"\{[^{}]+\}")
_WORD = re.compile(r"[a-z0-9']+")
_LOWERCASE_I = re.compile(r"(?:^|[.!?]\s+)i(?:\s|['’])")


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _stable_index(value: str, size: int) -> int:
    if size < 1:
        raise ValueError("cannot select from an empty sequence")
    return int.from_bytes(_digest(value)[:8], "big") % size


def _split(example_id: str, validation_percent: int) -> str:
    return (
        "validation"
        if _stable_index(f"split:{example_id}", 100) < validation_percent
        else "train"
    )


def _lower_first(value: str) -> str:
    if re.match(r"^I(?:\s|['’])", value):
        return value
    return value[:1].lower() + value[1:] if value else value


def _recommendation_from_choice(value: str) -> str:
    prefix = "I will "
    recommendation = value[len(prefix) :] if value.startswith(prefix) else value
    return recommendation[:1].upper() + recommendation[1:]


def _render_messages(messages: list[dict[str, str]]) -> str:
    labels = {"user": "User", "assistant": "Assistant"}
    return "\n".join(
        f"{labels[message['role']]}: {message['content']}" for message in messages
    )


def _length_bucket(value: str) -> str:
    words = len(value.split())
    if words <= 4:
        return "very_short"
    if words <= 12:
        return "short"
    if words <= 30:
        return "medium"
    return "long"


def _position_variant_shift(attempt: int, position: int, total: int) -> int:
    if attempt == 0:
        return 0
    attempt -= 1
    if position == 0:
        return 1 + attempt % _VARIANT_RADIX
    if position == total - 1:
        return 1 + (attempt // _VARIANT_RADIX) % _VARIANT_RADIX
    return 1 + (attempt + position * 7) % _VARIANT_RADIX


def _default_variant_shift(
    blueprint: dict[str, Any], card: dict[str, str], position: int
) -> int:
    return 1 + _stable_index(
        f"default-surface:{blueprint['blueprint_id']}:{card['card_id']}:{position}",
        _VARIANT_RADIX,
    )


def _render_candidates(
    templates: tuple[str, ...],
    frames: tuple[str, ...],
    values: dict[str, str],
) -> list[str]:
    base_candidates = [template.format(**values).strip() for template in templates]
    candidates: list[str] = []
    for base_index, base in enumerate(base_candidates):
        selected_frames = frames if base_index == 0 else ("{}",)
        for frame in selected_frames:
            value = base if frame == "{}" else frame.format(_lower_first(base))
            if value not in candidates:
                candidates.append(value)
    return candidates


def _choose_variant(
    templates: tuple[str, ...],
    *,
    frames: tuple[str, ...],
    rank: int,
    stage: str,
    style: str,
    target_length: str,
    values: dict[str, str],
    variant_shift: int = 0,
) -> str:
    candidates = _render_candidates(templates, frames, values)
    exact = [value for value in candidates if _length_bucket(value) == target_length]
    if not exact:
        order = {"very_short": 0, "short": 1, "medium": 2, "long": 3}
        target = order[target_length]
        distance = min(abs(order[_length_bucket(value)] - target) for value in candidates)
        exact = [
            value
            for value in candidates
            if abs(order[_length_bucket(value)] - target) == distance
        ]
    if style in {"concise_practical", "concise_empathetic"}:
        exact.sort(key=lambda value: (len(value.split()), value))
    elif style == "stepwise_helpful":
        exact.sort(key=lambda value: (-len(value.split()), value))
    else:
        exact.sort(
            key=lambda value: _digest(
                f"style-order:{style}:{stage}:{values['card_id']}:{value}"
            )
        )
    offset = _stable_index(f"{stage}:{style}:{values['card_id']}", len(exact))
    strides = [
        value
        for value in range(1, len(exact) + 1)
        if gcd(value, len(exact)) == 1
    ]
    stride = strides[_stable_index(f"surface-stride:{stage}", len(strides))]
    return exact[(rank + offset + variant_shift * stride) % len(exact)]


_TASK_TEMPLATES: dict[str, tuple[str, ...]] = {
    "state_goal": (
        "{goal}",
        "Can you help me think this through? {goal}",
        "I want to make a practical choice. {goal}",
        "I could use a clear plan here: {goal_lower}",
        "Please help me narrow this down. {goal}",
        "I am trying to decide what to do next. {goal}",
        "Here is what I need help with: {goal_lower}",
        "Could we work through one decision? {goal}",
        "Could you help me choose a sensible next step? {goal}",
        "I would like a second opinion before I act. {goal}",
    ),
    "acknowledge_goal": (
        "{acknowledgement}",
        "Understood. {acknowledgement}",
        "That is a clear goal. {acknowledgement}",
        "That gives us a clear scope. {acknowledgement}",
        "Yes. {acknowledgement}",
        "Let us make the trade-off explicit. {acknowledgement}",
        "I follow the decision you are trying to make. {acknowledgement}",
        "We can approach this without guessing. {acknowledgement}",
    ),
    "provide_detail": (
        "{detail}",
        "One useful detail is this: {detail_lower}",
        "For context, {detail_lower}",
        "The main constraint is that {detail_lower}",
        "There is one more thing to account for: {detail_lower}",
        "The relevant background is this: {detail_lower}",
        "A constraint that may change the answer is that {detail_lower}",
        "Here is the detail I do not want to lose: {detail_lower}",
    ),
    "ask_for_missing_detail": (
        "{question}",
        "Before comparing the options, {question_lower}",
        "One detail would change the recommendation: {question_lower}",
        "To avoid guessing, {question_lower}",
        "I need one point clarified. {question}",
        "A reliable answer depends on one fact. {question}",
        "Let us check the missing condition first. {question}",
        "The safest comparison starts with this question: {question_lower}",
    ),
    "choose_option": (
        "{priority}",
        "My priority is simple: {priority_lower}",
        "The deciding factor for me is that {priority_lower}",
        "I would rather optimize for this: {priority_lower}",
        "What matters most is that {priority_lower}",
        "I can make the trade-off if we preserve this: {priority_lower}",
        "My preference should follow one rule: {priority_lower}",
        "The outcome I care about is clear: {priority_lower}",
    ),
    "present_bounded_options": (
        "Two sensible options are {option_a} or {option_b}.",
        "I would compare only two paths: {option_a}, and {option_b}.",
        "The practical shortlist is {option_a} versus {option_b}.",
        "You can keep the decision bounded: choose between {option_a} and {option_b}.",
        "Given that priority, the realistic choices are {option_a} or {option_b}.",
        "That leaves a short comparison: {option_a}, or instead {option_b}.",
        "There is no need for a long list; compare {option_a} with {option_b}.",
        "On those terms, compare {option_a} directly with {option_b}.",
    ),
    "confirm_choice": (
        "{choice}",
        "I will go with this: {choice_lower}",
        "The first option fits better. {choice}",
        "That makes the decision easier. {choice}",
        "I have made the choice: {choice_lower}",
        "That settles it for me. {choice}",
        "I know which direction fits now. {choice}",
        "I am comfortable choosing on that basis. {choice}",
    ),
    "confirm_next_step": (
        "{next_step}",
        "Good. {next_step}",
        "That is a workable decision. {next_step}",
        "The next step is now clear. {next_step}",
        "You have a bounded plan. {next_step}",
        "You can act on that choice without adding more options. {next_step}",
        "That completes the decision. {next_step}",
        "The plan is specific enough to use. {next_step}",
    ),
    "give_direct_next_step": (
        "{next_step}",
        "A practical first step is simple: {next_step_lower}",
        "Start here: {next_step_lower}",
        "The immediate action is to {next_step_lower}",
        "For now, {next_step_lower}",
        "Keep it concrete: {next_step_lower}",
        "The most useful first move is to {next_step_lower}",
        "Without adding more options, {next_step_lower}",
    ),
    "request_written_confirmation": (
        "Before committing, ask the provider to confirm this in writing: {question}",
        "Get one written answer before proceeding: {question}",
        "Ask for a written reply to the decisive question: {question}",
        "Keep the commitment conditional until this is answered in writing: {question}",
        "The useful written confirmation is a direct answer to this: {question}",
        "Request written confirmation rather than relying on the listing: {question}",
        "Use one written check before booking or paying: {question}",
        "Have the provider answer this in writing before you commit: {question}",
    ),
    "give_recorded_reason": (
        "{recommendation} Record the stated priority and the constraint behind it.",
        "Use this bounded choice: {recommendation_lower} Write down which requirement it satisfies.",
        "The recommendation is specific: {recommendation_lower} Keep a short note of the deciding fact.",
        "Choose on the available evidence: {recommendation_lower} Record what would change the decision.",
        "A reviewable decision is possible: {recommendation_lower} Note the priority and supporting fact.",
        "Make the choice explicit: {recommendation_lower} Then record the reason in one sentence.",
        "Keep the rationale attached to the action: {recommendation_lower} Note the decisive condition.",
        "Use the stated priority as the record: {recommendation_lower} Write down the decisive constraint.",
    ),
    "give_choice_and_action": (
        "{recommendation} {next_step}",
        "The bounded choice is this: {recommendation_lower} Then {next_step_lower}",
        "Make one decision and act on it: {recommendation_lower} Next, {next_step_lower}",
        "Choose a single path: {recommendation_lower} The immediate action is to {next_step_lower}",
        "The recommendation is to {recommendation_lower} Follow it by taking this step: {next_step_lower}",
        "Use this choice: {recommendation_lower} Then complete the concrete next step: {next_step_lower}",
        "Keep the result specific: {recommendation_lower} After that, {next_step_lower}",
        "Finish with this choice: {recommendation_lower} Act now by doing the following: {next_step_lower}",
    ),
    "offer_safe_alternative": (
        "{safe_alternative}",
        "A safer fallback is available: {safe_alternative_lower}",
        "I cannot confirm the preferred option from here. As a fallback, {safe_alternative_lower}",
        "If the preferred path is unavailable, {safe_alternative_lower}",
        "To avoid overpromising, use this fallback: {safe_alternative_lower}",
        "A cautious alternative would be to {safe_alternative_lower}",
        "There is a practical fallback that avoids an unsupported promise: {safe_alternative_lower}",
        "When the preferred option cannot be confirmed, {safe_alternative_lower}",
    ),
}


_EMPATHY_TEMPLATES: dict[str, tuple[str, ...]] = {
    "share_situation": (
        "{situation}",
        "I have been trying to put this into words. {situation}",
        "I could use a calm perspective. {situation}",
        "Something has been sitting with me. {situation}",
        "I want to talk through something ordinary but important. {situation}",
        "I am still making sense of a feeling. {situation}",
        "This has been on my mind today. {situation}",
        "I would like to say this out loud. {situation}",
    ),
    "acknowledge_emotion": (
        "{acknowledgement}",
        "That reaction makes sense. {acknowledgement}",
        "I can see why this stayed with you. {acknowledgement}",
        "It sounds like this matters a lot. {acknowledgement}",
        "There is a real reason this affected you. {acknowledgement}",
        "Your response fits what happened. {acknowledgement}",
        "That carries more weight than it may seem from outside. {acknowledgement}",
        "I hear how strongly this landed. {acknowledgement}",
    ),
    "expand_feeling": (
        "{detail}",
        "The part I keep returning to is this: {detail_lower}",
        "What makes it harder is that {detail_lower}",
        "I think the feeling is stronger because {detail_lower}",
        "There is another layer to it. {detail}",
        "I notice one detail keeps intensifying it: {detail_lower}",
        "The feeling is tied to something specific. {detail}",
        "What stays with me is that {detail_lower}",
    ),
    "invite_detail_without_assumption": (
        "{question}",
        "If you want to unpack it, {question_lower}",
        "We do not have to assume the answer. {question}",
        "It may help to name the important part. {question}",
        "You can decide how much to explore. {question}",
        "A useful distinction might be available here. {question}",
        "We can stay close to what you actually know. {question}",
        "One gentle question may clarify the feeling. {question}",
    ),
    "reflect_on_need": (
        "{need}",
        "I think what I need is straightforward: {need_lower}",
        "Putting it plainly, {need_lower}",
        "The useful outcome for me would be this: {need_lower}",
        "I can name the need more clearly now. {need}",
        "What I am asking from this moment is simple: {need_lower}",
        "There is a practical need under the feeling. {need}",
        "I do not need a complete solution; {need_lower}",
    ),
    "offer_grounded_support": (
        "{support}",
        "A small, grounded step could help. {support}",
        "You do not need to solve everything at once. {support}",
        "Keeping it manageable may be best. {support}",
        "The response can stay small and concrete. {support}",
        "A measured next step is enough for now. {support}",
        "You can support yourself without forcing a quick resolution. {support}",
        "It may help to work with what is within reach. {support}",
    ),
    "follow_up": (
        "{follow_up}",
        "That feels manageable. {follow_up}",
        "I can start there. {follow_up}",
        "That gives me something concrete to try. {follow_up}",
        "I know where to begin now. {follow_up}",
        "That is specific enough to use. {follow_up}",
        "I can carry that into the next moment. {follow_up}",
        "The next step feels clearer. {follow_up}",
    ),
    "close_supportively": (
        "{closing}",
        "That sounds like a thoughtful next step. {closing}",
        "You have given this careful thought. {closing}",
        "There is no need to rush the rest. {closing}",
        "That is a kind and realistic place to stop. {closing}",
        "You can let this understanding settle. {closing}",
        "Nothing more has to be decided right now. {closing}",
        "The next step can remain modest. {closing}",
    ),
}


_TASK_FRAMES: dict[str, tuple[str, ...]] = {
    "state_goal": (
        "{}",
        "For now, {}",
        "In practical terms, {}",
        "Before I decide, {}",
    ),
    "acknowledge_goal": (
        "{}",
        "In that case, {}",
        "Keeping the scope narrow, {}",
        "On that basis, {}",
    ),
    "provide_detail": (
        "{}",
        "More specifically, {}",
        "The context is simple: {}",
        "One point matters here: {}",
    ),
    "ask_for_missing_detail": (
        "{}",
        "To narrow it down, {}",
        "Before moving on, {}",
        "The deciding question is this: {}",
    ),
    "choose_option": (
        "{}",
        "For the final choice, {}",
        "Looking at the trade-off, {}",
        "To keep the decision clear, {}",
    ),
    "present_bounded_options": (
        "{}",
        "Within that boundary, {}",
        "For a short comparison, {}",
        "Keeping only realistic choices, {}",
    ),
    "confirm_choice": (
        "{}",
        "With that comparison made, {}",
        "The choice is now specific: {}",
        "Based on the stated priority, {}",
    ),
    "confirm_next_step": (
        "{}",
        "From here, {}",
        "To complete the plan, {}",
        "For the immediate next step, {}",
    ),
    "give_direct_next_step": (
        "{}",
        "As a first step, {}",
        "To make progress, {}",
        "For one concrete action, {}",
    ),
    "request_written_confirmation": (
        "{}",
        "Before any commitment, {}",
        "For a verifiable answer, {}",
        "To keep a written record, {}",
    ),
    "give_recorded_reason": (
        "{}",
        "For a reviewable decision, {}",
        "To preserve the rationale, {}",
        "For a short decision record, {}",
    ),
    "give_choice_and_action": (
        "{}",
        "To finish the decision, {}",
        "For one choice and one action, {}",
        "Keeping the result concrete, {}",
    ),
    "offer_safe_alternative": (
        "{}",
        "If confirmation is unavailable, {}",
        "To keep the fallback bounded, {}",
        "Without making an unsupported assumption, {}",
    ),
}


_EMPATHY_FRAMES: dict[str, tuple[str, ...]] = {
    "share_situation": (
        "{}",
        "Right now, {}",
        "Said plainly, {}",
        "What I notice is this: {}",
    ),
    "acknowledge_emotion": (
        "{}",
        "Without rushing the feeling, {}",
        "Seen in context, {}",
        "Staying close to what happened, {}",
    ),
    "expand_feeling": (
        "{}",
        "Looking at it more closely, {}",
        "There is room for one more detail: {}",
        "As the feeling settles, {}",
    ),
    "invite_detail_without_assumption": (
        "{}",
        "Without deciding for you, {}",
        "If it feels useful, {}",
        "To understand rather than assume, {}",
    ),
    "reflect_on_need": (
        "{}",
        "Underneath the reaction, {}",
        "For the present moment, {}",
        "Keeping the need modest, {}",
    ),
    "offer_grounded_support": (
        "{}",
        "As a grounded response, {}",
        "Without trying to fix everything, {}",
        "For one manageable step, {}",
    ),
    "follow_up": (
        "{}",
        "With that in mind, {}",
        "For the next small step, {}",
        "Leaving room to adjust, {}",
    ),
    "close_supportively": (
        "{}",
        "For now, {}",
        "Without forcing a conclusion, {}",
        "As a place to pause, {}",
    ),
}


def _surface_target_length(
    blueprint: dict[str, Any],
    card: dict[str, str],
    position: int,
    *,
    kind: str,
) -> str:
    stage = blueprint["dialogue_stages"][position]
    templates = (
        _TASK_TEMPLATES[stage]
        if kind == "task_oriented"
        else _EMPATHY_TEMPLATES[stage]
    )
    frames = (
        _TASK_FRAMES[stage]
        if kind == "task_oriented"
        else _EMPATHY_FRAMES[stage]
    )
    values = _task_values(card) if kind == "task_oriented" else _empathy_values(card)
    counts = Counter(
        _length_bucket(value)
        for value in _render_candidates(templates, frames, values)
    )
    preferred = blueprint["target_length_pattern"][position]
    return max(counts, key=lambda bucket: (counts[bucket], bucket == preferred))


def _surface_target_pattern(
    blueprint: dict[str, Any], card: dict[str, str], *, kind: str
) -> list[str]:
    return [
        _surface_target_length(blueprint, card, position, kind=kind)
        for position in range(len(blueprint["dialogue_stages"]))
    ]


def _join_sentences(base: str, addition: str) -> str:
    return f"{base.strip()} {addition.strip()}".strip()


def _expand_scenario_cards(
    groups: dict[str, list[dict[str, str]]],
    context_cards: list[dict[str, str]],
    *,
    kind: str,
) -> dict[str, list[dict[str, str]]]:
    if not context_cards:
        raise ValueError(f"missing original context cards for {kind}")
    expanded: dict[str, list[dict[str, str]]] = {}
    for category, cards in groups.items():
        resolved: list[dict[str, str]] = []
        for card in cards:
            for context in context_cards:
                item = dict(card)
                context_id = context["context_id"]
                item["card_id"] = f"{card['card_id']}:context:{context_id}"
                item["context_id"] = context_id
                if kind == "task_oriented":
                    item["goal"] = _join_sentences(
                        card["goal"], context["goal_addition"]
                    )
                    item["detail"] = _join_sentences(
                        card["detail"], context["detail_addition"]
                    )
                else:
                    item["situation"] = _join_sentences(
                        card["situation"], context["situation_addition"]
                    )
                    item["detail"] = _join_sentences(
                        card["detail"], context["detail_addition"]
                    )
                resolved.append(item)
        if len({card["card_id"] for card in resolved}) != len(resolved):
            raise ValueError(f"duplicate expanded scenario card IDs in {category}")
        expanded[category] = resolved
    return expanded


def _balanced_select(
    rows: list[dict[str, Any]],
    *,
    pilot_size: int,
    seed: int,
) -> list[tuple[dict[str, Any], int]]:
    if pilot_size % 2:
        raise ValueError("pilot_size must be even to balance the two corpus kinds")
    target_per_kind = pilot_size // 2
    by_kind: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_kind[row["corpus_kind"]][row["category"]].append(row)
    required = {"task_oriented", "empathetic_conversation"}
    if set(by_kind) != required:
        raise ValueError(f"expected exactly these blueprint kinds: {sorted(required)}")

    selected: list[tuple[dict[str, Any], int]] = []
    for kind in sorted(required):
        groups = by_kind[kind]
        categories = sorted(groups, key=lambda value: _digest(f"{seed}:{kind}:{value}"))
        base, remainder = divmod(target_per_kind, len(categories))
        for category_index, category in enumerate(categories):
            quota = base + int(category_index < remainder)
            candidates = sorted(
                groups[category],
                key=lambda row: _digest(f"{seed}:{row['blueprint_id']}"),
            )
            if len(candidates) < quota:
                raise ValueError(
                    f"category {category} has {len(candidates)} blueprints but needs {quota}"
                )
            selected.extend((row, rank) for rank, row in enumerate(candidates[:quota]))
    return selected


def _task_values(card: dict[str, str]) -> dict[str, str]:
    recommendation = _recommendation_from_choice(card["choice"])
    return {
        **card,
        "goal_lower": _lower_first(card["goal"]),
        "detail_lower": _lower_first(card["detail"]),
        "question_lower": _lower_first(card["question"]),
        "priority_lower": _lower_first(card["priority"]),
        "choice_lower": _lower_first(card["choice"]),
        "next_step_lower": _lower_first(card["next_step"]),
        "recommendation": recommendation,
        "recommendation_lower": _lower_first(recommendation),
        "safe_alternative_lower": _lower_first(card["safe_alternative"]),
    }


_TASK_CONTEXT_RESPONSE_STAGE = {
    "availability_change": "offer_safe_alternative",
    "before_contact": "present_bounded_options",
    "before_payment": "ask_for_missing_detail",
    "bounded_information": "ask_for_missing_detail",
    "change_if_unverified": "offer_safe_alternative",
    "choice_and_action": "give_choice_and_action",
    "confirmed_vs_unknown": "ask_for_missing_detail",
    "decidable_now": "give_direct_next_step",
    "direct": "give_direct_next_step",
    "fallback_needed": "offer_safe_alternative",
    "one_decisive_question": "ask_for_missing_detail",
    "preference_vs_requirement": "present_bounded_options",
    "priority_first": "present_bounded_options",
    "record_reason": "give_recorded_reason",
    "remote_decision": "ask_for_missing_detail",
    "reversible_first_step": "give_direct_next_step",
    "time_to_verify": "ask_for_missing_detail",
    "timing_and_constraint": "ask_for_missing_detail",
    "two_realistic_paths": "present_bounded_options",
    "written_confirmation": "request_written_confirmation",
}

_QUESTION_STAGES = {"ask_for_missing_detail", "request_written_confirmation"}


def _adapt_task_card(
    blueprint: dict[str, Any], card: dict[str, str]
) -> dict[str, str]:
    needs_priority_in_opening = card["context_id"] in {
        "choice_and_action",
        "priority_first",
        "record_reason",
    }
    if not needs_priority_in_opening or "choose_option" in blueprint["dialogue_stages"]:
        return card
    return {**card, "goal": _join_sentences(card["goal"], card["priority"])}


def _task_context_is_satisfied(stages: list[str], preferred: str) -> bool:
    if preferred == "give_choice_and_action":
        return (
            "give_choice_and_action" in stages
            or {"confirm_choice", "confirm_next_step"}.issubset(stages)
        )
    if preferred == "give_direct_next_step":
        return stages[-1] != "acknowledge_goal"
    if preferred in {"give_recorded_reason", "request_written_confirmation"}:
        return preferred in stages
    if preferred == "offer_safe_alternative":
        return (
            "offer_safe_alternative" in stages[1:]
            or "present_bounded_options" in stages[1:]
        )
    return preferred in stages[1:]


def _adapt_task_blueprint(
    blueprint: dict[str, Any], card: dict[str, str]
) -> dict[str, Any]:
    stages = list(blueprint["dialogue_stages"])
    context_id = card["context_id"]
    preferred = _TASK_CONTEXT_RESPONSE_STAGE[context_id]

    if not _task_context_is_satisfied(stages, preferred):
        stages[-1] = preferred

    return {
        **blueprint,
        "dialogue_stages": stages,
        "target_question_turns": sum(stage in _QUESTION_STAGES for stage in stages),
    }


def _task_message_at(
    blueprint: dict[str, Any],
    card: dict[str, str],
    phrase_rank: int,
    position: int,
    variant_shift: int = 0,
) -> str:
    stage = blueprint["dialogue_stages"][position]
    return _choose_variant(
        _TASK_TEMPLATES[stage],
        frames=_TASK_FRAMES[stage],
        rank=phrase_rank,
        stage=stage,
        style=blueprint["response_style"],
        target_length=_surface_target_length(
            blueprint, card, position, kind="task_oriented"
        ),
        values=_task_values(card),
        variant_shift=variant_shift,
    )


def _task_messages(
    blueprint: dict[str, Any],
    cards: list[dict[str, str]],
    rank: int,
    variant_shift: int = 0,
    selected_card: dict[str, str] | None = None,
) -> tuple[list[dict[str, str]], str]:
    card = selected_card or cards[rank % len(cards)]
    phrase_rank = rank // len(cards)
    messages = []
    for position in range(len(blueprint["dialogue_stages"])):
        shift = (
            _position_variant_shift(
                variant_shift, position, len(blueprint["dialogue_stages"])
            )
            if variant_shift
            else _default_variant_shift(blueprint, card, position)
        )
        text = _task_message_at(
            blueprint,
            card,
            phrase_rank,
            position,
            shift,
        )
        messages.append({"role": blueprint["target_speaker_pattern"][position], "content": text})
    return messages, card["card_id"]


def _empathy_values(card: dict[str, str]) -> dict[str, str]:
    return {
        **card,
        "detail_lower": _lower_first(card["detail"]),
        "question_lower": _lower_first(card["question"]),
        "need_lower": _lower_first(card["need"]),
    }


def _empathy_message_at(
    blueprint: dict[str, Any],
    card: dict[str, str],
    rank: int,
    position: int,
    variant_shift: int = 0,
) -> str:
    stage = blueprint["dialogue_stages"][position]
    return _choose_variant(
        _EMPATHY_TEMPLATES[stage],
        frames=_EMPATHY_FRAMES[stage],
        rank=rank,
        stage=stage,
        style=blueprint["response_style"],
        target_length=_surface_target_length(
            blueprint, card, position, kind="empathetic_conversation"
        ),
        values=_empathy_values(card),
        variant_shift=variant_shift,
    )


def _empathy_messages(
    blueprint: dict[str, Any],
    cards: list[dict[str, str]],
    rank: int,
    variant_shift: int = 0,
) -> tuple[list[dict[str, str]], str]:
    card = cards[rank % len(cards)]
    messages = []
    for position in range(len(blueprint["dialogue_stages"])):
        shift = (
            _position_variant_shift(
                variant_shift, position, len(blueprint["dialogue_stages"])
            )
            if variant_shift
            else _default_variant_shift(blueprint, card, position)
        )
        text = _empathy_message_at(
            blueprint,
            card,
            rank,
            position,
            shift,
        )
        messages.append({"role": blueprint["target_speaker_pattern"][position], "content": text})
    return messages, card["card_id"]


def _normalize_words(value: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(value.lower()))


def _ngrams(value: str, size: int) -> set[tuple[str, ...]]:
    words = _normalize_words(value)
    return {tuple(words[index : index + size]) for index in range(len(words) - size + 1)}


def _audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("surface dataset is empty")
    rendered = [row["rendered_text"] for row in rows]
    prompts = [row["prompt"] for row in rows]
    responses = [row["response"] for row in rows]
    all_messages = [message["content"] for row in rows for message in row["messages"]]
    if len(set(rendered)) != len(rendered):
        raise ValueError("duplicate rendered conversations")
    if len(set(prompts)) != len(prompts):
        raise ValueError("duplicate opening prompts")
    if any(_PLACEHOLDER.search(message) for message in all_messages):
        raise ValueError("unrendered template placeholder")
    if any(_LOWERCASE_I.search(message) for message in all_messages):
        raise ValueError("lowercase first-person pronoun")
    for row in rows:
        if row["messages"][0]["role"] != "user" or row["messages"][-1]["role"] != "assistant":
            raise ValueError("dialogues must start with user and end with assistant")
        for position, message in enumerate(row["messages"]):
            expected = "user" if position % 2 == 0 else "assistant"
            content = message["content"]
            if message["role"] != expected or not content.strip():
                raise ValueError("dialogue roles must alternate and contain text")
            if content != content.strip() or content[-1] not in ".?!":
                raise ValueError("messages must be trimmed and end with punctuation")
            if len(content) > 600:
                raise ValueError("message exceeds compact assistant limit")

    length_total = 0
    length_matches = 0
    question_matches = 0
    styles = Counter()
    safe_alternatives = 0
    safe_alternatives_with_followup = 0
    task_context_total = 0
    task_context_matches = 0
    for row in rows:
        contract = json.loads(row["answer_json"])
        expected_lengths = contract["target_length_pattern"]
        realized_lengths = [_length_bucket(message["content"]) for message in row["messages"]]
        length_total += len(expected_lengths)
        length_matches += sum(
            expected == realized
            for expected, realized in zip(expected_lengths, realized_lengths)
        )
        realized_questions = sum(
            message["content"].rstrip().endswith("?") for message in row["messages"]
        )
        question_matches += realized_questions == contract["target_question_turns"]
        styles[contract["response_style"]] += 1
        stages = contract["dialogue_stages"]
        if row["task"] == "practical_dialogue":
            context_id = contract["scenario_card_id"].rsplit(":context:", 1)[-1]
            preferred = _TASK_CONTEXT_RESPONSE_STAGE[context_id]
            task_context_total += 1
            task_context_matches += _task_context_is_satisfied(stages, preferred)
        safe_alternatives += "offer_safe_alternative" in stages
        safe_alternatives_with_followup += (
            "offer_safe_alternative" in stages[:-1]
            and "confirm_choice" in stages
            and "confirm_next_step" in stages
        )

    length_match_ratio = length_matches / length_total
    question_match_ratio = question_matches / len(rows)
    if length_match_ratio < 0.95:
        raise ValueError(f"surface length contract below 95%: {length_match_ratio:.3f}")
    if question_match_ratio != 1.0:
        raise ValueError(f"surface question contract mismatch: {question_match_ratio:.3f}")
    task_context_match_ratio = task_context_matches / task_context_total
    if task_context_match_ratio != 1.0:
        raise ValueError(
            "surface task-context contract mismatch: "
            f"{task_context_match_ratio:.3f}"
        )

    unique_message_ratio = len(set(all_messages)) / len(all_messages)
    if unique_message_ratio < 0.5:
        raise ValueError(
            f"surface message diversity below 50%: {unique_message_ratio:.3f}"
        )
    unique_final_response_ratio = len(set(responses)) / len(responses)
    if unique_final_response_ratio < 0.35:
        raise ValueError(
            "surface final-response diversity below 35%: "
            f"{unique_final_response_ratio:.3f}"
        )
    task_rows = sum(row["task"] == "practical_dialogue" for row in rows)
    safe_alternative_ratio = safe_alternatives / task_rows if task_rows else 0.0
    if safe_alternative_ratio > 0.25:
        raise ValueError(
            f"safe alternatives exceed 25% of practical dialogues: "
            f"{safe_alternative_ratio:.3f}"
        )

    four_grams = Counter(
        gram for message in all_messages for gram in _ngrams(message, 4)
    )
    repeated_four_grams = {" ".join(key): value for key, value in four_grams.items() if value > 1}
    maximum_four_gram_repetitions = max(repeated_four_grams.values(), default=1)
    if maximum_four_gram_repetitions > max(32, ceil(len(all_messages) * 0.05)):
        raise ValueError(
            "a repeated four-word phrase appears in more than 5% of messages"
        )
    top_repeated = dict(
        sorted(repeated_four_grams.items(), key=lambda item: (-item[1], item[0]))[:20]
    )
    category_counts = Counter(f"{row['task']}:{row['domain']}" for row in rows)
    return {
        "rows": len(rows),
        "unique_rendered_ratio": len(set(rendered)) / len(rendered),
        "unique_prompt_ratio": len(set(prompts)) / len(prompts),
        "unique_final_response_ratio": unique_final_response_ratio,
        "unique_message_ratio": unique_message_ratio,
        "mean_messages_per_dialogue": sum(len(row["messages"]) for row in rows) / len(rows),
        "maximum_message_characters": max(map(len, all_messages)),
        "placeholder_leaks": 0,
        "length_contract_match_ratio": length_match_ratio,
        "question_contract_match_ratio": question_match_ratio,
        "task_context_contract_match_ratio": task_context_match_ratio,
        "response_style_counts": dict(sorted(styles.items())),
        "safe_alternative_dialogues": safe_alternatives,
        "safe_alternative_ratio_within_practical_dialogues": safe_alternative_ratio,
        "safe_alternative_dialogues_with_resolution_followup": (
            safe_alternatives_with_followup
        ),
        "maximum_four_gram_repetitions": maximum_four_gram_repetitions,
        "top_repeated_four_grams": top_repeated,
        "category_counts": dict(sorted(category_counts.items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
    }


def _counts(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]) -> dict[str, int]:
    return dict(sorted(Counter(key(row) for row in rows).items()))


def build_conversation_surface(
    blueprints_root: Path,
    scenarios_path: Path,
    output_root: Path,
    *,
    examples: int = 10_000,
    seed: int = 42,
    validation_percent: int = 5,
) -> dict[str, Any]:
    if examples < 64:
        raise ValueError("examples must be at least 64")
    if not 1 <= validation_percent <= 25:
        raise ValueError("validation_percent must be between 1 and 25")
    blueprint_path = blueprints_root / "blueprints.parquet"
    blueprint_rows = pq.read_table(blueprint_path).to_pylist()
    scenarios = json.loads(scenarios_path.read_text())
    context_cards = scenarios["context_cards"]
    task_cards = _expand_scenario_cards(
        scenarios["task_scenarios"],
        context_cards["task_oriented"],
        kind="task_oriented",
    )
    empathy_cards = _expand_scenario_cards(
        scenarios["empathy_scenarios"],
        context_cards["empathetic_conversation"],
        kind="empathetic_conversation",
    )
    selected = _balanced_select(blueprint_rows, pilot_size=examples, seed=seed)

    rows: list[dict[str, Any]] = []
    used_cards: Counter[str] = Counter()
    used_rendered: set[str] = set()
    used_prompts: set[str] = set()
    used_responses: set[str] = set()
    used_messages: set[str] = set()
    for blueprint, rank in selected:
        kind = blueprint["corpus_kind"]
        category = blueprint["category"]
        if kind == "task_oriented":
            if category not in task_cards:
                raise ValueError(f"missing authored task cards for {category}")
            category_cards = task_cards[category]
            card = _adapt_task_card(
                blueprint, category_cards[rank % len(category_cards)]
            )
            surface_blueprint = _adapt_task_blueprint(blueprint, card)
            phrase_rank = rank // len(category_cards)
            messages, card_id = _task_messages(
                surface_blueprint,
                category_cards,
                rank,
                selected_card=card,
            )
            render_position = lambda position, shift: _task_message_at(
                surface_blueprint, card, phrase_rank, position, shift
            )
            task = "practical_dialogue"
        else:
            surface_blueprint = blueprint
            if category not in empathy_cards:
                raise ValueError(f"missing authored empathy cards for {category}")
            category_cards = empathy_cards[category]
            card = category_cards[rank % len(category_cards)]
            messages, card_id = _empathy_messages(
                blueprint, category_cards, rank
            )
            render_position = lambda position, shift: _empathy_message_at(
                blueprint, card, rank, position, shift
            )
            task = "empathetic_dialogue"

        final_position = len(messages) - 1
        current_messages: set[str] = set()
        for position, message in enumerate(messages):
            dedicated = (
                used_prompts
                if position == 0
                else used_responses
                if position == final_position
                else set()
            )
            content = message["content"]
            collision = (
                content in used_messages
                or content in current_messages
                or content in dedicated
            )
            if collision:
                for shift in range(1, _VARIANT_RADIX + 1):
                    candidate = render_position(position, shift)
                    if (
                        candidate not in used_messages
                        and candidate not in current_messages
                        and candidate not in dedicated
                    ):
                        messages[position] = {**message, "content": candidate}
                        content = candidate
                        break
                else:
                    if position in {0, final_position}:
                        for shift in range(1, _VARIANT_RADIX + 1):
                            candidate = render_position(position, shift)
                            if (
                                candidate not in current_messages
                                and candidate not in dedicated
                            ):
                                messages[position] = {
                                    **message,
                                    "content": candidate,
                                }
                                content = candidate
                                break
                        else:
                            if position == 0:
                                raise ValueError(
                                    f"could not render a unique prompt for "
                                    f"{blueprint['blueprint_id']}"
                                )
            current_messages.add(content)

        prompt = messages[0]["content"]
        response = messages[final_position]["content"]

        rendered = _render_messages(messages)
        if rendered in used_rendered:
            raise ValueError(
                f"could not render a unique dialogue for {blueprint['blueprint_id']}"
            )
        used_rendered.add(rendered)
        used_prompts.add(prompt)
        used_responses.add(response)
        used_messages.update(message["content"] for message in messages)
        used_cards[card_id] += 1
        example_id = f"conversation:{hashlib.sha256(rendered.encode()).hexdigest()[:20]}"
        rows.append(
            {
                "example_id": example_id,
                "task": task,
                "mode": "chat" if len(messages) > 2 else "instruct",
                "difficulty": blueprint["difficulty"],
                "dataset_id": DATASET_ID,
                "domain": category,
                "language": "en",
                "split": _split(example_id, validation_percent),
                "messages": messages,
                "prompt": messages[0]["content"],
                "response": messages[-1]["content"],
                "rendered_text": rendered,
                "source_keys": [blueprint["blueprint_id"], card_id],
                "evidence": [],
                "answer_json": json.dumps(
                    {
                        "blueprint_id": blueprint["blueprint_id"],
                        "scenario_card_id": card_id,
                        "dialogue_stages": surface_blueprint["dialogue_stages"],
                        "response_style": blueprint["response_style"],
                        "target_length_pattern": _surface_target_pattern(
                            surface_blueprint, card, kind=kind
                        ),
                        "target_question_turns": surface_blueprint[
                            "target_question_turns"
                        ],
                    },
                    sort_keys=True,
                ),
                "source": SURFACE_SOURCE,
                "source_urls": [],
                "license": SURFACE_LICENSE,
                "version": "1.0.0",
            }
        )
    rows.sort(key=lambda row: row["example_id"])
    audit = _audit(rows)

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    output_path = temporary / "conversations.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=INSTRUCTION_SCHEMA),
        output_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    audit_path = temporary / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = {
        "format": SURFACE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "human-authored general conversation surface dataset",
        "surface_text": {
            "authorship": "Complexity original scenario cards and phrase libraries",
            "model_generated": False,
            "source_utterances_accessed": False,
            "license": SURFACE_LICENSE,
        },
        "seed": seed,
        "examples": examples,
        "validation_percent": validation_percent,
        "inputs": {
            "blueprints": {
                "path": str(blueprint_path),
                "sha256": file_sha256(blueprint_path),
            },
            "scenarios": {
                "path": str(scenarios_path),
                "sha256": file_sha256(scenarios_path),
            },
        },
        "counts": {
            "examples": len(rows),
            "by_task": _counts(rows, lambda row: row["task"]),
            "by_domain": _counts(rows, lambda row: row["domain"]),
            "by_mode": _counts(rows, lambda row: row["mode"]),
            "scenario_cards_used": len(used_cards),
        },
        "scenario_usage": dict(sorted(used_cards.items())),
        "audit": audit,
        "files": {
            "conversations.parquet": {
                "bytes": output_path.stat().st_size,
                "sha256": file_sha256(output_path),
            },
            "audit.json": {
                "bytes": audit_path.stat().st_size,
                "sha256": file_sha256(audit_path),
            },
        },
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return manifest


def build_conversation_surface_pilot(
    blueprints_root: Path,
    scenarios_path: Path,
    output_root: Path,
    *,
    pilot_size: int = 512,
    seed: int = 42,
    validation_percent: int = 5,
) -> dict[str, Any]:
    """Backward-compatible entry point for earlier 512-row pilot commands."""
    return build_conversation_surface(
        blueprints_root,
        scenarios_path,
        output_root,
        examples=pilot_size,
        seed=seed,
        validation_percent=validation_percent,
    )
