from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .common import (
    _choose_variant,
    _default_variant_shift,
    _digest,
    _length_bucket,
    _lower_first,
    _position_variant_shift,
    _recommendation_from_choice,
    _render_candidates,
)
from .templates import (
    _EMPATHY_FRAMES,
    _EMPATHY_TEMPLATES,
    _TASK_FRAMES,
    _TASK_TEMPLATES,
)


def _surface_target_length(
    blueprint: dict[str, Any],
    card: dict[str, str],
    position: int,
    *,
    kind: str,
) -> str:
    stage = blueprint["dialogue_stages"][position]
    templates = (
        _TASK_TEMPLATES[stage] if kind == "task_oriented" else _EMPATHY_TEMPLATES[stage]
    )
    frames = _TASK_FRAMES[stage] if kind == "task_oriented" else _EMPATHY_FRAMES[stage]
    values = _task_values(card) if kind == "task_oriented" else _empathy_values(card)
    counts = Counter(
        _length_bucket(value) for value in _render_candidates(templates, frames, values)
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


def _adapt_task_card(blueprint: dict[str, Any], card: dict[str, str]) -> dict[str, str]:
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
        return "give_choice_and_action" in stages or {
            "confirm_choice",
            "confirm_next_step",
        }.issubset(stages)
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
        messages.append(
            {"role": blueprint["target_speaker_pattern"][position], "content": text}
        )
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
        messages.append(
            {"role": blueprint["target_speaker_pattern"][position], "content": text}
        )
    return messages, card["card_id"]
