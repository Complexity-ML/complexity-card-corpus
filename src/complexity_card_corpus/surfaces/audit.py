from __future__ import annotations

import json
from collections import Counter
from math import ceil
from typing import Any, Callable

from .common import _LOWERCASE_I, _PLACEHOLDER, _WORD, _length_bucket
from .rendering import _TASK_CONTEXT_RESPONSE_STAGE, _task_context_is_satisfied


def _normalize_words(value: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(value.lower()))


def _ngrams(value: str, size: int) -> set[tuple[str, ...]]:
    words = _normalize_words(value)
    return {
        tuple(words[index : index + size]) for index in range(len(words) - size + 1)
    }


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
        if (
            row["messages"][0]["role"] != "user"
            or row["messages"][-1]["role"] != "assistant"
        ):
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
    train_source_cards: set[str] = set()
    validation_source_cards: set[str] = set()
    for row in rows:
        contract = json.loads(row["answer_json"])
        source_cards = (
            validation_source_cards
            if row["split"] == "validation"
            else train_source_cards
        )
        source_cards.add(contract["scenario_card_id"])
        expected_lengths = contract["target_length_pattern"]
        realized_lengths = [
            _length_bucket(message["content"]) for message in row["messages"]
        ]
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
        raise ValueError(
            f"surface question contract mismatch: {question_match_ratio:.3f}"
        )
    task_context_match_ratio = task_context_matches / task_context_total
    if task_context_match_ratio != 1.0:
        raise ValueError(
            f"surface task-context contract mismatch: {task_context_match_ratio:.3f}"
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
    repeated_four_grams = {
        " ".join(key): value for key, value in four_grams.items() if value > 1
    }
    maximum_four_gram_repetitions = max(repeated_four_grams.values(), default=1)
    if maximum_four_gram_repetitions > max(32, ceil(len(all_messages) * 0.05)):
        raise ValueError(
            "a repeated four-word phrase appears in more than 5% of messages"
        )
    top_repeated = dict(
        sorted(repeated_four_grams.items(), key=lambda item: (-item[1], item[0]))[:20]
    )
    category_counts = Counter(f"{row['task']}:{row['domain']}" for row in rows)
    leaking_source_cards = train_source_cards & validation_source_cards
    if leaking_source_cards:
        raise ValueError("source scenario cards leak across conversation splits")
    return {
        "rows": len(rows),
        "unique_rendered_ratio": len(set(rendered)) / len(rendered),
        "unique_prompt_ratio": len(set(prompts)) / len(prompts),
        "unique_final_response_ratio": unique_final_response_ratio,
        "unique_message_ratio": unique_message_ratio,
        "mean_messages_per_dialogue": sum(len(row["messages"]) for row in rows)
        / len(rows),
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
        "split_holdout_unit": "scenario_card_id",
        "source_card_split_overlap": len(leaking_source_cards),
    }


def _counts(
    rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]
) -> dict[str, int]:
    return dict(sorted(Counter(key(row) for row in rows).items()))
