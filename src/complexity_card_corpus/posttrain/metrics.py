from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from typing import Any

from .constants import (
    DATASET_LICENSE,
    _FORBIDDEN_ASSISTANT_META_PHRASES,
    _FORBIDDEN_USER_META_PHRASES,
    _INTENT_FIELD,
    _MASKED_RESPONSE_FIELDS,
    _MAX_FAMILY_SKELETON_SHARE,
    _MAX_SURFACE_FORMULATION_SHARE,
    _WORD,
)


def _tokens(value: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD.finditer(value)]


def _p95(values: list[int]) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * 0.95))
    return float(ordered[index])


def _length_statistics(values: list[str]) -> dict[str, float | int]:
    lengths = [len(_tokens(value)) for value in values]
    return {
        "items": len(lengths),
        "minimum_tokens": min(lengths),
        "mean_tokens": round(statistics.fmean(lengths), 3),
        "median_tokens": round(statistics.median(lengths), 3),
        "p95_tokens": _p95(lengths),
        "maximum_tokens": max(lengths),
    }


def _mattr(tokens: list[str], window: int = 100) -> float:
    """Return moving-average type-token ratio over fixed lexical windows."""
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    counts = Counter(tokens[:window])
    total = len(counts) / window
    windows = 1
    for index in range(window, len(tokens)):
        outgoing = tokens[index - window]
        counts[outgoing] -= 1
        if counts[outgoing] == 0:
            del counts[outgoing]
        counts[tokens[index]] += 1
        total += len(counts) / window
        windows += 1
    return total / windows


def _ngram_statistics(messages: list[str], size: int) -> dict[str, Any]:
    counts: Counter[tuple[str, ...]] = Counter()
    message_counts: Counter[tuple[str, ...]] = Counter()
    for message in messages:
        tokens = _tokens(message)
        grams = [
            tuple(tokens[index : index + size])
            for index in range(len(tokens) - size + 1)
        ]
        counts.update(grams)
        message_counts.update(set(grams))
    windows = sum(counts.values())
    distinct = len(counts)
    singleton_ngrams = sum(value == 1 for value in counts.values())
    most_common = counts.most_common(10)
    return {
        "window_tokens": size,
        "windows": windows,
        "distinct_ngrams": distinct,
        "distinct_ngram_ratio": round(distinct / windows, 6),
        "singleton_ngrams": singleton_ngrams,
        "singleton_distinct_ratio": round(singleton_ngrams / distinct, 6),
        "singleton_window_ratio": round(singleton_ngrams / windows, 6),
        "maximum_occurrences": max(counts.values(), default=0),
        "maximum_message_coverage": round(
            max(message_counts.values(), default=0) / len(messages), 6
        )
        if messages
        else 0.0,
        "top_repeated_ngrams": [
            {
                "text": " ".join(gram),
                "occurrences": occurrences,
                "message_count": message_counts[gram],
                "message_rate": round(message_counts[gram] / len(messages), 6)
                if messages
                else 0.0,
            }
            for gram, occurrences in most_common
        ],
    }


def _text_statistics(values: list[str]) -> dict[str, Any]:
    tokens = [token for value in values for token in _tokens(value)]
    return {
        "items": len(values),
        "exact_unique_items": len(set(values)),
        "exact_uniqueness_ratio": round(len(set(values)) / len(values), 6)
        if values
        else 0.0,
        "length": _length_statistics(values),
        "lexical": {
            "word_occurrences": len(tokens),
            "observed_vocabulary": len(set(tokens)),
            "raw_type_token_ratio": round(len(set(tokens)) / len(tokens), 6)
            if tokens
            else 0.0,
            "mattr_100": round(_mattr(tokens, 100), 6),
        },
        "four_grams": _ngram_statistics(values, 4),
        "eight_grams": _ngram_statistics(values, 8),
    }


def _masked_response(response: str, answer: dict[str, Any]) -> str:
    replacements = [
        (str(answer[source]).strip().rstrip("."), f"<{target}>")
        for source, target in _MASKED_RESPONSE_FIELDS
        if str(answer.get(source, "")).strip().rstrip(".")
    ]
    masked = response
    for value, placeholder in sorted(replacements, key=lambda item: -len(item[0])):
        masked = re.sub(re.escape(value), placeholder, masked, flags=re.IGNORECASE)
    scenario_id = str(answer.get("scenario_id", ""))
    codes = {scenario_id.split(":")[-1][:6] if scenario_id else ""}
    prefix = re.match(r"^(?:for\s+)?hand\s+([0-9a-f]{6})\s*[:—-]", masked, re.I)
    if prefix:
        codes.add(prefix.group(1))
    for code in sorted(filter(None, codes), key=len, reverse=True):
        masked = re.sub(
            rf"\b{re.escape(code)}(?:-[ab])?\b",
            "<id>",
            masked,
            flags=re.IGNORECASE,
        )
    masked = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "<date>", masked)
    masked = re.sub(r"\$\d+(?:\.\d+)?", "<amount>", masked)
    masked = re.sub(r"\b\d{1,2}:\d{2}\b", "<time>", masked)
    masked = re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", masked)
    return re.sub(r"\s+", " ", masked).strip().lower()


def _masked_diversity(
    responses: list[str], answers: list[dict[str, Any]]
) -> dict[str, Any]:
    skeletons = [
        _masked_response(response, answer)
        for response, answer in zip(responses, answers, strict=True)
    ]
    counts = Counter(skeletons)
    maximum = max(counts.values(), default=0)
    maximum_share = maximum / len(skeletons) if skeletons else 0.0
    if maximum_share >= _MAX_SURFACE_FORMULATION_SHARE:
        raise ValueError(
            "masked response skeleton reaches the five-percent ceiling: "
            f"{maximum_share:.3%}"
        )
    return {
        "masked_fields": [target for _, target in _MASKED_RESPONSE_FIELDS],
        "masked_surface_variables": [
            "scenario_code",
            "reference_suffix",
            "date",
            "amount",
            "time",
            "number",
        ],
        "skeletons": len(skeletons),
        "distinct_skeletons": len(counts),
        "exact_skeleton_uniqueness_ratio": round(len(counts) / len(skeletons), 6)
        if skeletons
        else 0.0,
        "maximum_examples_per_skeleton": maximum,
        "maximum_skeleton_share": round(maximum_share, 6),
        "strict_share_limit": _MAX_SURFACE_FORMULATION_SHARE,
        "four_gram_stats": _ngram_statistics(skeletons, 4),
        "eight_gram_stats": _ngram_statistics(skeletons, 8),
        "top_repeated_skeletons": [
            {"text": text, "examples": count, "share": round(count / len(skeletons), 6)}
            for text, count in counts.most_common(10)
        ],
    }


def _audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rendered = [row["rendered_text"] for row in rows]
    responses = [row["response"] for row in rows]
    messages = [message["content"] for row in rows for message in row["messages"]]
    user_prompts = [
        message["content"]
        for row in rows
        for message in row["messages"]
        if message["role"] == "user"
    ]
    assistant_messages = [
        message["content"]
        for row in rows
        for message in row["messages"]
        if message["role"] == "assistant"
    ]
    assistant_meta_hits = [
        {"phrase": phrase, "text": message}
        for message in assistant_messages
        for phrase in _FORBIDDEN_ASSISTANT_META_PHRASES
        if phrase in message.lower()
    ]
    user_meta_hits = [
        {"phrase": phrase, "text": message}
        for message in user_prompts
        for phrase in _FORBIDDEN_USER_META_PHRASES
        if phrase in message.lower()
    ]
    if assistant_meta_hits:
        raise ValueError(
            "assistant text describes how to answer instead of answering directly: "
            f"{assistant_meta_hits[0]}"
        )
    if user_meta_hits:
        raise ValueError(
            "user text asks for meta-response construction instead of the task: "
            f"{user_meta_hits[0]}"
        )
    train_cards: set[str] = set()
    validation_cards: set[str] = set()
    split_groups: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    exact_anchor_rows = 0
    answers: list[dict[str, Any]] = []
    first_user_messages: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        answer = json.loads(row["answer_json"])
        answers.append(answer)
        first_user_messages[answer["scenario_id"]][answer["mode"]] = next(
            message["content"]
            for message in row["messages"]
            if message["role"] == "user"
        )
        target = validation_cards if row["split"] == "validation" else train_cards
        target.add(answer["scenario_id"])
        split_groups[row["split"]].add(
            (answer["family"], answer["domain"], answer["intent"])
        )
        normalized = row["rendered_text"].lower()
        if all(
            normalized.count(answer[field].rstrip(".").lower()) == 1
            for field in ("state", "constraint")
        ):
            exact_anchor_rows += 1
    card_overlap = train_cards & validation_cards
    group_overlap = split_groups["train"] & split_groups["validation"]
    if card_overlap:
        raise ValueError("source scenarios leak across post-training splits")
    if group_overlap:
        raise ValueError("family/domain/intent groups leak across post-training splits")
    if len(set(rendered)) != len(rendered):
        raise ValueError("duplicate post-training conversations")
    if any(row["license"] != DATASET_LICENSE for row in rows):
        raise ValueError("post-training license mismatch")
    if exact_anchor_rows != len(rows):
        raise ValueError("state and constraint must each appear exactly once")

    paired_prompt_surfaces = [
        modes
        for modes in first_user_messages.values()
        if "instruct" in modes and "chat" in modes
    ]
    exact_first_user_matches = sum(
        modes["instruct"] == modes["chat"] for modes in paired_prompt_surfaces
    )
    chat_opener_prefix_matches = sum(
        modes["instruct"].startswith(modes["chat"]) for modes in paired_prompt_surfaces
    )
    if exact_first_user_matches or chat_opener_prefix_matches:
        raise ValueError(
            "paired chat and instruct prompts must use independent surface openings"
        )

    unique_final_response_ratio = len(set(responses)) / len(responses)
    if unique_final_response_ratio < 0.95:
        raise ValueError(
            "post-training final responses are not sufficiently individualized: "
            f"{unique_final_response_ratio:.3f}"
        )

    card_contracts: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for answer in answers:
        card_hand = answer.get("card_hand")
        if not isinstance(card_hand, dict):
            raise ValueError("post-training answer is missing its card-hand contract")
        if card_hand.get("cards") != ["situation", "data", "rule", "goal"]:
            raise ValueError("post-training card hand has an invalid card sequence")
        contract = tuple(card_hand.get("completion_contract", ()))
        if not contract:
            raise ValueError("post-training card hand has an empty completion contract")
        card_contracts[answer["family"]].add(contract)
    if set(card_contracts) != set(_INTENT_FIELD):
        raise ValueError("post-training card hands do not cover all task families")

    final_response_stats = _text_statistics(responses)
    masked_response_diversity = _masked_diversity(responses, answers)
    maximum_final_phrase_share = masked_response_diversity["eight_gram_stats"][
        "maximum_message_coverage"
    ]
    if maximum_final_phrase_share >= _MAX_SURFACE_FORMULATION_SHARE:
        raise ValueError(
            "a masked final-response eight-token phrase reaches the "
            "five-percent ceiling: "
            f"{maximum_final_phrase_share:.3%}"
        )
    all_message_tokens = [token for message in messages for token in _tokens(message)]
    vocabulary = set(all_message_tokens)
    family_metrics: dict[str, dict[str, Any]] = {}
    for family in sorted({row["task"] for row in rows}):
        family_rows = [row for row in rows if row["task"] == family]
        family_answers = [
            json.loads(row["answer_json"])["scenario_id"] for row in family_rows
        ]
        family_responses = [row["response"] for row in family_rows]
        family_answer_json = [json.loads(row["answer_json"]) for row in family_rows]
        family_skeletons = [
            _masked_response(response, answer)
            for response, answer in zip(
                family_responses, family_answer_json, strict=True
            )
        ]
        family_skeleton_counts = Counter(family_skeletons)
        maximum_family_skeleton_share = max(
            family_skeleton_counts.values(), default=0
        ) / len(family_skeletons)
        if maximum_family_skeleton_share >= _MAX_FAMILY_SKELETON_SHARE:
            raise ValueError(
                f"{family} response skeleton reaches the family repetition ceiling: "
                f"{maximum_family_skeleton_share:.3%}"
            )
        family_metrics[family] = {
            "examples": len(family_rows),
            "source_scenarios": len(set(family_answers)),
            "unique_final_response_ratio": round(
                len(set(family_responses)) / len(family_responses), 6
            ),
            "mean_response_tokens": _length_statistics(family_responses)["mean_tokens"],
            "masked_response_templates": len(family_skeleton_counts),
            "maximum_masked_template_share": round(maximum_family_skeleton_share, 6),
            "strict_masked_template_share_limit": _MAX_FAMILY_SKELETON_SHARE,
        }
    family_source_scenarios: dict[str, set[str]] = defaultdict(set)
    for answer in answers:
        family_source_scenarios[answer["family"]].add(answer["scenario_id"])

    return {
        "rows": len(rows),
        "source_scenarios": len(train_cards | validation_cards),
        "split_holdout_units": ["scenario_id", "family+domain+intent"],
        "source_scenario_split_overlap": len(card_overlap),
        "semantic_group_split_overlap": len(group_overlap),
        "train_semantic_groups": len(split_groups["train"]),
        "validation_semantic_groups": len(split_groups["validation"]),
        "split_example_counts": dict(
            sorted(Counter(row["split"] for row in rows).items())
        ),
        "family_example_counts": dict(
            sorted(Counter(row["task"] for row in rows).items())
        ),
        "family_source_scenario_counts": dict(
            sorted(
                (family, len(scenarios))
                for family, scenarios in family_source_scenarios.items()
            )
        ),
        "mode_example_counts": dict(
            sorted(Counter(row["mode"] for row in rows).items())
        ),
        "exact_conversation_uniqueness_ratio": len(set(rendered)) / len(rendered),
        "exact_final_response_uniqueness_ratio": unique_final_response_ratio,
        "duplicate_final_response_rows": len(responses) - len(set(responses)),
        "card_game": {
            "cards_per_hand": ["situation", "data", "rule", "goal"],
            "families": len(card_contracts),
            "family_completion_contracts": {
                family: [list(contract) for contract in sorted(contracts)]
                for family, contracts in sorted(card_contracts.items())
            },
            "all_hands_have_non_empty_contract": True,
        },
        "model_generated_dialogue_rows": sum(
            bool(answer["model_generated_dialogue"]) for answer in answers
        ),
        "single_state_and_constraint_ratio": exact_anchor_rows / len(rows),
        "paired_prompt_surface_stats": {
            "paired_scenarios": len(paired_prompt_surfaces),
            "exact_first_user_message_matches": exact_first_user_matches,
            "chat_opener_is_instruct_prefix": chat_opener_prefix_matches,
            "shared_semantics": [
                "scenario_id",
                "state",
                "constraint",
                "desired_outcome",
            ],
        },
        "response_repetition_gate": {
            "maximum_masked_eight_token_message_coverage": maximum_final_phrase_share,
            "strict_share_limit": _MAX_SURFACE_FORMULATION_SHARE,
            "measured_from_rendered_responses": True,
        },
        "masked_response_diversity": masked_response_diversity,
        "role_text_stats": {
            "user_prompts": _text_statistics(user_prompts),
            "assistant_messages": _text_statistics(assistant_messages),
            "final_responses": final_response_stats,
        },
        "natural_language_gate": {
            "assistant_meta_instruction_hits": len(assistant_meta_hits),
            "user_meta_request_hits": len(user_meta_hits),
            "forbidden_assistant_phrases": list(_FORBIDDEN_ASSISTANT_META_PHRASES),
            "forbidden_user_phrases": list(_FORBIDDEN_USER_META_PHRASES),
        },
        "message_length_stats": _length_statistics(messages),
        "response_length_stats": _length_statistics(responses),
        "lexical_stats": {
            "word_occurrences": len(all_message_tokens),
            "observed_vocabulary": len(vocabulary),
            "raw_type_token_ratio": round(len(vocabulary) / len(all_message_tokens), 6),
            "mattr_100": round(_mattr(all_message_tokens, 100), 6),
        },
        "four_gram_stats": _ngram_statistics(messages, 4),
        "eight_gram_stats": _ngram_statistics(messages, 8),
        "family_metrics": family_metrics,
    }
