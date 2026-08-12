from __future__ import annotations

import json
from pathlib import Path

import pytest

from complexity_card_corpus.conversational import render_casual_conversation_rows
from complexity_card_corpus.sft.projection import _project_sft_conversation
from complexity_card_corpus.variable_by.reservoirs import (
    CASUAL_ARC_CARDS,
    CASUAL_INTENT_CARDS,
)


REGISTRY = (
    Path(__file__).parents[1]
    / "data/conversation/original/casual-conversation-decks-v1.json"
)


def test_casual_registry_has_decks_subcards_and_large_capacity() -> None:
    registry = json.loads(REGISTRY.read_text())

    assert registry["license"] == "CC BY-NC 4.0"
    assert len(registry["topic_cards"]) >= 20
    assert len(registry["context_cards"]) >= 20
    assert set(registry["surface_decks"]) == {
        "user_opening",
        "assistant_entry",
        "user_follow_up",
        "assistant_follow_up",
        "user_shift",
        "assistant_closing",
    }
    assert all(
        len(deck) >= 8 for deck in registry["surface_decks"].values()
    )
    assert len(registry["surface_decks"]["assistant_closing"]) >= 32
    assert all(
        {
            "opening",
            "acknowledgement",
            "question",
            "detail",
            "reply",
            "shift",
            "closing",
        }
        <= set(card["subcards"])
        for card in registry["topic_cards"]
    )
    semantic_pairs = len(registry["topic_cards"]) * len(
        registry["context_cards"]
    ) * len(CASUAL_INTENT_CARDS) * len(CASUAL_ARC_CARDS)
    assert semantic_pairs == 30_240


def test_casual_surface_is_additive_natural_and_below_five_percent(
) -> None:
    rows, result = render_casual_conversation_rows(REGISTRY, seed=42)
    repeated_rows, repeated = render_casual_conversation_rows(REGISTRY, seed=42)

    assert result["content_sha256"] == repeated["content_sha256"]
    assert rows == repeated_rows
    assert result["counts"]["examples"] == 30_240
    assert result["counts"]["by_task"] == {"casual_conversation": 30_240}
    assert result["counts"]["by_split"] == {
        "train": 28_728,
        "validation": 1_512,
    }
    assert result["audit"]["largest_surface_hand_share"] <= 0.05
    assert result["audit"]["largest_response_structure_share"] <= 0.05
    assert result["audit"]["unique_conversation_ratio"] == 1.0
    assert result["audit"]["unique_final_response_ratio"] == 1.0
    assert result["audit"]["source_pair_split_overlap"] == 0
    assert result["audit"]["conversation_quality"]["passed"] is True
    assert result["audit"]["conversation_quality"]["grammar_defects"] == {}
    final_sentence_counts = result["audit"]["conversation_quality"][
        "final_sentence_counts"
    ]
    assert set(final_sentence_counts) == {1, 2, 3}
    assert all(count / 30_240 >= 0.25 for count in final_sentence_counts.values())

    assert {row["task"] for row in rows} == {"casual_conversation"}
    assert {row["mode"] for row in rows} == {"chat"}
    assert {len(row["messages"]) for row in rows} == {4, 6}
    assert all(row["messages"][0]["role"] == "user" for row in rows)
    assert all(row["messages"][-1]["role"] == "assistant" for row in rows)
    assert all(
        1 <= len(row["messages"][-1]["content"].split(". ")) <= 3
        for row in rows
    )
    assert all(
        " card" not in f" {message['content'].lower()}"
        and " deck" not in f" {message['content'].lower()}"
        and "hand " not in message["content"].lower()
        for row in rows
        for message in row["messages"]
    )
    metadata = [json.loads(row["answer_json"]) for row in rows]
    assert all(
        "surface[user_opening]" in item["deck_topology"]["variable_by"]
        and "topic[reply_lower]" in item["deck_topology"]["variable_by"]
        and "context[closing]" in item["deck_topology"]["variable_by"]
        and "intent[user_opening]" in item["deck_topology"]["variable_by"]
        and "arc[user_follow_up]" in item["deck_topology"]["variable_by"]
        and "semantic[closing]" in item["deck_topology"]["variable_by"]
        for item in metadata
    )


def test_default_casual_release_deals_one_row_per_semantic_pair(
) -> None:
    _rows, result = render_casual_conversation_rows(REGISTRY, seed=42)

    registry = json.loads(REGISTRY.read_text())
    semantic_pairs = (
        len(registry["topic_cards"])
        * len(registry["context_cards"])
        * len(CASUAL_INTENT_CARDS)
        * len(CASUAL_ARC_CARDS)
    )
    assert result["counts"]["examples"] == semantic_pairs
    assert result["audit"]["largest_source_pair_variant_count"] == 1


def test_sft_projection_preserves_casual_dialogue_turns() -> None:
    rows, _summary = render_casual_conversation_rows(REGISTRY, seed=42)

    for row in rows[:20]:
        projected, _cards = _project_sft_conversation(
            row["messages"],
            example_id=row["example_id"],
            task=row["task"],
            answer_json=row["answer_json"],
        )
        assert projected == row["messages"]


def test_casual_surface_rejects_semantic_unit_recycling(
) -> None:
    with pytest.raises(ValueError, match="exceeds card capacity 30240"):
        render_casual_conversation_rows(
            REGISTRY,
            examples=30_241,
            seed=42,
        )
