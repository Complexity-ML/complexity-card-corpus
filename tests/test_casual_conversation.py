from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from complexity_card_corpus.conversational import build_casual_conversation_surface
from complexity_card_corpus.sft.projection import _project_sft_conversation


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
        "assistant_closing_bridge",
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
    )
    surface_capacity = 1
    for deck in registry["surface_decks"].values():
        surface_capacity *= len(deck)
    assert semantic_pairs >= 400
    assert semantic_pairs * surface_capacity >= 1_000_000


def test_casual_surface_is_additive_natural_and_below_five_percent(
    tmp_path: Path,
) -> None:
    result = build_casual_conversation_surface(
        REGISTRY,
        tmp_path / "casual",
        examples=2_000,
        seed=42,
    )
    repeated = build_casual_conversation_surface(
        REGISTRY,
        tmp_path / "repeat",
        examples=2_000,
        seed=42,
    )

    assert result["files"]["conversations.parquet"]["sha256"] == repeated[
        "files"
    ]["conversations.parquet"]["sha256"]
    assert result["counts"]["examples"] == 2_000
    assert result["counts"]["by_task"] == {"casual_conversation": 2_000}
    assert result["audit"]["largest_surface_hand_share"] <= 0.05
    assert result["audit"]["largest_response_structure_share"] <= 0.05
    assert result["audit"]["unique_conversation_ratio"] == 1.0
    assert result["audit"]["unique_final_response_ratio"] == 1.0
    assert result["audit"]["source_pair_split_overlap"] == 0

    rows = pq.read_table(tmp_path / "casual/conversations.parquet").to_pylist()
    assert {row["task"] for row in rows} == {"casual_conversation"}
    assert {row["mode"] for row in rows} == {"chat"}
    assert {len(row["messages"]) for row in rows} == {4, 6}
    assert all(row["messages"][0]["role"] == "user" for row in rows)
    assert all(row["messages"][-1]["role"] == "assistant" for row in rows)
    assert all(
        " card" not in f" {message['content'].lower()}"
        and " deck" not in f" {message['content'].lower()}"
        and "hand " not in message["content"].lower()
        for row in rows
        for message in row["messages"]
    )


def test_default_casual_release_deals_one_row_per_semantic_pair(
    tmp_path: Path,
) -> None:
    result = build_casual_conversation_surface(
        REGISTRY,
        tmp_path / "casual-default",
        seed=42,
    )

    registry = json.loads(REGISTRY.read_text())
    semantic_pairs = len(registry["topic_cards"]) * len(registry["context_cards"])
    assert result["counts"]["examples"] == semantic_pairs
    assert result["audit"]["largest_source_pair_variant_count"] == 1


def test_sft_projection_preserves_casual_dialogue_turns(tmp_path: Path) -> None:
    build_casual_conversation_surface(REGISTRY, tmp_path / "casual", seed=42)
    rows = pq.read_table(tmp_path / "casual/conversations.parquet").to_pylist()

    for row in rows[:20]:
        projected, _cards = _project_sft_conversation(
            row["messages"],
            example_id=row["example_id"],
            task=row["task"],
            answer_json=row["answer_json"],
        )
        assert projected == row["messages"]


def test_casual_surface_hydrates_to_thirty_thousand_without_recycling(
    tmp_path: Path,
) -> None:
    result = build_casual_conversation_surface(
        REGISTRY,
        tmp_path / "casual-30k",
        examples=30_000,
        seed=42,
    )

    assert result["counts"]["examples"] == 30_000
    assert result["audit"]["unique_conversation_ratio"] == 1.0
    assert result["audit"]["unique_final_response_ratio"] == 1.0
    assert result["audit"]["largest_four_gram_share_per_message"] <= 0.05
