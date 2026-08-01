from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from complexity_card_corpus.build import CARD_SCHEMA, DOCUMENT_SCHEMA, RELATION_SCHEMA
from complexity_card_corpus.instruct import (
    IGNORE_INDEX,
    _naturalize_assistant_target,
    build_instruction_dataset,
    tokenize_instruction_dataset,
)
from complexity_card_corpus.training_cards import TrainingCards
from complexity_card_corpus.chat_template import (
    CHAT_TEMPLATE_ID,
    render_system_prefix,
)
from complexity_card_corpus.package import package_instructions_for_hugging_face
from complexity_card_corpus.tokenize import load_encoding


def _card(dataset_id: str, split: str, key: str, name: str) -> dict:
    return {
        "dataset_id": dataset_id,
        "domain": "fantasy",
        "themes": ["test atlas"],
        "language": "en",
        "split": split,
        "key": key,
        "kind": "artifact",
        "name": name,
        "aliases": [],
        "summary": f"{name} records a precise test property.",
        "description": f"{name} records a precise test property and no other claim.",
        "facts": [f"Documented fact: {name} is part of the test atlas."],
        "tags": ["test"],
        "attributes_json": json.dumps({"material": "blue glass", "status": "catalogued"}),
        "source": "Complexity original test cards",
        "source_urls": [],
        "license": "CC BY-NC 4.0",
        "version": "1.0.0",
    }


def _tiny_corpus(root: Path) -> None:
    root.mkdir()
    cards = [
        _card("train-deck", "train", "artifact:alpha", "Alpha Lens"),
        _card("train-deck", "train", "artifact:beta", "Beta Bell"),
        _card("validation-deck", "validation", "artifact:gamma", "Gamma Key"),
        _card("validation-deck", "validation", "artifact:delta", "Delta Map"),
    ]
    relations = [
        {
            "dataset_id": "train-deck",
            "split": "train",
            "from_key": "artifact:alpha",
            "relation": "reveals",
            "to_dataset_id": "train-deck",
            "to_key": "artifact:beta",
            "detail": "Alpha Lens reveals Beta Bell.",
        },
        {
            "dataset_id": "validation-deck",
            "split": "validation",
            "from_key": "artifact:gamma",
            "relation": "locates",
            "to_dataset_id": "validation-deck",
            "to_key": "artifact:delta",
            "detail": "Gamma Key locates Delta Map.",
        },
    ]
    documents = [
        {
            "document_id": "train-deck:path:alpha:00",
            "dataset_id": "train-deck",
            "domain": "fantasy",
            "themes": ["test atlas"],
            "language": "en",
            "split": "train",
            "template": "path",
            "source_keys": ["artifact:alpha", "artifact:beta"],
            "text": "Relationship path from Alpha Lens\n\nAlpha Lens reveals Beta Bell.",
            "source": "Complexity original test cards",
            "source_urls": [],
            "license": "CC BY-NC 4.0",
            "version": "1.0.0",
        },
        {
            "document_id": "validation-deck:path:gamma:00",
            "dataset_id": "validation-deck",
            "domain": "fantasy",
            "themes": ["test atlas"],
            "language": "en",
            "split": "validation",
            "template": "path",
            "source_keys": ["artifact:gamma", "artifact:delta"],
            "text": "Relationship path from Gamma Key\n\nGamma Key locates Delta Map.",
            "source": "Complexity original test cards",
            "source_urls": [],
            "license": "CC BY-NC 4.0",
            "version": "1.0.0",
        },
    ]
    pq.write_table(pa.Table.from_pylist(cards, schema=CARD_SCHEMA), root / "cards.parquet")
    pq.write_table(
        pa.Table.from_pylist(relations, schema=RELATION_SCHEMA),
        root / "relations.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(documents, schema=DOCUMENT_SCHEMA),
        root / "documents.parquet",
    )


def test_original_instructions_are_deterministic_and_deck_split(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _tiny_corpus(corpus)
    first = build_instruction_dataset(corpus, tmp_path / "first")
    second = build_instruction_dataset(corpus, tmp_path / "second")
    assert first["files"]["instructions.parquet"]["sha256"] == second["files"]["instructions.parquet"]["sha256"]

    rows = pq.read_table(tmp_path / "first/instructions.parquet").to_pylist()
    assert {row["split"] for row in rows} == {"train", "validation"}
    assert len({row["example_id"] for row in rows}) == len(rows)
    assert {row["mode"] for row in rows} == {"instruct", "chat"}
    assert all(row["license"] == "CC BY-NC 4.0" for row in rows)
    for row in rows:
        expected_deck = "train-deck" if row["split"] == "train" else "validation-deck"
        assert row["dataset_id"] == expected_deck
        assert row["messages"][0]["role"] == "user"
        assert row["messages"][-1]["role"] == "assistant"
        assert row["evidence"]
        if row["task"] == "structured_extraction":
            assert json.loads(row["response"]) == json.loads(row["answer_json"])


def test_sft_bin_masks_user_tokens_and_supervises_assistant(tmp_path: Path) -> None:
    tokenizer = Path("/Users/boris/Dev/complexity-framework/tokenizer-o200k")
    if not tokenizer.exists():
        return
    corpus = tmp_path / "corpus"
    _tiny_corpus(corpus)
    build_instruction_dataset(corpus, tmp_path / "instructions")
    manifest = tokenize_instruction_dataset(
        tmp_path / "instructions/instructions.parquet",
        tokenizer,
        tmp_path / "tokenized",
    )
    assert manifest["total_examples"] > 0
    assert manifest["chat_template_id"] == CHAT_TEMPLATE_ID
    template_path = tmp_path / "tokenized" / "chat_template.json"
    assert template_path.exists()
    template = json.loads(template_path.read_text())
    assert template["id"] == CHAT_TEMPLATE_ID
    assert template["assistant_only_loss"] is True
    assert template["training_projection"] == (
        "naturalize_card_hand_target_final_assistant"
    )
    for partition, metadata in manifest["partitions"].items():
        assert set(metadata["conditioning_card_counts"]) == {
            "surface",
            "dialogue_state",
            "output",
            "evidence",
            "reasoning",
            "style",
            "context_density",
            "noise",
            "uncertainty",
        }
        assert all(
            sum(counts.values()) == metadata["examples"]
            for counts in metadata["conditioning_card_counts"].values()
        )
        input_ids = np.fromfile(
            tmp_path / "tokenized" / partition / "input_ids.bin",
            dtype="<u4",
        )
        labels = np.fromfile(
            tmp_path / "tokenized" / partition / "labels.bin",
            dtype="<i4",
        )
        assert len(input_ids) == len(labels) == metadata["num_tokens"]
        assert np.any(labels == IGNORE_INDEX)
        supervised = labels != IGNORE_INDEX
        assert np.any(supervised)
        with (tmp_path / "tokenized" / partition / "examples.jsonl").open() as handle:
            examples = [json.loads(line) for line in handle]
        source_rows = {
            row["example_id"]: row
            for row in pq.read_table(
                tmp_path / "instructions/instructions.parquet"
            ).to_pylist()
        }
        for example in examples:
            start = example["offset"]
            end = start + example["num_tokens"]
            local_inputs = input_ids[start:end]
            local_labels = labels[start:end]
            local_supervised = local_labels[:-1] != IGNORE_INDEX
            assert np.array_equal(
                local_inputs[1:][local_supervised],
                local_labels[:-1][local_supervised],
            )
            decoded = load_encoding(tokenizer)[0].decode(local_inputs.tolist())
            source = source_rows[example["example_id"]]
            assert decoded.startswith(render_system_prefix(template) + "User:\n")
            assert "\n\nAssistant:\n" in decoded
            assert "SITUATION CARD" not in decoded
            assert "DATA CARD" not in decoded
            assert "RULE CARD" not in decoded
            assert "GOAL CARD" not in decoded
            assert "card hand" not in decoded.lower()
            assert example["hand_id"] == source["example_id"]
            assert example["training_representation"] == "natural_instruction"
            assert set(example["conditioning_cards"]) == {
                "surface",
                "dialogue_state",
                "output",
                "evidence",
                "reasoning",
                "style",
                "context_density",
                "noise",
                "uncertainty",
            }
            has_card_hand = any(
                "SITUATION CARD" in message["content"]
                for message in source["messages"]
                if message["role"] == "user"
            )
            assert example["source_representation"] == (
                "card_hand" if has_card_hand else "conversation"
            )
            assert example["cards"] == (
                ["situation", "data", "rule", "goal"] if has_card_hand else []
            )
            intermediate_assistant_messages = [
                message["content"]
                for message in source["messages"][:-1]
                if message["role"] == "assistant"
            ]
            assert not any(
                message in decoded for message in intermediate_assistant_messages
            )
            assert "For hand " not in decoded
        assert int(supervised.sum()) == metadata["supervised_tokens"]

    package = package_instructions_for_hugging_face(
        tmp_path / "instructions",
        tmp_path / "tokenized",
        tmp_path / "hf",
    )
    assert package["format"] == "complexity-atlas-instruct-hf-package-v1"
    assert (tmp_path / "hf/data/train.parquet").exists()
    assert (tmp_path / "hf/data/validation.parquet").exists()
    assert (tmp_path / "hf/tokenized/o200k/train/input_ids.bin").exists()
    assert (tmp_path / "hf/tokenized/o200k/train/labels.bin").exists()
    assert "No language model generated" in (tmp_path / "hf/README.md").read_text()
    assert "/Users/" not in (tmp_path / "hf/manifest.json").read_text()


def test_sft_target_naturalization_removes_contract_labels() -> None:
    cards = TrainingCards(
        surface="conversational",
        dialogue_state="new_request",
        output="equation_and_check",
        evidence="sufficient",
        reasoning="calculate_then_verify",
        style="plain",
        context_density="focused",
        noise="none",
        uncertainty="answerable",
    )
    messages = [
        {"role": "user", "content": "Calculate it."},
        {
            "role": "assistant",
            "content": (
                "Hand ABCDEF — Equation: 24 / 3 = 8. Total: 8 items per person. "
                "Check: 3 × 8 = 24."
            ),
        },
    ]
    target = _naturalize_assistant_target(
        messages,
        task="reasoning_verification",
        cards=cards,
        example_id="example-1",
    )
    assert "Hand ABCDEF" not in target
    assert "Equation:" not in target
    assert "Total:" not in target
    assert "Check:" not in target
    assert "24 / 3 = 8" in target
    assert "8 items per person" in target
    assert "3 × 8 = 24" in target
    assert "independently, independently" not in target
    assert "because inspect" not in target


def test_grounded_target_starts_with_the_answer_not_a_source_wrapper() -> None:
    cards = TrainingCards(
        surface="direct",
        dialogue_state="new_request",
        output="direct_prose",
        evidence="partial",
        reasoning="locate_then_answer",
        style="concise",
        context_density="focused",
        noise="none",
        uncertainty="state_limits",
    )
    messages = [
        {"role": "user", "content": "What does the source establish?"},
        {
            "role": "assistant",
            "content": (
                "For hand ABCDEF: The documented answer is: The two reports cover "
                "different scopes. The global state is unknown. This is limited to "
                "Source ABCDEF."
            ),
        },
    ]
    target = _naturalize_assistant_target(
        messages,
        task="grounded_qa",
        cards=cards,
        example_id="example-2",
    )
    assert target == (
        "The two reports cover different scopes. The global state is unknown."
    )


def test_explanation_target_preserves_sentence_boundaries() -> None:
    cards = TrainingCards(
        surface="plain",
        dialogue_state="new_request",
        output="direct_prose",
        evidence="sufficient",
        reasoning="explain_then_check",
        style="pedagogical",
        context_density="focused",
        noise="none",
        uncertainty="answerable",
    )
    messages = [
        {"role": "user", "content": "Explain it."},
        {
            "role": "assistant",
            "content": (
                "Hand ABCDEF — Core idea: in plain terms, RAM is temporary. "
                "Example: A saved file remains after restart. "
                "Check: Which copy survives?"
            ),
        },
    ]
    target = _naturalize_assistant_target(
        messages,
        task="explanation_learning",
        cards=cards,
        example_id="example-3",
    )
    assert "RAM is temporary." in target
    assert "A saved file remains after restart." in target
    assert target.endswith("Which copy survives?")


def test_critique_target_is_direct_prose_without_storage_labels() -> None:
    cards = TrainingCards(
        surface="direct",
        dialogue_state="new_request",
        output="revision",
        evidence="sufficient",
        reasoning="critique_then_rewrite",
        style="plain",
        context_density="focused",
        noise="none",
        uncertainty="answerable",
    )
    messages = [
        {"role": "user", "content": "Review the draft."},
        {
            "role": "assistant",
            "content": (
                "Weakness: the claim exceeds the evidence. Revision: Three of five "
                "testers finished sooner. The result does not prove a universal gain."
            ),
        },
    ]
    target = _naturalize_assistant_target(
        messages,
        task="critique_revision",
        cards=cards,
        example_id="example-4",
    )
    assert "Weakness:" not in target
    assert "Revision:" not in target
    assert "three of five" in target.lower()


def test_safety_target_removes_card_contract_labels() -> None:
    cards = TrainingCards(
        surface="direct",
        dialogue_state="new_request",
        output="protective_action",
        evidence="partial",
        reasoning="protect_then_escalate",
        style="plain",
        context_density="focused",
        noise="none",
        uncertainty="state_limits",
    )
    messages = [
        {"role": "user", "content": "What should I do?"},
        {
            "role": "assistant",
            "content": (
                "Immediate action: Do not share the code. "
                "Boundary: The request is unverified. "
                "Escalate through the provider's official support channel."
            ),
        },
    ]
    target = _naturalize_assistant_target(
        messages,
        task="safety_uncertainty",
        cards=cards,
        example_id="example-5",
    )
    assert "Immediate action:" not in target
    assert "Boundary:" not in target
    assert "Do not share the code" in target
    assert "official support channel" in target
