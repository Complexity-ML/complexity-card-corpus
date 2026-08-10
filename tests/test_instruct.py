from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from complexity_card_corpus.build import CARD_SCHEMA, DOCUMENT_SCHEMA, RELATION_SCHEMA
from complexity_card_corpus.sft import (
    IGNORE_INDEX,
    build_instruction_dataset,
    load_heldout_evaluation,
    tokenize_instruction_dataset,
)
from complexity_card_corpus.sft.schema import INSTRUCTION_SCHEMA
from complexity_card_corpus.sft.answer_development import develop_answer
from complexity_card_corpus.sft.dialogue_links import preserve_linked_dialogue
from complexity_card_corpus.sft.evaluation import (
    _audit_sft_projection,
    _normalized_opening,
    _text_repetition_signatures,
    assert_sft_opening_diversity,
    assert_sft_repetition_quality,
    audit_sft_opening_diversity,
    audit_sft_repetition_quality,
    filter_sft_repetition_quality,
)
from complexity_card_corpus.sft.tokenization import _load_instruction_sources
from complexity_card_corpus.sft.surface_selection import select_balanced_sft_surfaces
from complexity_card_corpus.sft.surface_variation import SurfaceVariationBalancer
from complexity_card_corpus.sft.language import _inline_sentence
from complexity_card_corpus.sft.projection import (
    _project_sft_conversation,
    _project_sft_exchange,
)
from complexity_card_corpus.sft.selection import (
    _balance_response_card_hands,
    _balance_task_domains,
    _balance_task_families,
    _deduplicate_exact_prompts,
    _deduplicate_exact_responses,
    _deduplicate_structural_rows,
    _normalized_structure,
)
from complexity_card_corpus.sft.target import (
    _apply_semantic_resolution,
    _naturalize_assistant_target,
)
from complexity_card_corpus.training_cards import (
    TrainingCards,
    deal_training_cards,
    natural_dialogue_deck,
)
from complexity_card_corpus.chat_template import (
    CHAT_TEMPLATE_ID,
    render_system_prefix,
)
from complexity_card_corpus.english_morphology import correct_indefinite_articles
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
        "attributes_json": json.dumps(
            {"material": "blue glass", "status": "catalogued"}
        ),
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
    pq.write_table(
        pa.Table.from_pylist(cards, schema=CARD_SCHEMA), root / "cards.parquet"
    )
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
    assert (
        first["files"]["instructions.parquet"]["sha256"]
        == second["files"]["instructions.parquet"]["sha256"]
    )

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


def test_supplementary_conversation_source_is_aliased_and_audited(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    _tiny_corpus(corpus)
    build_instruction_dataset(corpus, tmp_path / "instructions")
    primary = tmp_path / "instructions/instructions.parquet"
    supplement_row = {
        **pq.read_table(primary).to_pylist()[0],
        "example_id": "conversation:supplement",
        "task": "empathetic_dialogue",
        "mode": "chat",
        "messages": [
            {"role": "user", "content": "I feel stuck after this setback."},
            {"role": "assistant", "content": "That sounds discouraging."},
            {"role": "user", "content": "I still want to try once more."},
            {
                "role": "assistant",
                "content": "Choose one small next step and give yourself time to reassess.",
            },
        ],
        "prompt": "I feel stuck after this setback.",
        "response": "Choose one small next step and give yourself time to reassess.",
        "rendered_text": "supplement dialogue",
        "answer_json": json.dumps(
            {"dialogue_stages": ["share_situation", "acknowledge_emotion"]}
        ),
    }
    clarifying_empathy_row = {
        **supplement_row,
        "example_id": "conversation:empathetic-clarification",
        "answer_json": json.dumps(
            {
                "dialogue_stages": [
                    "share_situation",
                    "acknowledge_emotion",
                    "expand_feeling",
                    "invite_detail_without_assumption",
                ]
            }
        ),
    }
    planning_row = {
        **supplement_row,
        "example_id": "conversation:planning-supplement",
        "task": "practical_dialogue",
        "messages": [
            {"role": "user", "content": "Help me compare the two available routes."},
            {
                "role": "assistant",
                "content": "Compare the direct route with the quieter route before choosing.",
            },
        ],
        "prompt": "Help me compare the two available routes.",
        "response": "Compare the direct route with the quieter route before choosing.",
        "rendered_text": "planning supplement dialogue",
        "answer_json": json.dumps(
            {"dialogue_stages": ["state_goal", "present_bounded_options"]}
        ),
    }
    supplement = tmp_path / "supplement.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [supplement_row, clarifying_empathy_row, planning_row],
            schema=INSTRUCTION_SCHEMA,
        ),
        supplement,
    )

    rows, sources, combined_sha256 = _load_instruction_sources(
        primary,
        [supplement],
    )

    loaded = next(row for row in rows if row["example_id"] == "conversation:supplement")
    assert loaded["task"] == "conversation_empathy"
    clarified = next(
        row
        for row in rows
        if row["example_id"] == "conversation:empathetic-clarification"
    )
    assert clarified["task"] == "context_clarification"
    planned = next(
        row for row in rows if row["example_id"] == "conversation:planning-supplement"
    )
    assert planned["task"] == "planning_comparison"
    assert sources[1]["task_aliases"] == {
        "empathetic_dialogue->context_clarification": 1,
        "empathetic_dialogue->conversation_empathy": 1,
        "practical_dialogue->planning_comparison": 1,
    }
    assert len(combined_sha256) == 64


def test_instruction_sources_reject_duplicate_ids(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _tiny_corpus(corpus)
    build_instruction_dataset(corpus, tmp_path / "instructions")
    primary = tmp_path / "instructions/instructions.parquet"

    with pytest.raises(ValueError, match="duplicate id"):
        _load_instruction_sources(primary, [primary])


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
    encoding = load_encoding(tokenizer)[0]
    try:
        eos_id = encoding.encode_single_token(template["eos_token"])
    except AttributeError:
        eos_id = encoding.eot_token
    assert template["id"] == CHAT_TEMPLATE_ID
    assert template["assistant_only_loss"] is True
    assert template["training_projection"] == (
        "naturalize_card_hand_preserve_assistant_turns"
    )
    projected_path = tmp_path / "tokenized/projected.parquet"
    assert projected_path.exists()
    projected_rows = pq.read_table(projected_path).to_pylist()
    row_by_id = {row["example_id"]: row for row in projected_rows}
    assert len(projected_rows) == manifest["total_examples"]
    assert manifest["projected_parquet"]["examples"] == len(projected_rows)
    assert manifest["release_quality"]["checks"]["no_exact_duplicate_train_responses"]
    assert manifest["release_quality"]["checks"]["no_exact_duplicate_train_prompts"]
    assert manifest["release_quality"]["exact_train_response_uniqueness_ratio"] == 1.0
    assert (
        manifest["surface_selection"][
            "post_variation_exact_response_deduplication"
        ]["exact_response_uniqueness_ratio"]
        == 1.0
    )
    assert (
        manifest["surface_selection"][
            "post_variation_exact_prompt_deduplication"
        ]["exact_prompt_uniqueness_ratio"]
        == 1.0
    )
    assert manifest["release_quality"]["target_training_examples"] is None
    assert manifest["release_quality"]["target_supervised_training_tokens"] is None
    assert not any(
        "requested_target" in check
        for check in manifest["release_quality"]["checks"]
    )
    assert (
        "at_least_fourteen_training_families" in manifest["release_quality"]["checks"]
    )
    assert set(manifest["release_quality"]["response_length_bands"]) == {
        "direct_1_25",
        "short_26_45",
        "standard_46_80",
        "extended_81_plus",
    }
    assert {row["split"] for row in projected_rows} == {"train", "validation"}
    assert all("SITUATION CARD" not in row["prompt"] for row in projected_rows)
    assert all("Hand " not in row["response"] for row in projected_rows)
    assert all(
        row["messages"][-1]["content"] == row["response"] for row in projected_rows
    )
    assert any(len(row["messages"]) == 4 for row in projected_rows)
    assert all(
        " CARD" not in message["content"]
        for row in projected_rows
        for message in row["messages"]
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
            "response_order",
            "response_bridge",
            "response_layout",
            "response_opening",
            "natural_opening",
            "natural_link",
            "natural_update",
            "natural_depth",
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
            supervised_mask = local_labels != IGNORE_INDEX
            boundaries = np.diff(
                np.pad(supervised_mask.astype(np.int8), (1, 1))
            )
            supervised_starts = np.flatnonzero(boundaries == 1)
            supervised_ends = np.flatnonzero(boundaries == -1)
            source = source_rows[example["example_id"]]
            assert len(supervised_starts) == len(supervised_ends) == sum(
                message["role"] == "assistant"
                for message in row_by_id[example["example_id"]]["messages"]
            )
            for run_start, run_end in zip(
                supervised_starts,
                supervised_ends,
                strict=True,
            ):
                assert local_labels[run_end - 1] == eos_id
                decoded_target = encoding.decode(
                    local_labels[run_start:run_end].tolist()
                )
                assert "\n\nUser:\n" not in decoded_target
            decoded = load_encoding(tokenizer)[0].decode(local_inputs.tolist())
            assert decoded.startswith(render_system_prefix(template) + "User:\n")
            assert "\n\nAssistant:\n" in decoded
            assert "SITUATION CARD" not in decoded
            assert "DATA CARD" not in decoded
            assert "RULE CARD" not in decoded
            assert "GOAL CARD" not in decoded
            assert "card hand" not in decoded.lower()
            assert example["hand_id"] == source["example_id"]
            assert example["training_representation"] in {
                "natural_instruction",
                "natural_multi_turn",
            }
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
                "response_order",
                "response_bridge",
                "response_layout",
                "response_opening",
                "natural_opening",
                "natural_link",
                "natural_update",
                "natural_depth",
            }
            assert (
                example["response_card_hand"]
                == row_by_id[example["example_id"]]["response_card_hand"]
            )
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
            if not has_card_hand and len(source["messages"]) > 2:
                assert all(
                    message in decoded for message in intermediate_assistant_messages
                )
            else:
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
                "For hand ABCDEF: Source ABCDEF supports this answer: "
                "The documented answer is: The supplied record establishes this: "
                "The two reports cover different scopes. The global state is unknown. "
                "The answer remains limited to Source ABCDEF."
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
    assert "a saved file remains after restart." in target.lower()
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
    assert "do not share the code" in target.lower()
    assert "official support channel" in target


def test_clarification_and_empathy_materialize_visible_layout_cards() -> None:
    cards = TrainingCards(
        surface="direct",
        dialogue_state="new_request",
        output="direct_prose",
        evidence="partial",
        reasoning="resolve_ambiguity",
        style="calm",
        context_density="focused",
        noise="none",
        uncertainty="state_limits",
        response_layout="line_breaks",
    )
    clarification = _naturalize_assistant_target(
        [
            {"role": "user", "content": "Clarify the request."},
            {
                "role": "assistant",
                "content": (
                    "Understood: the requested format is not specified. "
                    "Would you prefer a table or a short paragraph? "
                    "For now, preserve the result without choosing a format."
                ),
            },
        ],
        task="context_clarification",
        cards=cards,
        example_id="example:visible-context-layout",
    )
    empathy = _naturalize_assistant_target(
        [
            {"role": "user", "content": "I keep replaying the mistake."},
            {
                "role": "assistant",
                "content": (
                    "It makes sense that the moment keeps returning to you. "
                    "The replay does not have to produce a perfect explanation tonight. "
                    "You can decide whether to pause or take one small step. "
                    "What would feel most useful right now?"
                ),
            },
        ],
        task="conversation_empathy",
        cards=cards,
        example_id="example:visible-empathy-layout",
    )

    assert clarification.count("\n") == 2
    assert clarification.count("?") == 1
    assert "Understood:" not in clarification
    assert empathy.count("\n") == 3
    assert empathy.count("?") == 1


def test_clarification_supports_additional_visible_layout_cards() -> None:
    messages = [
        {"role": "user", "content": "Which result should I use?"},
        {
            "role": "assistant",
            "content": (
                "My current reading: The request names two possible results. "
                "One point to resolve: Which result should be used? "
                "Until confirmed, use the reversible option."
            ),
        },
    ]
    base = dict(
        surface="direct",
        dialogue_state="new_request",
        output="question_and_default",
        evidence="partial",
        reasoning="resolve_ambiguity",
        style="plain",
        context_density="focused",
        noise="none",
        uncertainty="state_limits",
    )
    spaced = _naturalize_assistant_target(
        messages,
        task="context_clarification",
        cards=TrainingCards(**base, response_layout="spaced_lines"),
        example_id="example:spaced-clarification",
    )
    opening = _naturalize_assistant_target(
        messages,
        task="context_clarification",
        cards=TrainingCards(**base, response_layout="opening_break"),
        example_id="example:opening-break-clarification",
    )
    assert spaced.count("\n\n") == 2
    assert opening.count("\n\n") == 1
    assert spaced.count("?") == opening.count("?") == 1


def test_all_post_training_families_project_to_direct_answers() -> None:
    source = Path("build/post-training/conversations.parquet")
    if not source.exists():
        return
    rows = pq.read_table(source).to_pylist()
    representatives = {}
    for row in rows:
        representatives.setdefault(row["task"], row)
    assert len(representatives) == 14
    forbidden = (
        "hand ",
        "next step:",
        "owner:",
        "timing:",
        "core idea:",
        "example:",
        "check:",
        "decision:",
        "action:",
        "open point:",
        "weakness:",
        "revision:",
        "immediate action:",
        "boundary:",
        "each description states",
        "remain feasible under the stated limits",
        "the response should",
    )
    for task, row in representatives.items():
        _prompt, answer, _cards = _project_sft_exchange(
            row["messages"],
            example_id=row["example_id"],
            task=task,
            answer_json=row["answer_json"],
        )
        lowered = answer.lower()
        assert answer.strip(), task
        assert not any(phrase in lowered for phrase in forbidden), (task, answer)


def test_every_generalist_contract_has_a_direct_projection_without_build() -> None:
    responses = {
        "practical_action": (
            "Next step: ask the office to confirm Thursday. "
            "Owner: the office confirms it. Timing: before noon."
        ),
        "explanation_learning": (
            "Core idea: a cache keeps reused data close. "
            "Example: a browser stores a recent asset. Check: what can be reused?"
        ),
        "troubleshooting": (
            "1. Preserve the log. 2. Change one setting. 3. Repeat the test. "
            "Direct check: confirm that the error is absent. "
            "Regression check: repeat the known-good operation."
        ),
        "writing_transformation": (
            "Meeting A12345 — Decision: review complete. "
            "Action: Mina adds captions. Open item: release remains undecided."
        ),
        "planning_comparison": (
            "Choose A because it meets the limit. Sequence: verify it, then book it. "
            "Fallback trigger: if A fails, compare again."
        ),
        "conversation_empathy": "It makes sense to feel uncertain about the change.",
        "safety_uncertainty": (
            "Immediate action: leave the room. Boundary: do not investigate. "
            "Escalate to emergency services from outside."
        ),
        "grounded_qa": (
            "The documented answer is: The office opens at ten. "
            "This is limited to Source ABC123."
        ),
        "summarization_synthesis": (
            "Decision: keep the case open. Action: Mina checks it tomorrow. "
            "Open point: the cause remains unknown."
        ),
        "extraction_classification": '{"status": "pending"}',
        "reasoning_verification": (
            "Equation: 12 / 3 = 4. Total: 4 items. Check: 4 × 3 = 12."
        ),
        "critique_revision": (
            "Weakness: the claim exceeds the evidence. "
            "Revision: Three testers reported an improvement."
        ),
        "brainstorming_creativity": (
            "1. Shared shelf. 2. Monthly exchange. 3. Request list. "
            "Each description states how the option fits."
        ),
        "context_clarification": (
            "Understood: the format is unknown. Would you prefer a table or prose?"
        ),
    }
    forbidden = (
        "hand ",
        "next step:",
        "owner:",
        "timing:",
        "core idea:",
        "example:",
        "check:",
        "decision:",
        "action:",
        "open point:",
        "weakness:",
        "revision:",
        "immediate action:",
        "boundary:",
        "sequence:",
        "fallback trigger:",
        "each description states",
    )
    for task, response in responses.items():
        _prompt, target, _cards = _project_sft_exchange(
            [
                {"role": "user", "content": "Answer the request."},
                {"role": "assistant", "content": response},
            ],
            example_id=f"unit:{task}",
            task=task,
            answer_json="{}",
        )
        assert target
        assert not any(phrase in target.lower() for phrase in forbidden), (
            task,
            target,
        )


def test_extraction_projection_canonicalizes_json_key_casing_recursively() -> None:
    _prompt, answer, _cards = _project_sft_exchange(
        [
            {"role": "user", "content": "Normalize this record."},
            {
                "role": "assistant",
                "content": (
                    '{"Record_id":"A1","Environment":"iOS",'
                    '"Nested":{"Name":"Rin"}}'
                ),
            },
        ],
        example_id="test:canonical-json",
        task="extraction_classification",
        answer_json="{}",
    )
    assert json.loads(answer) == {
        "record_id": "A1",
        "environment": "iOS",
        "nested": {"name": "Rin"},
    }
    assert answer == (
        '{"record_id":"A1","environment":"iOS","nested":{"name":"Rin"}}'
    )


def test_brainstorm_projection_preserves_content_after_comparison() -> None:
    response = (
        "Candidate set: 1. Fold and Compare. 2. Fraction Match. 3. Missing Piece. "
        "Fit with the brief: All three options fit the paper-only limit. "
        "Comparison result: The alternatives emphasize different strengths. "
        "Select this option: Fold and Compare because equality is directly visible."
    )
    _prompt, target, _cards = _project_sft_exchange(
        [
            {"role": "user", "content": "Propose three activities."},
            {"role": "assistant", "content": response},
        ],
        example_id="brainstorm:preserve-tail",
        task="brainstorming_creativity",
        answer_json="{}",
    )

    assert "The alternatives emphasize different strengths." in target
    assert target.endswith(
        "Select Fold and Compare because equality is directly visible."
    )
    assert "Comparison result:" not in target


@pytest.mark.parametrize(
    "target",
    (
        "The options are feasible. Comparison result:",
        "The recommendation is",
        "The final alternative —",
    ),
)
def test_projection_audit_rejects_incomplete_targets(target: str) -> None:
    with pytest.raises(ValueError, match="model-facing answer is incomplete"):
        _audit_sft_projection(
            [
                {
                    "example_id": "incomplete:target",
                    "task": "brainstorming_creativity",
                    "_projected_target": target,
                }
            ]
        )


def test_troubleshooting_projection_naturalizes_inline_check_label() -> None:
    _prompt, target, _cards = _project_sft_exchange(
        [
            {"role": "user", "content": "Help diagnose the sync failure."},
            {
                "role": "assistant",
                "content": (
                    "1. Preserve the log. 2. In an isolated profile, perform "
                    "this check: Read the remote folder listing without "
                    "modifying it. 3. Compare the new log with the control."
                ),
            },
        ],
        example_id="unit:troubleshooting:inline-check",
        task="troubleshooting",
        answer_json="{}",
    )
    assert "check:" not in target.lower()
    assert "perform this test:" in target.lower()


def test_writing_projection_removes_revised_text_rubric() -> None:
    _prompt, target, _cards = _project_sft_exchange(
        [
            {"role": "user", "content": "Rewrite this public notice."},
            {
                "role": "assistant",
                "content": (
                    "Here is the revised text: The east entrance will be "
                    "closed on day 21 for inspection."
                ),
            },
        ],
        example_id="unit:writing:revised-text",
        task="writing_transformation",
        answer_json="{}",
    )
    assert target == "The east entrance will be closed on day 21 for inspection."


def test_inline_sentence_preserves_a_named_subject() -> None:
    assert _inline_sentence("Mina will finish the review.") == (
        "Mina will finish the review."
    )
    assert _inline_sentence("The review remains open.") == ("the review remains open.")


def test_structural_normalization_deduplicates_slot_variants() -> None:
    first = "Mina should verify case A19 by day 12, then record the result."
    second = "Mina should verify case B72 by day 27, then record the result."
    assert _normalized_structure(first) == _normalized_structure(second)
    rows = [
        {"example_id": "b", "task": "planning_comparison", "target": second},
        {"example_id": "a", "task": "planning_comparison", "target": first},
        {
            "example_id": "c",
            "task": "planning_comparison",
            "target": "Compare both options before choosing one.",
        },
    ]
    kept, audit = _deduplicate_structural_rows(
        rows,
        target_key="target",
        max_per_structure=1,
    )
    assert [row["example_id"] for row in kept] == ["a", "c"]
    assert audit["dropped_structural_duplicates"] == 1


def test_structural_deduplication_preserves_the_same_shape_across_domains() -> None:
    rows = [
        {
            "example_id": "finance",
            "task": "reasoning_verification",
            "domain": "personal_finance",
            "target": "Verify case A19 by day 12, then record the result.",
        },
        {
            "example_id": "travel",
            "task": "reasoning_verification",
            "domain": "travel_time",
            "target": "Verify case B72 by day 27, then record the result.",
        },
    ]

    kept, audit = _deduplicate_structural_rows(rows, target_key="target")

    assert [row["example_id"] for row in kept] == ["finance", "travel"]
    assert audit["dropped_structural_duplicates"] == 0
    assert audit["structural_deduplication_unit"] == (
        "task+domain+response_structure"
    )


def test_structural_deduplication_allows_a_schema_specific_limit() -> None:
    rows = [
        {
            "example_id": f"json-{index}",
            "task": "extraction_classification",
            "target": json.dumps({"case": f"A{index:05d}", "status": "pending"}),
        }
        for index in range(4)
    ]
    kept, audit = _deduplicate_structural_rows(
        rows,
        target_key="target",
        max_per_structure=1,
        per_task_limits={"extraction_classification": 3},
    )
    assert len(kept) == 3
    assert audit["dropped_structural_duplicates"] == 1
    assert audit["maximum_retained_per_structure"] == 3


def test_response_card_balance_caps_a_dominant_hand_without_upsampling() -> None:
    hands = {}
    index = 0
    while len(hands) < 4:
        cards = deal_training_cards(
            task="reasoning_verification",
            mode="instruct",
            example_id=f"balance-hand:{index}",
        )
        hands.setdefault(cards.response_structure_signature, cards)
        index += 1
    cards = list(hands.values())
    rows = [
        {
            "example_id": f"dominant-{item:02d}",
            "task": "reasoning_verification",
            "_conditioning_cards": cards[0],
        }
        for item in range(20)
    ]
    for hand_index, hand in enumerate(cards[1:], start=1):
        rows.extend(
            {
                "example_id": f"minor-{hand_index}-{item:02d}",
                "task": "reasoning_verification",
                "_conditioning_cards": hand,
            }
            for item in range(5)
        )
    kept, audit = _balance_response_card_hands(rows, maximum_share=0.25)
    counts = Counter(
        row["_conditioning_cards"].response_structure_signature for row in kept
    )
    assert len(kept) == 20
    assert set(counts.values()) == {5}
    assert audit["dropped_overrepresented_response_hands"] == 15
    assert audit["tasks"]["reasoning_verification"]["maximum_hand_share_after"] == 0.25


def test_sft_opening_signature_normalizes_slots_and_punctuation() -> None:
    assert _normalized_opening("1. Verify case A192 by day 27, then continue.") == (
        "verify case slot by slot"
    )


def test_sft_opening_quality_gate_accepts_exactly_five_percent() -> None:
    opening_words = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
        "omicron pi rho sigma tau upsilon"
    ).split()
    rows = [
        {
            "example_id": f"balanced:{index:03d}",
            "task": "explanation_learning",
            "_projected_target": (
                f"{opening_words[index // 5]} begins this distinct answer form."
            ),
        }
        for index in range(100)
    ]
    audit = assert_sft_opening_diversity(rows)
    assert audit["passed"] is True
    assert audit["tasks"]["explanation_learning"]["maximum_opening_share"] == 0.05


def test_sft_opening_quality_gate_reports_every_family_above_five_percent() -> None:
    rows = [
        {
            "example_id": f"repeated:{index:03d}",
            "task": "safety_uncertainty",
            "_projected_target": (
                "Act on the immediate risk first, then use the verified channel."
                if index < 14
                else (
                    "opening"
                    + chr(ord("a") + index // 26)
                    + chr(ord("a") + index % 26)
                    + " supports this distinct bounded action."
                )
            ),
        }
        for index in range(100)
    ]
    audit = audit_sft_opening_diversity(rows)
    assert audit["passed"] is False
    assert audit["violations"][0]["task"] == "safety_uncertainty"
    assert audit["violations"][0]["maximum_opening_share"] == 0.14
    with pytest.raises(
        ValueError,
        match=r"safety_uncertainty=14\.00%.*act on the immediate risk",
    ):
        assert_sft_opening_diversity(rows)


def test_sft_opening_quality_gate_exempts_structured_json_contracts() -> None:
    rows = [
        {
            "example_id": f"json:{index:03d}",
            "task": "extraction_classification",
            "_projected_target": '{"item":"A","status":"ready"}',
        }
        for index in range(100)
    ]
    audit = assert_sft_opening_diversity(rows)
    task = audit["tasks"]["extraction_classification"]
    assert task["exempt"] is True
    assert task["audited"] is False


def test_sft_repetition_signatures_cover_edges_sentences_and_internal_spans() -> None:
    signatures = _text_repetition_signatures(
        "User: 1. Verify case A192 before release. Then preserve the signed record.",
        side="response",
    )
    assert signatures["response_opening_3"] == {"verify case slot"}
    assert signatures["response_closing_5"] == {"then preserve the signed record"}
    assert "verify case slot before release" in signatures["response_sentence"]
    assert signatures["response_span_8"] == {
        "verify case slot before release then preserve the",
        "case slot before release then preserve the signed",
        "slot before release then preserve the signed record",
    }


def test_sft_repetition_gate_reports_every_repeated_surface_dimension() -> None:
    rows = []
    for index in range(100):
        marker = chr(ord("a") + index // 26) + chr(ord("a") + index % 26)
        repeated = index < 9
        rows.append(
            {
                "example_id": f"surface:{index:03d}",
                "task": "practical_action",
                "_projected_prompt": (
                    "Use this recurring internal phrase to organize the request. "
                    f"Prompt marker {marker}."
                    if repeated
                    else f"{marker} request."
                ),
                "_projected_target": (
                    f"Answer marker {marker}. "
                    "Use this recurring internal phrase to organize the response."
                    if repeated
                    else f"{marker} answer."
                ),
            }
        )
    audit = audit_sft_repetition_quality(rows)
    violations = {(item["task"], item["dimension"]) for item in audit["violations"]}
    assert ("practical_action", "prompt_span_8") in violations
    assert ("practical_action", "response_sentence") in violations
    with pytest.raises(ValueError, match=r"practical_action\.prompt_span_8=9\.00%"):
        assert_sft_repetition_quality(rows)


def test_sft_repetition_filter_drops_only_overrepresented_compositions() -> None:
    rows = []
    for index in range(120):
        marker = chr(ord("a") + index // 26) + chr(ord("a") + index % 26)
        repeated = index < 12
        rows.append(
            {
                "example_id": f"filtered:{index:03d}",
                "task": "practical_action",
                "_projected_prompt": (
                    "Use this recurring internal phrase to organize the request. "
                    f"Prompt marker {marker}."
                    if repeated
                    else f"{marker} request."
                ),
                "_projected_target": (
                    f"Answer marker {marker}. "
                    "Use this recurring internal phrase to organize the response."
                    if repeated
                    else f"{marker} answer."
                ),
            }
        )

    kept, selection = filter_sft_repetition_quality(rows)
    repeated_kept = sum(
        "recurring internal phrase" in row["_projected_prompt"] for row in kept
    )

    assert 0 < selection["dropped_examples"] < 12
    assert repeated_kept <= int(len(kept) * 0.05)
    assert selection["final_audit"]["passed"] is True, selection["final_audit"][
        "violations"
    ]
    assert [row["example_id"] for row in kept] == [
        row["example_id"] for row in filter_sft_repetition_quality(rows)[0]
    ]


def test_parallel_repetition_filter_matches_serial_selection() -> None:
    rows = []
    for task in ("practical_action", "grounded_qa"):
        for index in range(120):
            marker = chr(ord("a") + index // 26) + chr(ord("a") + index % 26)
            repeated = index < 12
            rows.append(
                {
                    "example_id": f"parallel:{task}:{index:03d}",
                    "task": task,
                    "domain": f"domain-{index % 20}",
                    "_projected_prompt": (
                        "Use this recurring internal phrase to organize the request. "
                        f"Prompt marker {marker}."
                        if repeated
                        else f"{marker} request for {task}."
                    ),
                    "_projected_target": (
                        f"Answer marker {marker}. Use this recurring internal phrase "
                        "to organize the response."
                        if repeated
                        else f"{marker} answer for {task}."
                    ),
                }
            )

    serial, serial_audit = filter_sft_repetition_quality(rows, workers=1)
    parallel, parallel_audit = filter_sft_repetition_quality(rows, workers=4)

    assert [row["example_id"] for row in parallel] == [
        row["example_id"] for row in serial
    ]
    assert parallel_audit["final_audit"] == serial_audit["final_audit"]


def test_sft_repetition_filter_preserves_twenty_domain_subcard_balance() -> None:
    domain_words = (
        "amber birch cedar dune elm fern grove hazel iris jade "
        "kelp linen moss north olive pearl quartz reed sage thistle"
    ).split()
    rows = []
    item_words = ("alpha", "bravo", "cobalt", "delta", "ember", "frost")
    for domain_index, domain in enumerate(domain_words):
        for item_index in range(6):
            common = (
                " Shared guidance remains identical across these selected records."
                if item_index == 0
                else ""
            )
            item_word = item_words[item_index]
            pair_word = f"{domain}{item_word}"
            cards = TrainingCards(
                surface="plain",
                dialogue_state="new_request",
                output="direct_prose",
                evidence="sufficient",
                reasoning="direct_response",
                style="plain",
                context_density="focused",
                noise="none",
                uncertainty="answerable",
                response_order=f"order-{domain}-{item_index}",
                response_bridge=f"bridge-{domain}-{item_index}",
                response_layout=f"layout-{domain}-{item_index}",
                response_opening=f"opening-{domain}-{item_index}",
            )
            rows.append(
                {
                    "example_id": f"domain-balanced:{domain}:{item_index}",
                    "task": "grounded_qa",
                    "domain": domain,
                    "_projected_prompt": (
                        f"{domain} source establishes one bounded local fact."
                        f"{common} Request {pair_word} evidence and close "
                        f"with {pair_word}."
                    ),
                    "_projected_target": (
                        f"{pair_word} answer records distinct local evidence "
                        f"for {pair_word}."
                    ),
                    "_conditioning_cards": cards,
                }
            )

    kept, selection = filter_sft_repetition_quality(rows)
    domain_counts = Counter(row["domain"] for row in kept)

    assert len(kept) == 100
    assert set(domain_counts.values()) == {5}
    assert selection["final_audit"]["passed"] is True, selection["final_audit"][
        "violations"
    ]
    assert selection["tasks"]["grounded_qa"]["domain_ceiling_preserved"] is True


def test_sft_repetition_gate_includes_invisible_response_card_hands() -> None:
    rows = []
    for index in range(100):
        marker = chr(ord("a") + index // 26) + chr(ord("a") + index % 26)
        cards = TrainingCards(
            surface="plain",
            dialogue_state="new_request",
            output="direct_prose",
            evidence="sufficient",
            reasoning="direct_response",
            style="plain",
            context_density="focused",
            noise="none",
            uncertainty="answerable",
            response_order="repeated" if index < 8 else f"order-{marker}",
            response_bridge="plain",
            response_layout="paragraph",
            response_opening="bare",
        )
        rows.append(
            {
                "example_id": f"cards:{index:03d}",
                "task": "grounded_qa",
                "_projected_prompt": f"Prompt marker {marker} asks one fact.",
                "_projected_target": f"Answer marker {marker} gives one fact.",
                "_conditioning_cards": cards,
            }
        )
    audit = audit_sft_repetition_quality(rows)
    hand = audit["tasks"]["grounded_qa"]["dimensions"]["response_card_hand"]
    assert hand["maximum_share"] == 0.08
    assert hand["passed"] is False


def test_sft_repetition_gate_includes_one_card_away_response_siblings() -> None:
    rows = []
    for index in range(100):
        marker = chr(ord("a") + index // 26) + chr(ord("a") + index % 26)
        if index < 8:
            # Eight exact hands remain unique, but differ only by opening.
            order = "idea>example>check"
            bridge = "plain"
            layout = "paragraph"
        else:
            order = f"order-{marker}"
            bridge = f"bridge-{marker}"
            layout = f"layout-{marker}"
        cards = TrainingCards(
            surface="plain",
            dialogue_state="new_request",
            output="direct_prose",
            evidence="sufficient",
            reasoning="direct_response",
            style="plain",
            context_density="focused",
            noise="none",
            uncertainty="answerable",
            response_order=order,
            response_bridge=bridge,
            response_layout=layout,
            response_opening=f"opening-{marker}",
        )
        rows.append(
            {
                "example_id": f"sibling:{index:03d}",
                "task": "explanation_learning",
                "_projected_prompt": f"Prompt marker {marker} asks one question.",
                "_projected_target": f"Answer marker {marker} explains one fact.",
                "_conditioning_cards": cards,
            }
        )

    audit = audit_sft_repetition_quality(rows)
    dimensions = audit["tasks"]["explanation_learning"]["dimensions"]
    assert dimensions["response_card_hand"]["maximum_share"] == 0.01
    sibling = dimensions["response_card_sibling_without_opening"]
    assert sibling["maximum_share"] == 0.08
    assert sibling["passed"] is False


def test_sft_repetition_gate_only_exempts_json_from_prose_shape_checks() -> None:
    rows = [
        {
            "example_id": f"json:{index:03d}",
            "task": "extraction_classification",
            "_projected_prompt": f"Classify record marker {index} into the schema.",
            "_projected_target": f'{{"item":"item-{index}","status":"ready"}}',
        }
        for index in range(100)
    ]
    audit = audit_sft_repetition_quality(rows)
    dimensions = audit["tasks"]["extraction_classification"]["dimensions"]
    assert dimensions["response_opening_3"]["structured_prose_exempt"] is True
    assert dimensions["response_opening_3"]["audited"] is False
    assert dimensions["response_exact"]["structured_prose_exempt"] is False
    assert dimensions["prompt_opening_3"]["audited"] is True


def test_surface_selection_balances_existing_hands_without_new_card_axes() -> None:
    labels = (
        "amber",
        "birch",
        "cedar",
        "dune",
        "elm",
        "fern",
        "grove",
        "hazel",
        "iris",
        "jade",
        "kelp",
        "linen",
        "moss",
        "north",
        "olive",
        "pearl",
        "quartz",
        "reed",
        "sage",
        "thistle",
        "umber",
        "violet",
        "willow",
        "xenia",
        "yarrow",
        "zephyr",
        "acorn",
        "brook",
        "clover",
        "drift",
        "ember",
        "flint",
    )
    rows = [
        {
            "example_id": f"balanced-surface:{index:03d}",
            "task": "grounded_qa",
            "split": "train",
        }
        for index in range(100)
    ]

    def dealer(row, selection_key):
        index = int(selection_key.rsplit(":", 1)[-1])
        label = labels[index]
        return TrainingCards(
            surface="plain",
            dialogue_state="new_request",
            output="direct_prose",
            evidence="sufficient",
            reasoning="direct_response",
            style="plain",
            context_density="focused",
            noise="none",
            uncertainty="answerable",
            response_order=label,
            response_bridge=f"bridge-{label}",
            response_layout=f"layout-{label}",
            response_opening=f"opening-{label}",
        )

    def projector(row, selection_key):
        cards = dealer(row, selection_key)
        label = cards.response_order
        return [
            {
                "role": "user",
                "content": f"{label} request for {row['example_id']}.",
            },
            {
                "role": "assistant",
                "content": f"{label} answer for {row['example_id']}.",
            },
        ], cards

    selected, audit = select_balanced_sft_surfaces(
        rows,
        dealer=dealer,
        projector=projector,
    )
    hands = Counter(
        row["_conditioning_cards"].response_structure_signature for row in selected
    )
    assert len(hands) == 32
    assert max(hands.values()) == 4
    assert audit["method"] == (
        "least_used_response_hand_and_sibling_neighbourhood"
    )
    assert (
        audit["tasks"]["grounded_qa"]["maximum_selected_sibling_share"]
        <= 0.05
    )
    assert audit["new_card_axes"] == 0


def test_surface_variation_balances_existing_language_without_new_cards() -> None:
    balancer = SurfaceVariationBalancer()
    source = [
        {
            "role": "user",
            "content": (
                "Generate three meaningfully different options. Compare their fit "
                "with the stated limits. Select the strongest one."
            ),
        },
        {
            "role": "assistant",
            "content": "The options stay within the supplied brief.",
        },
    ]
    rewritten = [
        balancer.rewrite_messages(
            source,
            task="brainstorming_creativity",
            example_id=f"brainstorm:{index}",
        )[0]["content"]
        for index in range(100)
    ]
    counts = Counter(rewritten)
    assert max(counts.values()) / len(rewritten) <= 0.05
    assert all("three" in item.lower() for item in rewritten)
    audit = balancer.audit()
    assert audit["new_card_axes"] == 0
    assert audit["applications"] == {
        "brainstorming_creativity:user:brainstorm-directive": 100
    }


def test_brainstorm_boundary_variants_preserve_subject_verb_agreement() -> None:
    balancer = SurfaceVariationBalancer()
    source = [
        {"role": "user", "content": "Compare three feasible options."},
        {
            "role": "assistant",
            "content": "The options remain bounded by the explicit brief.",
        },
    ]
    generated = [
        balancer.rewrite_messages(
            source,
            task="brainstorming_creativity",
            example_id=f"brainstorm:agreement:{index}",
        )[1]["content"]
        for index in range(256)
    ]

    invalid = re.compile(
        r"\b(?:every proposal|each idea|the candidate set|the option pool) "
        r"(?:stay|respect|remain|fit)\b",
        flags=re.IGNORECASE,
    )
    assert not any(invalid.search(text) for text in generated)


def test_surface_variation_preserves_dynamic_grounded_request() -> None:
    balancer = SurfaceVariationBalancer()
    messages = balancer.rewrite_messages(
        [
            {
                "role": "user",
                "content": (
                    "Using only Source A1B2, state the recorded year and whether "
                    "the architect is identified."
                ),
            },
            {
                "role": "assistant",
                "content": "No unstated detail is inferred.",
            },
        ],
        task="grounded_qa",
        example_id="grounded:dynamic",
    )
    assert "state the recorded year" in messages[0]["content"].lower()
    assert "architect is identified" in messages[0]["content"].lower()
    assert "No unstated detail is inferred" not in messages[1]["content"]


def test_grounded_response_cards_change_visible_order_and_layout() -> None:
    messages = [
        {"role": "user", "content": "Use only the supplied source."},
        {
            "role": "assistant",
            "content": (
                "The recorded battery life is 18 hours. "
                "Water resistance remains unknown. "
                "Check both fields against the supplied specification."
            ),
        },
    ]
    targets = {
        _naturalize_assistant_target(
            messages,
            task="grounded_qa",
            cards=deal_training_cards(
                task="grounded_qa",
                mode="instruct",
                example_id=f"grounded-visible-hand:{index}",
            ),
            example_id=f"grounded-visible-hand:{index}",
        )
        for index in range(1_024)
    }

    assert len(targets) >= 20
    assert any(target.startswith("Water resistance") for target in targets)
    assert any("\n- " in target or target.startswith("- ") for target in targets)


def test_response_cards_create_many_reasoning_shapes_from_one_answer() -> None:
    messages = [
        {"role": "user", "content": "Calculate it."},
        {
            "role": "assistant",
            "content": (
                "Equation: using the supplied values, 24 / 3 = 8. "
                "Total: this gives 8 items per person. "
                "Check: independently, 3 × 8 = 24."
            ),
        },
    ]
    targets = {
        _naturalize_assistant_target(
            messages,
            task="reasoning_verification",
            cards=deal_training_cards(
                task="reasoning_verification",
                mode="instruct",
                example_id=f"reasoning-shape:{index}",
            ),
            example_id=f"reasoning-shape:{index}",
        )
        for index in range(128)
    }
    assert len(targets) >= 30
    assert all("using the supplied values" not in target.lower() for target in targets)
    assert all("Equation:" not in target for target in targets)


def test_exact_response_deduplication_keeps_one_deterministic_example() -> None:
    rows = [
        {
            "example_id": "b",
            "task": "planning_comparison",
            "_projected_target": "Use route B.",
        },
        {
            "example_id": "a",
            "task": "planning_comparison",
            "_projected_target": "Use route B.",
        },
        {
            "example_id": "c",
            "task": "planning_comparison",
            "_projected_target": "Use route A.",
        },
    ]
    kept, audit = _deduplicate_exact_responses(rows)
    assert [row["example_id"] for row in kept] == ["a", "c"]
    assert audit["dropped_exact_response_duplicates"] == 1
    assert audit["exact_response_uniqueness_ratio"] == 1.0


def test_exact_prompt_deduplication_keeps_one_deterministic_target() -> None:
    rows = [
        {
            "example_id": "b",
            "_projected_prompt": "Choose a route.",
            "_projected_target": "Route B is viable.",
        },
        {
            "example_id": "a",
            "_projected_prompt": "Choose a route.",
            "_projected_target": "Route A is viable.",
        },
        {
            "example_id": "c",
            "_projected_prompt": "Explain the route.",
            "_projected_target": "It is direct.",
        },
    ]
    kept, audit = _deduplicate_exact_prompts(rows)
    assert [row["example_id"] for row in kept] == ["a", "c"]
    assert audit["dropped_exact_prompt_duplicates"] == 1
    assert audit["exact_prompt_uniqueness_ratio"] == 1.0


def test_family_balance_caps_only_dominant_families() -> None:
    rows = [{"example_id": f"a-{index}", "task": "a"} for index in range(7)] + [
        {"example_id": f"b-{index}", "task": "b"} for index in range(2)
    ]
    kept, audit = _balance_task_families(rows, max_examples_per_family=3)
    assert Counter(row["task"] for row in kept) == {"a": 3, "b": 2}
    assert audit["dropped_for_family_balance"] == 4


def test_family_balance_preserves_semantic_domain_coverage() -> None:
    rows = [
        {
            "example_id": f"{domain}-{index}",
            "task": "grounded_qa",
            "domain": domain,
        }
        for domain in ("policy", "science", "travel", "technical")
        for index in range(12)
    ]

    kept, audit = _balance_task_families(rows, max_examples_per_family=20)

    assert Counter(row["domain"] for row in kept) == {
        "policy": 5,
        "science": 5,
        "technical": 5,
        "travel": 5,
    }
    assert audit["selection_strategy"] == "deterministic_domain_round_robin"


def test_domain_balance_caps_a_dominant_domain_when_twenty_are_realized() -> None:
    rows = [
        {
            "example_id": f"dominant-{index:03d}",
            "task": "planning_comparison",
            "domain": "dominant",
        }
        for index in range(100)
    ]
    for domain_index in range(1, 20):
        rows.extend(
            {
                "example_id": f"domain-{domain_index:02d}-{index:03d}",
                "task": "planning_comparison",
                "domain": f"domain_{domain_index:02d}",
            }
            for index in range(10)
        )

    kept, audit = _balance_task_domains(rows, maximum_share=0.05)
    counts = Counter(row["domain"] for row in kept)

    assert len(kept) == 200
    assert max(counts.values()) / len(kept) <= 0.05
    assert audit["dropped_overrepresented_domains"] == 90
    task_audit = audit["tasks"]["planning_comparison"]
    assert task_audit["distinct_domains"] == 20
    assert task_audit["requires_tank_hydration"] is False
    assert task_audit["maximum_domain_share_after"] <= 0.05


def test_domain_balance_reports_when_five_percent_is_mathematically_impossible() -> (
    None
):
    rows = [
        {
            "example_id": f"domain-{domain_index:02d}-{index:03d}",
            "task": "safety_uncertainty",
            "domain": f"domain_{domain_index:02d}",
        }
        for domain_index in range(8)
        for index in range(12 if domain_index == 0 else 4)
    ]

    kept, audit = _balance_task_domains(rows, maximum_share=0.05)
    task_audit = audit["tasks"]["safety_uncertainty"]

    assert len(kept) == 32
    assert task_audit["distinct_domains"] == 8
    assert task_audit["effective_maximum_share"] == 0.125
    assert task_audit["maximum_domain_share_after"] == 0.125
    assert task_audit["requires_tank_hydration"] is True
    assert audit["tasks_requiring_tank_hydration"] == ["safety_uncertainty"]


def test_sft_projection_turns_synthetic_cards_into_linked_dialogue() -> None:
    messages = [
        {
            "role": "user",
            "content": "SITUATION CARD\nA report is late.\n\nDATA CARD\nThe owner is Mara.",
        },
        {"role": "assistant", "content": "I can help."},
        {
            "role": "user",
            "content": "RULE CARD\nDo not invent a deadline.\n\nGOAL CARD\nWrite a concise follow-up.",
        },
        {
            "role": "assistant",
            "content": "Hand ABC123 — Ask Mara for the report and request a confirmed delivery time.",
        },
    ]
    example_id = next(
        f"example:four-turn:{index}"
        for index in range(100)
        if deal_training_cards(
            task="writing_transformation",
            mode="chat",
            example_id=f"example:four-turn:{index}",
        ).natural_depth
        == "linked"
    )
    projected, _cards = _project_sft_conversation(
        messages,
        example_id=example_id,
        task="writing_transformation",
        answer_json="{}",
    )
    assert [message["role"] for message in projected] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert all(" CARD" not in message["content"] for message in projected)
    assert "Hand " not in projected[-1]["content"]
    assert "late" in projected[0]["content"].lower()
    assert "concise follow-up" in projected[2]["content"].lower()
    assert "deadline" in projected[2]["content"].lower()


def test_sft_projection_preserves_clarification_dialogue() -> None:
    messages = [
        {
            "role": "user",
            "content": "SITUATION CARD\nThe requested format is unclear.\n\nDATA CARD\nNo earlier example is available.",
        },
        {"role": "assistant", "content": "I can clarify it."},
        {
            "role": "user",
            "content": "RULE CARD\nDo not guess the format.\n\nGOAL CARD\nAsk one focused question.",
        },
        {
            "role": "assistant",
            "content": "Understood: would you prefer a table or a short paragraph?",
        },
    ]
    example_id = next(
        f"example:clarification-four-turn:{index}"
        for index in range(100)
        if deal_training_cards(
            task="context_clarification",
            mode="chat",
            example_id=f"example:clarification-four-turn:{index}",
        ).natural_depth
        == "linked"
    )
    projected, _cards = _project_sft_conversation(
        messages,
        example_id=example_id,
        task="context_clarification",
        answer_json="{}",
    )
    assert [message["role"] for message in projected] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert all(" CARD" not in message["content"] for message in projected)


def test_sft_projection_keeps_most_card_hands_as_direct_requests() -> None:
    preserved = sum(
        preserve_linked_dialogue(f"example:dialogue-share:{index}")
        for index in range(1_000)
    )

    assert 160 <= preserved <= 240


@pytest.mark.parametrize("task", tuple(natural_dialogue_deck()))
def test_sft_projection_renders_natural_family_link_decks(task: str) -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "SITUATION CARD\nOne bounded decision remains open.\n\n"
                "DATA CARD\nThe supplied record contains one usable fact."
            ),
        },
        {"role": "assistant", "content": "I can work with that."},
        {
            "role": "user",
            "content": (
                "RULE CARD\nDo not invent information.\n\n"
                "GOAL CARD\nProvide one useful result."
            ),
        },
        {"role": "assistant", "content": "Use the recorded fact."},
    ]
    example_id = next(
        f"example:natural-family:{task}:{index}"
        for index in range(100)
        if deal_training_cards(
            task=task,
            mode="chat",
            example_id=f"example:natural-family:{task}:{index}",
        ).natural_depth
        == "linked"
    )

    projected, cards = _project_sft_conversation(
        messages,
        example_id=example_id,
        task=task,
        answer_json=json.dumps({"subject": "the bounded request"}),
    )
    rendered = "\n".join(message["content"] for message in projected)

    assert [message["role"] for message in projected] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert cards.natural_depth == "linked"
    assert " CARD" not in rendered
    assert "{" not in rendered and "}" not in rendered


@pytest.mark.parametrize(
    ("stored_data", "expected_payload"),
    (
        (
            "Learning card: DNS resolves a host name into a network address.",
            "DNS resolves a host name into a network address.",
        ),
        (
            "Learning card ID LR-120: DNS maps a host name to a network address.",
            "DNS maps a host name to a network address.",
        ),
        (
            "Calculation card ID 601367: convert 8 metres to centimetres.",
            "Convert 8 metres to centimetres.",
        ),
        (
            "Creative constraint card 8EF8AC: teach fractions using paper.",
            "Teach fractions using paper.",
        ),
    ),
)
def test_sft_projection_keeps_card_provenance_out_of_model_text(
    stored_data: str,
    expected_payload: str,
) -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "SITUATION CARD\nA bounded example is available.\n\n"
                f"DATA CARD\n{stored_data}\n\n"
                "RULE CARD\nUse only the supplied facts.\n\n"
                "GOAL CARD\nExplain the example clearly."
            ),
        },
        {"role": "assistant", "content": "The supplied example is sufficient."},
    ]

    projected, _cards = _project_sft_conversation(
        messages,
        example_id=f"example:metadata-only:{stored_data}",
        task="explanation_learning",
        answer_json="{}",
    )
    model_text = "\n".join(message["content"] for message in projected)

    assert expected_payload in model_text
    assert "Learning card" not in model_text
    assert "Calculation card" not in model_text
    assert "Creative constraint card" not in model_text


def test_sft_projection_preserves_genuine_non_card_dialogue() -> None:
    messages = [
        {"role": "user", "content": "I am nervous about tomorrow."},
        {"role": "assistant", "content": "That reaction makes sense."},
        {"role": "user", "content": "What is one manageable step?"},
        {
            "role": "assistant",
            "content": "Write down the first task and prepare only what it requires.",
        },
    ]
    projected, _cards = _project_sft_conversation(
        messages,
        example_id="conversation:genuine",
        task="conversation_empathy",
        answer_json="{}",
    )
    assert projected == messages


def test_semantic_projection_preserves_an_authored_non_card_answer() -> None:
    messages = [
        {"role": "user", "content": "Explain the result."},
        {
            "role": "assistant",
            "content": "The total is 12 because three groups of four make 12.",
        },
    ]
    metadata = {
        "scenario_id": "scenario:do-not-surface",
        "subject": "a grouped total",
        "surface_intent": "verify the proposed result",
        "source_state": "A candidate answer is available but has not been checked.",
        "source_constraint": "Confirm the result through a second simple check.",
        "fallback_surface": "Return to a smaller calculation with fewer moving parts.",
        "desired_outcome": "A second method confirms the proposed result.",
        "variant": 3,
    }
    _prompt, target, _cards = _project_sft_exchange(
        messages,
        example_id="example:semantic-resolution",
        task="reasoning_verification",
        answer_json=json.dumps(metadata),
    )
    assert target == "The total is 12 because three groups of four make 12."
    assert "candidate answer" not in target.lower()
    assert "return to a smaller" not in target.lower()
    assert "do-not-surface" not in target
    assert "scenario:" not in target


def test_semantic_resolution_never_pads_a_direct_answer_with_meta_discourse() -> None:
    target = "Paris is the capital of France."
    developed = _apply_semantic_resolution(
        target,
        task="grounded_qa",
        metadata={
            "scenario_id": "scenario:legacy",
            "subject": "France",
            "surface_intent": "answer the question",
            "source_state": "The source is available.",
            "source_constraint": "Use only the source.",
            "fallback_surface": "Return to a smaller causal model.",
            "desired_outcome": "The answer is established.",
            "variant": 3,
        },
        example_id="example:legacy",
    )
    assert developed == target


def test_generic_answer_development_is_disabled_for_every_family() -> None:
    tasks = (
        "context_clarification",
        "conversation_empathy",
        "critique_revision",
        "explanation_learning",
        "grounded_qa",
        "reasoning_verification",
        "summarization_synthesis",
    )
    for task in tasks:
        answer = "Paris is the capital of France."
        assert develop_answer(
            answer,
            task=task,
            metadata={"subject": "France"},
            example_id=f"development:{task}",
        ) == answer


@pytest.mark.parametrize(
    "phrase",
    (
        "The supported takeaway is that the result is bounded.",
        "The response can therefore stay specific: use the stated value.",
        "The supplied numbers give 48.",
        "The supplied material keeps the value open.",
    ),
)
def test_projection_rejects_generic_meta_discourse_even_when_surface_varies(
    phrase: str,
) -> None:
    with pytest.raises(ValueError, match="control rubric"):
        _audit_sft_projection(
            [
                {
                    "example_id": "meta-discourse:0",
                    "task": "grounded_qa",
                    "_projected_target": phrase,
                }
            ]
        )


def test_practical_surface_projection_removes_internal_control_colons() -> None:
    messages = [
        {"role": "user", "content": "Help me organize four coffee orders."},
        {
            "role": "assistant",
            "content": (
                "Use this choice: use a written list with one name per drink. "
                "Then complete the concrete next step: read back each order."
            ),
        },
    ]
    target = _naturalize_assistant_target(
        messages,
        task="practical_action",
        cards=deal_training_cards(
            task="practical_action",
            mode="instruct",
            example_id="conversation:coffee",
        ),
        example_id="conversation:coffee",
    )
    assert target == (
        "Use this choice. Use a written list with one name per drink. "
        "Then read back each order."
    )
    assert "next step:" not in target.lower()


def test_reasoning_projection_preserves_single_letter_variable_a() -> None:
    messages = [
        {"role": "user", "content": "Verify the slot calculation."},
        {
            "role": "assistant",
            "content": (
                "Equation: (6 - 1) + 4 = 9. Total: 9. "
                "Check: A occupies slot 5, immediately before B at slot 6."
            ),
        },
    ]
    target = _naturalize_assistant_target(
        messages,
        task="reasoning_verification",
        cards=deal_training_cards(
            task="reasoning_verification",
            mode="instruct",
            example_id="reasoning:variables",
        ),
        example_id="reasoning:variables",
    )
    assert "An occupies" not in correct_indefinite_articles(target)
    assert "slot 5 is occupied by A" in target


def test_heldout_evaluation_is_separately_authored() -> None:
    path = Path("data/evaluation/generalist-heldout-v1.json")
    rows = load_heldout_evaluation(path)
    assert len(rows) >= 28
    assert {row["task"] for row in rows} == {
        "practical_action",
        "explanation_learning",
        "troubleshooting",
        "writing_transformation",
        "planning_comparison",
        "conversation_empathy",
        "safety_uncertainty",
        "grounded_qa",
        "summarization_synthesis",
        "extraction_classification",
        "reasoning_verification",
        "critique_revision",
        "brainstorming_creativity",
        "context_clarification",
    }
    assert all(row["split"] == "validation" for row in rows)
    assert all(
        json.loads(row["answer_json"])["evaluation_source"] == "separately_authored"
        for row in rows
    )
    assert len({_normalized_structure(row["response"]) for row in rows}) == len(rows)


def test_v2_evaluation_has_700_source_separated_examples() -> None:
    rows = load_heldout_evaluation(Path("data/evaluation/generalist-heldout-v2.json"))
    assert len(rows) == 700
    assert Counter(row["task"] for row in rows) == {
        task: 50
        for task in {
            "practical_action",
            "explanation_learning",
            "troubleshooting",
            "writing_transformation",
            "planning_comparison",
            "conversation_empathy",
            "safety_uncertainty",
            "grounded_qa",
            "summarization_synthesis",
            "extraction_classification",
            "reasoning_verification",
            "critique_revision",
            "brainstorming_creativity",
            "context_clarification",
        }
    }
    assert len({row["prompt"] for row in rows}) == 700
    assert len({row["response"] for row in rows}) == 700
    sources = Counter(
        json.loads(row["answer_json"])["evaluation_source"] for row in rows
    )
    assert sources == {"separately_authored": 28, "source_separated_diagnostic": 672}


def test_tokenization_replaces_generated_validation_with_heldout(
    tmp_path: Path,
) -> None:
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
        heldout_evaluation_path=Path("data/evaluation/generalist-heldout-v2.json"),
    )
    assert manifest["partitions"]["eval"]["examples"] == 28
    assert manifest["train_eval_structure_overlap"] == 0
    assert manifest["heldout_evaluation"]["method"] == (
        "separately_authored_gold_with_diagnostic_companion"
    )
    assert manifest["heldout_evaluation"]["provenance_counts"] == {
        "separately_authored": 28,
        "source_separated_diagnostic": 672,
    }
    projected_rows = pq.read_table(tmp_path / "tokenized/projected.parquet").to_pylist()
    validation_rows = [row for row in projected_rows if row["split"] == "validation"]
    assert len(validation_rows) == 28
    assert all(row["example_id"].startswith("heldout:") for row in validation_rows)
    assert all(
        json.loads(row["answer_json"])["evaluation_source"] == "separately_authored"
        for row in load_heldout_evaluation(
            Path("data/evaluation/generalist-heldout-v2.json")
        )
        if row["example_id"] in {item["example_id"] for item in validation_rows}
    )
    eval_ids = {
        json.loads(line)["example_id"]
        for line in (tmp_path / "tokenized/eval/examples.jsonl")
        .read_text()
        .splitlines()
    }
    assert eval_ids
    assert all(example_id.startswith("heldout:") for example_id in eval_ids)


def test_article_correction_does_not_rewrite_identifier_suffixes() -> None:
    text = "Compare E20939-A or E20939-B before choosing a option."
    assert correct_indefinite_articles(text) == (
        "Compare E20939-A or E20939-B before choosing an option."
    )
