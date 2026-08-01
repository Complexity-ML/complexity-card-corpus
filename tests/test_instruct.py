from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from complexity_card_corpus.build import CARD_SCHEMA, DOCUMENT_SCHEMA, RELATION_SCHEMA
from complexity_card_corpus.sft import (
    IGNORE_INDEX,
    build_instruction_dataset,
    load_heldout_evaluation,
    tokenize_instruction_dataset,
)
from complexity_card_corpus.sft.language import _inline_sentence
from complexity_card_corpus.sft.projection import (
    _project_sft_conversation,
    _project_sft_exchange,
)
from complexity_card_corpus.sft.selection import (
    _balance_task_families,
    _deduplicate_exact_responses,
    _deduplicate_structural_rows,
    _normalized_structure,
)
from complexity_card_corpus.sft.target import _naturalize_assistant_target
from complexity_card_corpus.training_cards import TrainingCards
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
        "naturalize_card_hand_preserve_assistant_turns"
    )
    projected_path = tmp_path / "tokenized/projected.parquet"
    assert projected_path.exists()
    projected_rows = pq.read_table(projected_path).to_pylist()
    assert len(projected_rows) == manifest["total_examples"]
    assert manifest["projected_parquet"]["examples"] == len(projected_rows)
    assert manifest["release_quality"]["checks"]["no_exact_duplicate_train_responses"]
    assert manifest["release_quality"]["exact_train_response_uniqueness_ratio"] == 1.0
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
    assert "Do not share the code" in target
    assert "official support channel" in target


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
    kept, audit = _deduplicate_structural_rows(rows, target_key="target")
    assert [row["example_id"] for row in kept] == ["a", "c"]
    assert audit["dropped_structural_duplicates"] == 1


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


def test_family_balance_caps_only_dominant_families() -> None:
    rows = [{"example_id": f"a-{index}", "task": "a"} for index in range(7)] + [
        {"example_id": f"b-{index}", "task": "b"} for index in range(2)
    ]
    kept, audit = _balance_task_families(rows, max_examples_per_family=3)
    assert Counter(row["task"] for row in kept) == {"a": 3, "b": 2}
    assert audit["dropped_for_family_balance"] == 4


def test_sft_projection_preserves_a_real_four_turn_conversation() -> None:
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
    projected, _cards = _project_sft_conversation(
        messages,
        example_id="example:four-turn",
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


def test_semantic_projection_uses_authored_cards_without_trace_ids() -> None:
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
    assert "candidate answer" in target.lower()
    assert "second simple check" in target.lower()
    assert "second method" in target.lower()
    assert "do-not-surface" not in target
    assert "scenario:" not in target


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
    assert manifest["partitions"]["eval"]["examples"] == 700
    assert manifest["train_eval_structure_overlap"] == 0
    assert manifest["heldout_evaluation"]["method"] == "mixed_source_separated"
    assert manifest["heldout_evaluation"]["provenance_counts"] == {
        "separately_authored": 28,
        "source_separated_diagnostic": 672,
    }
    projected_rows = pq.read_table(tmp_path / "tokenized/projected.parquet").to_pylist()
    validation_rows = [row for row in projected_rows if row["split"] == "validation"]
    assert len(validation_rows) == 700
    assert all(row["example_id"].startswith("heldout:") for row in validation_rows)
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
