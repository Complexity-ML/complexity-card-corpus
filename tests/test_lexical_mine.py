from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa
import pytest

from complexity_card_corpus.build import file_sha256
from complexity_card_corpus.lexical_mine import (
    audit_source_overlap,
    build_lexical_mine,
)
from complexity_card_corpus.surface_reference import (
    SurfaceStructureAccumulator,
    compare_surface_structures,
)


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    taskmaster = raw / "taskmaster.json"
    taskmaster.write_text(
        json.dumps(
            [
                {
                    "conversation_id": "one",
                    "utterances": [
                        {"text": "Please verify the booking before payment."},
                        {"text": "The booking is pending, so verify it carefully."},
                    ],
                }
            ]
        )
    )
    empathetic = raw / "empathetic.tar.gz"
    fields = [
        "conv_id",
        "utterance_idx",
        "context",
        "prompt",
        "speaker_idx",
        "utterance",
        "selfeval",
        "tags",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(
        [
            {
                "conv_id": "one",
                "utterance_idx": "1",
                "context": "hopeful",
                "prompt": "",
                "speaker_idx": "0",
                "utterance": "I feel hopeful because the work is finally complete.",
                "selfeval": "",
                "tags": "",
            },
            {
                "conv_id": "one",
                "utterance_idx": "2",
                "context": "hopeful",
                "prompt": "",
                "speaker_idx": "1",
                "utterance": "That sounds encouraging. What happens next?",
                "selfeval": "",
                "tags": "",
            },
        ]
    )
    with tarfile.open(empathetic, "w:gz") as archive:
        for split in ("train", "valid", "test"):
            payload = buffer.getvalue().encode()
            info = tarfile.TarInfo(f"empatheticdialogues/{split}.csv")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    registry = tmp_path / "sources.json"
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "sources": [
                    {
                        "dataset_id": "taskmaster-fixture",
                        "kind": "taskmaster_json",
                        "filename": taskmaster.name,
                        "download_url": "https://example.test/taskmaster.json",
                        "source_url": "https://example.test/taskmaster",
                        "revision": "a" * 40,
                        "license": "CC BY 4.0",
                        "sha256": file_sha256(taskmaster),
                    },
                    {
                        "dataset_id": "empathetic-fixture",
                        "kind": "empathetic_tar",
                        "filename": empathetic.name,
                        "download_url": "https://example.test/empathetic.tar.gz",
                        "source_url": "https://example.test/empathetic",
                        "revision": f"artifact-sha256:{file_sha256(empathetic)}",
                        "license": "CC BY-NC 4.0",
                        "sha256": file_sha256(empathetic),
                    },
                ],
            }
        )
    )
    return registry, raw


def test_lexical_mine_keeps_tokens_and_aggregate_stats_only(tmp_path: Path) -> None:
    registry, raw = _fixtures(tmp_path)
    manifest = build_lexical_mine(
        registry,
        raw,
        tmp_path / "mine",
        min_count=1,
        max_capitalized_ratio=1,
    )
    rows = pq.read_table(tmp_path / "mine/lexicon.parquet").to_pylist()

    assert rows
    assert manifest["audit"]["max_retained_ngram_tokens"] == 1
    assert manifest["audit"]["source_text_retained"] is False
    assert manifest["audit"]["release_ready"] is False
    assert all(row["mined_unit"] == "single_normalized_token" for row in rows)
    assert all(" " not in row["token"] for row in rows)
    assert {"verify", "booking", "hopeful"} <= {row["token"] for row in rows}
    serialized = json.dumps(rows)
    assert "verify the booking" not in serialized
    assert "feel hopeful because" not in serialized
    assert set(manifest["audit"]["source_stats"]) == {
        "taskmaster-fixture",
        "empathetic-fixture",
    }
    for stats in manifest["audit"]["source_stats"].values():
        assert 0 < stats["unique_document_rate_estimate"] <= 1.01
        assert 0 < stats["unique_sentence_rate_estimate"] <= 1.01
        assert stats["distinct_counter"] == "linear_counting_2^24_bits"
        assert stats["surface_structure"]["window_tokens"] == 8
        assert stats["surface_structure"]["retained_lexical_ngrams"] is False


def test_surface_structure_profile_compares_abstract_eight_token_shapes() -> None:
    reference = SurfaceStructureAccumulator(window_tokens=8)
    reference.extend(
        [
            "First, the operator should verify the booking before payment.",
            "When the details change, the operator should verify them again.",
        ]
    )
    candidate = SurfaceStructureAccumulator(window_tokens=8)
    candidate.extend(
        [
            "First, the reviewer must compare the evidence before approval.",
            "When the inputs change, the reviewer should compare them again.",
        ]
    )

    report = compare_surface_structures(reference, candidate)

    assert report["window_tokens"] == 8
    assert report["reference"]["eight_token_windows"] > 0
    assert report["candidate"]["eight_token_windows"] > 0
    assert report["eight_token_shape_js_divergence_bits"] < 0.5
    assert 0 <= report["masked_repetition_total_variation"] <= 1
    assert set(report["masked_repetition_level_deltas"]) == {
        "unique",
        "2-4",
        "5-9",
        "10-24",
        "25+",
    }
    assert report["source_text_retained"] is False
    assert report["source_ngrams_retained"] is False


def test_lexical_mine_can_compare_scenarios_without_retaining_phrases(
    tmp_path: Path,
) -> None:
    registry, raw = _fixtures(tmp_path)
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text(
        json.dumps(
            {
                "scenario_id": "one",
                "situation": (
                    "First, the reviewer should verify the booking before payment. "
                    "When the details change, the reviewer should inspect them again."
                ),
            }
        )
        + "\n"
    )

    manifest = build_lexical_mine(
        registry,
        raw,
        tmp_path / "mine-with-comparison",
        min_count=1,
        max_capitalized_ratio=1,
        scenarios_path=scenarios,
    )
    comparison = manifest["audit"]["surface_structure_comparison"]

    assert comparison["candidate"]["sentences"] == 2
    assert comparison["window_tokens"] == 8
    assert comparison["source_ngrams_retained"] is False
    assert "verify the booking" not in json.dumps(comparison)


def test_source_overlap_audit_detects_long_copy_without_retaining_source(
    tmp_path: Path,
) -> None:
    registry, raw = _fixtures(tmp_path)
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text(
        json.dumps(
            {
                "scenario_id": "copied",
                "title": "Original title",
                "trigger": "Please verify the booking before payment and continue.",
                "situation": "Different words here.",
                "goal": "A bounded result.",
            }
        )
        + "\n"
    )
    report = audit_source_overlap(
        registry,
        raw,
        scenarios,
        window_tokens=6,
        fail_on_match=False,
    )

    assert report["passed"] is False
    assert report["matched_scenario_ids"] == ["copied"]
    assert report["source_text_retained"] is False
    assert report["source_window_hashes_retained"] is False
    with pytest.raises(ValueError, match="source overlap"):
        audit_source_overlap(registry, raw, scenarios, window_tokens=6)


def test_source_overlap_audit_accepts_original_language(tmp_path: Path) -> None:
    registry, raw = _fixtures(tmp_path)
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text(
        json.dumps(
            {
                "scenario_id": "original",
                "title": "Independent wording",
                "trigger": "A new constraint changes the local decision boundary.",
                "situation": "The operator needs a careful original response.",
                "goal": "Produce a useful result without reproducing source prose.",
            }
        )
        + "\n"
    )

    report = audit_source_overlap(registry, raw, scenarios, window_tokens=6)
    assert report["passed"] is True
    assert report["matched_scenarios"] == 0


def test_v2_parquet_registry_aggregates_messages_and_deletes_raw(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    dataset_dir = raw / "example__quality-chat"
    dataset_dir.mkdir(parents=True)
    parquet = dataset_dir / "train.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Please compare both options."},
                        {
                            "role": "assistant",
                            "content": "First verify the constraints, then compare them.",
                        },
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "Please compare both options."},
                        {
                            "role": "assistant",
                            "content": "Check the evidence before choosing an option.",
                        },
                    ]
                },
            ]
        ),
        parquet,
    )
    registry = tmp_path / "v2.json"
    registry.write_text(
        json.dumps(
            {
                "version": 2,
                "sources": [
                    {
                        "dataset_id": "example/quality-chat",
                        "kind": "parquet_fields",
                        "source_url": "https://example.test/quality-chat",
                        "revision": "b" * 40,
                        "license": "CC BY 4.0",
                        "origin": "test_fixture",
                        "messages_field": "messages",
                        "allowed_roles": ["user", "assistant"],
                        "artifacts": [
                            {
                                "filename": "data/train.parquet",
                                "download_url": "https://example.test/train.parquet",
                                "sha256": file_sha256(parquet),
                            }
                        ],
                    }
                ],
            }
        )
    )

    manifest = build_lexical_mine(
        registry,
        raw,
        tmp_path / "mine-v2",
        min_count=1,
        max_capitalized_ratio=1,
        delete_raw=True,
    )

    assert manifest["audit"]["source_stats"]["example/quality-chat"][
        "documents"
    ] == 4
    conversation_roles = manifest["audit"]["source_stats"][
        "example/quality-chat"
    ]["conversation_roles"]
    assert set(conversation_roles) == {"assistant", "user"}
    assert conversation_roles["user"]["documents"] == 2
    assert conversation_roles["assistant"]["documents"] == 2
    assert conversation_roles["user"]["question_rate"] == 0.0
    assert conversation_roles["assistant"]["question_rate"] == 0.0
    assert all(
        stats["surface_structure"]["retained_lexical_ngrams"] is False
        for stats in conversation_roles.values()
    )
    assert all(
        stats["surface_structure"]["masked_window_repetition"][
            "lexical_tokens_retained"
        ]
        is False
        for stats in conversation_roles.values()
    )
    user_repetition = conversation_roles["user"]["document_repetition"]
    assistant_repetition = conversation_roles["assistant"]["document_repetition"]
    assert user_repetition["maximum_occurrences"] == 2
    assert user_repetition["repeated_occurrence_share"] == 0.5
    assert user_repetition["levels"]["2-4"]["units"] == 1
    assert assistant_repetition["maximum_occurrences"] == 1
    assert assistant_repetition["levels"]["unique"]["units"] == 2
    assert user_repetition["hashes_retained"] is False
    assert manifest["sources"]["example/quality-chat"]["origin"] == "test_fixture"
    assert not parquet.exists()
