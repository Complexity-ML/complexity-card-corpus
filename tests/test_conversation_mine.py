from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path

import pyarrow.parquet as pq

from complexity_card_corpus.build import file_sha256
from complexity_card_corpus.conversation_mine import build_conversation_mine


def _write_taskmaster(path: Path) -> None:
    payload = [
        {
            "conversation_id": "dlg-test-1",
            "instruction_id": "restaurant-table-2",
            "utterances": [
                {
                    "index": 0,
                    "speaker": "USER",
                    "text": "Please reserve a quiet table for two tonight.",
                    "segments": [
                        {
                            "text": "two",
                            "annotations": [
                                {"name": "restaurant_reservation.num.guests"}
                            ],
                        }
                    ],
                },
                {
                    "index": 1,
                    "speaker": "ASSISTANT",
                    "text": "Which neighborhood would you prefer?",
                },
            ],
        }
    ]
    path.write_text(json.dumps(payload))


def _write_empathetic(path: Path) -> None:
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
    rows = [
        {
            "conv_id": "hit:1_conv:1",
            "utterance_idx": "1",
            "context": "joyful",
            "prompt": "I finally finished the project.",
            "speaker_idx": "1",
            "utterance": "I finally finished the project today.",
            "selfeval": "5|5|5",
            "tags": "",
        },
        {
            "conv_id": "hit:1_conv:1",
            "utterance_idx": "2",
            "context": "joyful",
            "prompt": "I finally finished the project.",
            "speaker_idx": "0",
            "utterance": "That sounds wonderful. How do you feel?",
            "selfeval": "5|5|5",
            "tags": "",
        },
    ]
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    with tarfile.open(path, "w:gz") as archive:
        for split in ("train", "valid", "test"):
            payload = csv_buffer.getvalue().encode()
            info = tarfile.TarInfo(f"empatheticdialogues/{split}.csv")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_conversation_mine_retains_structure_but_no_source_prose(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    taskmaster = raw / "taskmaster.json"
    empathetic = raw / "empathetic.tar.gz"
    _write_taskmaster(taskmaster)
    _write_empathetic(empathetic)
    registry = {
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
                "split": "unassigned",
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
    registry_path = tmp_path / "sources.json"
    registry_path.write_text(json.dumps(registry))

    first = build_conversation_mine(registry_path, raw, tmp_path / "first")
    second = build_conversation_mine(registry_path, raw, tmp_path / "second")

    assert first["files"]["raw_records.parquet"]["sha256"] == second["files"]["raw_records.parquet"]["sha256"]
    assert first["generation_enabled"] is False
    assert first["audit"]["source_text_rows"] == 0
    assert first["audit"]["prose_columns"] == []

    rows = pq.read_table(tmp_path / "first/raw_records.parquet").to_pylist()
    assert len(rows) == 4
    assert all(row["source_text_retained"] is False for row in rows)
    assert {row["corpus_kind"] for row in rows} == {
        "task_oriented",
        "empathetic_conversation",
    }
    task_row = next(row for row in rows if row["corpus_kind"] == "task_oriented")
    assert task_row["speaker_pattern"] == ["user", "assistant"]
    assert task_row["domain"] == "restaurant_reservation"
    assert task_row["slot_types"] == ["num.guests"]
    assert task_row["question_pattern"] == [False, True]
    empathy_row = next(row for row in rows if row["corpus_kind"] == "empathetic_conversation")
    assert empathy_row["emotion"] == "joyful"
    assert empathy_row["speaker_pattern"] == ["speaker_a", "speaker_b"]

    serialized_values = json.dumps(rows, ensure_ascii=False)
    for forbidden in (
        "Please reserve a quiet table",
        "Which neighborhood would you prefer",
        "I finally finished the project today",
        "That sounds wonderful",
    ):
        assert forbidden not in serialized_values
