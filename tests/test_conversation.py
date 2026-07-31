from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from complexity_card_corpus.conversation import FAMILIES, build_conversation_dataset


def test_conversation_dataset_is_small_model_focused_and_deterministic(tmp_path: Path) -> None:
    first = build_conversation_dataset(tmp_path / "first", examples=160, seed=7)
    second = build_conversation_dataset(tmp_path / "second", examples=160, seed=7)

    assert first["files"]["instructions.parquet"]["sha256"] == second["files"]["instructions.parquet"]["sha256"]
    assert first["files"]["raw_records.parquet"]["sha256"] == second["files"]["raw_records.parquet"]["sha256"]
    assert set(first["counts"]["examples_by_task"]) == set(FAMILIES)
    assert first["generation"]["external_instruction_datasets"] == []
    assert first["generation"]["model_generated"] is False

    rows = pq.read_table(tmp_path / "first/instructions.parquet").to_pylist()
    raw = pq.read_table(tmp_path / "first/raw_records.parquet").to_pylist()
    assert len(rows) == len(raw) == 160
    assert len({row["example_id"] for row in rows}) == 160
    assert len({row["rendered_text"] for row in rows}) == 160
    assert {row["domain"] for row in rows} == {"general_conversation"}
    assert {row["difficulty"] for row in rows} == {"easy"}
    assert {row["split"] for row in rows} == {"train", "validation"}
    assert all(len(row["messages"]) in {2, 4} for row in rows)
    assert all(len(row["response"]) <= 700 for row in rows)
    assert all(row["source_urls"] == [] for row in rows)
    assert all(row["license"] == "CC BY-NC 4.0" for row in rows)

    practical = next(row for row in rows if row["task"] == "practical_help")
    assert "\n1. " in practical["response"]
    assert "\n2. " in practical["response"]
    assert "\n3. " in practical["response"]

    text = "\n".join(row["rendered_text"].lower() for row in rows)
    for excluded in ("fantasy", "atlas", "sql", "python code", "minecraft"):
        assert excluded not in text


def test_conversation_raw_records_are_normalized_and_split_consistently(tmp_path: Path) -> None:
    build_conversation_dataset(tmp_path / "corpus", examples=80, seed=42)
    rows = pq.read_table(tmp_path / "corpus/instructions.parquet").to_pylist()
    raw = pq.read_table(tmp_path / "corpus/raw_records.parquet").to_pylist()
    raw_by_id = {row["record_id"]: row for row in raw}

    for row in rows:
        record_id = row["source_keys"][0]
        source = raw_by_id[record_id]
        assert row["split"] == source["split"]
        assert row["answer_json"] == source["fields_json"]
        assert isinstance(json.loads(source["fields_json"]), dict)
