from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from complexity_card_corpus.conversation_blueprint import build_conversation_blueprints
from complexity_card_corpus.conversation_mine import CONVERSATION_MINE_SCHEMA


def _mine_row(kind: str, category: str, index: int) -> dict:
    task = kind == "task_oriented"
    return {
        "record_id": f"mine:{kind}:{category}:{index}",
        "source_dataset": f"source-{kind}",
        "source_record_id": f"record-{category}-{index}",
        "source_record_sha256": f"{index + 1:064x}",
        "source_revision": "a" * 40,
        "source_url": "https://example.test/source",
        "source_license": "CC BY 4.0" if task else "CC BY-NC 4.0",
        "source_file_sha256": "b" * 64,
        "source_split": "train",
        "corpus_kind": kind,
        "domain": category if task else "everyday_emotion",
        "emotion": "" if task else category,
        "turn_count": 6,
        "speaker_pattern": ["user", "assistant"] * 3,
        "turn_signal_sequence": ["slot", "none", "reject", "none", "slot", "accept"] if task else ["statement", "question"] * 3,
        "slot_types": ["time"] if task else [],
        "question_pattern": [False, True, False, True, False, False],
        "utterance_length_buckets": ["short", "short", "medium", "short", "medium", "short"],
        "source_text_retained": False,
        "extraction_version": "conversation-structure-v1",
    }


def test_blueprints_are_balanced_deterministic_and_text_free(tmp_path: Path) -> None:
    mine = tmp_path / "mine"
    mine.mkdir()
    rows = []
    for category in ("booking", "ordering"):
        rows.extend(_mine_row("task_oriented", category, index) for index in range(3))
    for category in ("joyful", "sad"):
        rows.extend(_mine_row("empathetic_conversation", category, index) for index in range(3))
    pq.write_table(pa.Table.from_pylist(rows, schema=CONVERSATION_MINE_SCHEMA), mine / "raw_records.parquet")

    first = build_conversation_blueprints(mine, tmp_path / "first", seed=7)
    second = build_conversation_blueprints(mine, tmp_path / "second", seed=7)

    assert first["files"]["blueprints.parquet"]["sha256"] == second["files"]["blueprints.parquet"]["sha256"]
    assert first["target_per_kind"] == 6
    assert first["counts"]["by_kind"] == {
        "empathetic_conversation": 6,
        "task_oriented": 6,
    }
    assert all(item["spread"] == 0 for item in first["audit"]["category_balance"].values())
    assert first["audit"]["prose_columns"] == []
    assert first["generation_enabled"] is False

    blueprints = pq.read_table(tmp_path / "first/blueprints.parquet").to_pylist()
    assert len({row["source_structure_id"] for row in blueprints}) == len(blueprints)
    assert all(row["surface_text_generated"] is False for row in blueprints)
    assert all(row["target_speaker_pattern"][0] == "user" for row in blueprints)
    assert all(row["target_speaker_pattern"][-1] == "assistant" for row in blueprints)
    assert all(len(row["dialogue_stages"]) == row["target_turn_count"] for row in blueprints)
    assert all(
        row["target_question_turns"]
        == sum("ask" in stage or "invite" in stage for stage in row["dialogue_stages"])
        for row in blueprints
    )
    assert all(
        row["target_length_pattern"] == ["medium"] * row["target_turn_count"]
        for row in blueprints
    )
