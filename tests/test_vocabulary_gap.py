from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from complexity_card_corpus.vocabulary_gap import build_vocabulary_gap


def _lexical_row(token: str, source: str, occurrences: int, role: str) -> dict:
    return {
        "token": token,
        "role": role,
        "source_dataset": source,
        "source_license": "CC BY 4.0",
        "source_revision": source,
        "occurrences": occurrences,
        "document_count": occurrences,
        "mined_unit": "single_normalized_token",
        "source_text_retained": False,
        "release_ready": False,
        "extraction_version": "fixture",
    }


def test_vocabulary_gap_requires_cross_source_support_and_human_review(
    tmp_path: Path,
) -> None:
    lexicon = tmp_path / "lexicon.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _lexical_row("calibrate", "source-a", 45, "intent_term"),
                _lexical_row("calibrate", "source-b", 31, "vocabulary"),
                _lexical_row("isolated", "source-a", 200, "vocabulary"),
                _lexical_row("verified", "source-a", 80, "state_term"),
                _lexical_row("verified", "source-b", 50, "vocabulary"),
            ]
        ),
        lexicon,
    )
    conversations = tmp_path / "conversations.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Please verify this."},
                        {"role": "assistant", "content": "The result is verified."},
                    ]
                }
            ]
        ),
        conversations,
    )

    manifest = build_vocabulary_gap(
        lexicon,
        conversations,
        tmp_path / "gap",
        min_sources=2,
        min_occurrences_per_source=20,
    )
    with (tmp_path / "gap/vocabulary_review.csv").open() as stream:
        rows = list(csv.DictReader(stream))

    assert [row["token"] for row in rows] == ["calibrate"]
    assert rows[0]["review_status"] == "pending"
    assert rows[0]["source_count"] == "2"
    assert manifest["audit"]["current_observed_vocabulary"] == 7
    assert manifest["audit"]["eligible_cross_source_gaps"] == 1
    assert manifest["audit"]["source_text_retained"] is False
    assert manifest["audit"]["automatic_surface_insertion"] is False
    assert manifest["audit"]["release_ready"] is False
