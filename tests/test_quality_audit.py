from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from complexity_card_corpus.quality_audit import (
    audit_dataset_quality,
    audit_rows_quality,
)


def test_sklearn_audit_detects_source_group_leakage(tmp_path) -> None:
    rows = []
    for index in range(48):
        split = "train" if index < 36 else "validation"
        source_key = "scenario:leak" if index in {0, 40} else f"scenario:{index}"
        rows.append(
            {
                "example_id": f"example:{index:03d}",
                "task": "planning_comparison" if index % 2 else "grounded_qa",
                "split": split,
                "prompt": f"Question {index} about a distinct operational detail.",
                "response": f"Answer {index} preserves the distinct documented result.",
                "source_keys": [source_key],
            }
        )
    conversations = tmp_path / "conversations.parquet"
    output = tmp_path / "audit.json"
    pq.write_table(pa.Table.from_pylist(rows), conversations)

    audit = audit_dataset_quality(
        conversations,
        output,
        sample_size=48,
        max_features=2_000,
        workers=1,
    )

    assert audit["split_leakage"]["source_group_overlap_count"] == 1
    assert not audit["checks"]["no_source_group_leakage"]
    assert json.loads(output.read_text())["rows"] == 48


def test_basic_format_checks_count_every_row_while_bounding_the_preview() -> None:
    rows = [
        {
            "example_id": f"malformed:{index:03d}",
            "task": "grounded_qa",
            "split": "train",
            "prompt": f"Distinct prompt number {index} with enough surface text.",
            "response": "No",
            "source_keys": [f"scenario:{index}"],
        }
        for index in range(150)
    ]

    audit = audit_rows_quality(
        rows,
        input_label="format coverage regression",
        sample_size=150,
        max_features=2_000,
        cluster_count=8,
        workers=1,
    )

    assert audit["format_checks"]["coverage"] == "all_rows"
    assert audit["format_checks"]["malformed_count"] == 150
    assert len(audit["format_checks"]["malformed_preview"]) == 100
    assert not audit["checks"]["no_basic_format_failures"]


def test_basic_format_checks_accept_compact_json_responses() -> None:
    rows = [
        {
            "example_id": f"structured:{index:03d}",
            "task": "extraction_classification",
            "split": "train",
            "prompt": f"Extract the documented fields from record {index}.",
            "response": json.dumps({"record": index, "status": "valid"}, separators=(",", ":")),
            "source_keys": [f"scenario:{index}"],
        }
        for index in range(48)
    ]

    audit = audit_rows_quality(
        rows,
        input_label="structured response regression",
        sample_size=48,
        max_features=2_000,
        cluster_count=8,
        workers=1,
    )

    assert audit["format_checks"]["malformed_count"] == 0
    assert audit["checks"]["no_basic_format_failures"]
