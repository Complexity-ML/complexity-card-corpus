from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from complexity_card_corpus.package import package_sft_views_for_hugging_face


def test_sft_hub_views_share_one_physical_projection(tmp_path: Path) -> None:
    projected = pa.table(
        {
            "example_id": [f"example-{index}" for index in range(11)],
            "task": ["grounded_qa"] * 11,
            "mode": ["chat"] * 5 + ["instruct"] * 6,
            "difficulty": ["medium"] * 11,
            "domain": ["general"] * 11,
            "language": ["en"] * 11,
            "split": ["train"] * 5
            + ["train"] * 4
            + ["validation", "diagnostic"],
            "messages": [
                [
                    {"role": "user", "content": f"prompt {index}"},
                    {"role": "assistant", "content": f"response {index}"},
                ]
                for index in range(11)
            ],
            "prompt": [f"prompt {index}" for index in range(11)],
            "response": [f"response {index}" for index in range(11)],
            "structure_signature": [f"signature-{index}" for index in range(11)],
            "response_card_hand": [f"hand-{index}" for index in range(11)],
            "source_representation": ["original_cards"] * 11,
            "source": ["Complexity original"] * 11,
            "license": ["CC BY-NC 4.0"] * 11,
            "version": ["1.0.11"] * 11,
        }
    )
    projected_path = tmp_path / "projected.parquet"
    pq.write_table(projected, projected_path)

    output = tmp_path / "hf"
    output.mkdir()
    (output / "README.md").write_text(
        """---
language:
- en
configs:
- config_name: sft
  data_files:
  - split: train
    path: data/sft-v11/train-*.parquet
---

# Test dataset
"""
    )
    (output / "release.json").write_text(
        json.dumps({"release": "v1.0.11"}) + "\n"
    )

    release = package_sft_views_for_hugging_face(
        projected_path,
        output,
        release_slug="sft-v11",
        max_rows_per_shard=3,
        row_group_size=2,
    )

    assert release["examples"] == 11
    assert release["views"] == {
        "chat": {"train": 5},
        "instruct": {"train": 4, "validation": 1, "diagnostic": 1},
    }
    assert len(list((output / "data/sft-v11/chat").glob("train-*.parquet"))) == 2
    assert (
        len(list((output / "data/sft-v11/instruct").glob("train-*.parquet")))
        == 2
    )
    assert (output / "data/sft-v11/instruct/validation.parquet").exists()
    assert (output / "data/sft-v11/instruct/diagnostic.parquet").exists()

    physical_rows = sum(
        pq.ParquetFile(path).metadata.num_rows
        for path in (output / "data/sft-v11").rglob("*.parquet")
    )
    assert physical_rows == len(projected)

    readme = (output / "README.md").read_text()
    assert "- config_name: sft" not in readme
    assert "- config_name: chat" in readme
    assert "path: data/sft-v11/chat/train-*.parquet" in readme
    assert "- config_name: instruct" in readme
    assert "path: data/sft-v11/instruct/train-*.parquet" in readme

    sample = next((output / "data/sft-v11/chat").glob("train-*.parquet"))
    chat_columns = pq.read_schema(sample).names
    assert chat_columns == [
        "task",
        "difficulty",
        "domain",
        "domain_group",
        "language",
        "messages",
        "structure_signature",
        "response_card_hand",
        "source_representation",
        "source",
        "license",
        "version",
    ]
    instruct_columns = pq.read_schema(
        output / "data/sft-v11/instruct/validation.parquet"
    ).names
    assert instruct_columns == [
        "task",
        "difficulty",
        "domain",
        "domain_group",
        "language",
        "prompt",
        "response",
        "structure_signature",
        "response_card_hand",
        "source_representation",
        "source",
        "license",
        "version",
    ]
    assert "mode" not in chat_columns
    assert "split" not in chat_columns
    assert "example_id" not in chat_columns
    assert "example_id" not in instruct_columns
    assert set(pq.read_table(sample)["domain_group"].to_pylist()) == {
        "general_cross_domain"
    }
    metadata = pq.ParquetFile(sample).metadata.row_group(0).column(0)
    assert metadata.statistics is not None
    assert metadata.has_column_index
    assert metadata.has_offset_index
