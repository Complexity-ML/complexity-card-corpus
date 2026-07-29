from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from complexity_card_corpus.build import build_corpus
from complexity_card_corpus.mosaic import (
    build_mosaic,
    validate_mosaic_registry,
)
from complexity_card_corpus.mosaic_stream import (
    build_mosaic_shards,
    tokenize_mosaic_shards,
)
from complexity_card_corpus.oasst1 import build_alignment_cards
from complexity_card_corpus.package import package_for_hugging_face
from complexity_card_corpus.source import discover_datasets
from complexity_card_corpus.tokenize import tokenize_documents


ROOT = Path(__file__).parents[1]


def _oasst_row(
    message_id: str,
    parent_id: str | None,
    role: str,
    text: str,
    *,
    rank: int | None = None,
    quality: float = 0.8,
) -> dict:
    return {
        "message_id": message_id,
        "parent_id": parent_id,
        "message_tree_id": "tree-1",
        "role": role,
        "text": text,
        "rank": rank,
        "lang": "en",
        "review_count": 3,
        "review_result": True,
        "deleted": False,
        "synthetic": False,
        "tree_state": "ready_for_export",
        "detoxify": {"toxicity": 0.01},
        "labels": {
            "name": ["quality", "helpfulness", "spam", "pii"],
            "value": [quality, quality, 0.0, 0.0],
            "count": [3, 3, 3, 3],
        },
    }


def test_sources_form_a_valid_graph() -> None:
    datasets = discover_datasets(ROOT / "data/source")
    assert {dataset.metadata.domain for dataset in datasets} == {"computing", "fantasy"}
    assert sum(len(dataset.cards) for dataset in datasets) == 42


def test_oasst1_builds_instruct_and_chat_cards() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        [
            _oasst_row("u1", None, "prompter", "Explain UTF-8."),
            _oasst_row("a-low", "u1", "assistant", "A weaker answer.", rank=1),
            _oasst_row(
                "a1",
                "u1",
                "assistant",
                "UTF-8 encodes Unicode code points.",
                rank=0,
            ),
            _oasst_row("u2", "a1", "prompter", "How many bytes can it use?"),
            _oasst_row(
                "a2",
                "u2",
                "assistant",
                "It uses one to four bytes per code point.",
                rank=0,
            ),
        ]
    )
    cards, rejection_counts = build_alignment_cards(frame, split="train")
    assert [card.mode for card in cards] == ["instruct", "chat"]
    assert cards[0].messages[-1].source_message_id == "a1"
    assert len(cards[1].messages) == 4
    assert rejection_counts["accepted"] == 5


def test_build_is_deterministic_except_timestamp(tmp_path: Path) -> None:
    first = build_corpus(ROOT / "data/source", tmp_path / "first")
    second = build_corpus(ROOT / "data/source", tmp_path / "second")

    assert first["counts"] == second["counts"]
    for name in ("cards", "relations", "documents"):
        assert first["files"][name]["sha256"] == second["files"][name]["sha256"]

    documents = pq.read_table(tmp_path / "first/documents.parquet").to_pylist()
    assert {row["split"] for row in documents} == {"train", "validation"}
    assert {row["template"] for row in documents} == {
        "entity",
        "neighborhood",
        "path",
    }
    assert all(row["text"].strip() for row in documents)


def test_o200k_export_matches_index(tmp_path: Path) -> None:
    build_corpus(ROOT / "data/source", tmp_path / "corpus")
    tokenizer = Path("/Users/boris/Dev/complexity-framework/tokenizer-o200k")
    if not tokenizer.exists():
        return

    manifest = tokenize_documents(
        tmp_path / "corpus/documents.parquet",
        tokenizer,
        tmp_path / "tokenized",
    )
    assert set(manifest["partitions"]) == {"train", "eval"}
    for partition, metadata in manifest["partitions"].items():
        path = tmp_path / "tokenized" / partition / "tokens.bin"
        array = np.memmap(path, mode="r", dtype=np.dtype("<u4"))
        assert len(array) == metadata["num_tokens"]
        assert int(array.max()) == metadata["max_token_id"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
        index = json.loads(
            (tmp_path / "tokenized" / partition / "tokens.idx.json").read_text()
        )
        assert index == metadata

    package = package_for_hugging_face(
        tmp_path / "corpus",
        tmp_path / "tokenized",
        tmp_path / "hf",
    )
    assert (tmp_path / "hf/data/train.parquet").exists()
    assert (tmp_path / "hf/data/validation.parquet").exists()
    assert (tmp_path / "hf/tables/cards_train.parquet").exists()
    assert (tmp_path / "hf/tables/cards_validation.parquet").exists()
    assert (tmp_path / "hf/tokenized/o200k/train/tokens.bin").exists()
    assert package["format"] == "complexity-atlas-pretrain-hf-package-v1"
    packaged_index = json.loads(
        (tmp_path / "hf/tokenized/o200k/train/tokens.idx.json").read_text()
    )
    assert packaged_index["source_documents"] == "data/train.parquet"
    assert packaged_index["tokenizer"] == "tiktoken:o200k_base"
    assert (tmp_path / "hf/LICENSE.md").exists()
    assert "CC BY-NC 4.0" in (tmp_path / "hf/LICENSE.md").read_text()
    assert "/Users/" not in (tmp_path / "hf/manifest.json").read_text()


def test_mosaic_registry_rejects_unknown_license() -> None:
    registry = {
        "format": "complexity-atlas-source-registry-v1",
        "sources": [
            {
                "dataset_id": "unknown",
                "kind": "huggingface_parquet",
                "domain": "test",
                "language": "en",
                "license": "Unknown",
                "repo_id": "owner/repo",
                "revision": "abc123",
                "config": "default",
                "files": ["data.parquet"],
                "text_column": "text",
                "redistribution": True,
            }
        ],
    }
    try:
        validate_mosaic_registry(registry)
    except ValueError as error:
        assert "unsupported license" in str(error)
    else:
        raise AssertionError("Unknown licenses must be rejected")


def test_mosaic_filters_and_retains_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus_root = tmp_path / "atlas"
    build_corpus(ROOT / "data/source", corpus_root)
    external = tmp_path / "external.parquet"
    accepted = (
        "A carefully structured educational story about a clockmaker who "
        "tests every mechanism before recording the result. " * 3
    )
    pq.write_table(
        pa.table(
            {
                "text": [
                    accepted,
                    accepted,
                    "Contact private@example.com for the complete manuscript. " * 5,
                    "short",
                ]
            }
        ),
        external,
    )
    registry = {
        "format": "complexity-atlas-source-registry-v1",
        "sources": [
            {
                "dataset_id": "licensed-test",
                "kind": "huggingface_parquet",
                "domain": "education",
                "language": "en",
                "license": "Apache-2.0",
                "repo_id": "owner/repo",
                "revision": "abc123",
                "config": "stories",
                "files": ["data.parquet"],
                "text_column": "text",
                "minimum_characters": 100,
                "maximum_characters": 10_000,
                "redistribution": True,
            }
        ],
    }
    registry_path = tmp_path / "sources.json"
    registry_path.write_text(json.dumps(registry))
    monkeypatch.setattr(
        "complexity_card_corpus.mosaic.hf_hub_download",
        lambda **_: str(external),
    )

    manifest = build_mosaic(
        registry_path,
        corpus_root / "documents.parquet",
        tmp_path / "raw",
        tmp_path / "mosaic",
        workers=2,
    )
    assert manifest["counts"]["documents_by_source"]["licensed-test"] == 1
    assert manifest["counts"]["rejections"] == {
        "duplicate": 1,
        "email": 1,
        "too_short": 1,
    }
    rows = pq.read_table(tmp_path / "mosaic/documents.parquet").to_pylist()
    external_rows = [row for row in rows if row["dataset_id"] == "licensed-test"]
    assert external_rows[0]["license"] == "Apache-2.0"
    assert external_rows[0]["version"] == "abc123"
    assert external_rows[0]["source_urls"][0].startswith(
        "https://huggingface.co/datasets/owner/repo/blob/abc123/"
    )
    catalog = pq.read_table(tmp_path / "mosaic/sources.parquet").to_pylist()
    assert {row["kind"] for row in catalog} >= {
        "atlas_original",
        "huggingface_parquet",
    }


def test_streaming_mosaic_resumes_and_tokenizes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus_root = tmp_path / "atlas"
    build_corpus(ROOT / "data/source", corpus_root)
    external = tmp_path / "external.parquet"
    pq.write_table(
        pa.table(
            {
                "text": [
                    (
                        f"Lesson {index} explains a distinct mechanism with "
                        "evidence, examples, checks, and a concise conclusion. "
                    )
                    * 8
                    for index in range(32)
                ]
            }
        ),
        external,
    )
    registry = {
        "format": "complexity-atlas-source-registry-v1",
        "sources": [
            {
                "dataset_id": "stream-test",
                "kind": "huggingface_parquet",
                "domain": "education",
                "language": "en",
                "license": "Apache-2.0",
                "repo_id": "owner/repo",
                "revision": "abc123",
                "config": "lessons",
                "files": ["data.parquet"],
                "text_column": "text",
                "minimum_characters": 100,
                "maximum_characters": 10_000,
                "redistribution": True,
            }
        ],
    }
    registry_path = tmp_path / "sources.json"
    registry_path.write_text(json.dumps(registry))
    monkeypatch.setattr(
        "complexity_card_corpus.mosaic_stream.hf_hub_download",
        lambda **_: str(external),
    )
    output = tmp_path / "stream"
    first = build_mosaic_shards(
        registry_path,
        corpus_root / "documents.parquet",
        tmp_path / "raw",
        output,
        validation_per_mille=500,
        workers=2,
        batch_size=8,
    )
    second = build_mosaic_shards(
        registry_path,
        corpus_root / "documents.parquet",
        tmp_path / "raw",
        output,
        validation_per_mille=500,
        workers=2,
        batch_size=8,
    )
    assert first["counts"] == second["counts"]
    assert first["counts"]["source_files"] == 1
    assert len(list((output / "data/train").glob("*.parquet"))) >= 1
    assert len(list((output / "data/validation").glob("*.parquet"))) >= 1

    tokenizer = Path("/Users/boris/Dev/complexity-framework/tokenizer-o200k")
    if not tokenizer.exists():
        return
    tokenized = tokenize_mosaic_shards(
        output,
        tokenizer,
        target_train_tokens=500,
        target_eval_tokens=250,
        workers=2,
        batch_size=4,
    )
    assert tokenized["partitions"]["train"]["num_tokens"] >= 500
    assert tokenized["partitions"]["eval"]["num_tokens"] >= 250
    assert (output / "tokenized/o200k/train/tokens.bin").exists()
