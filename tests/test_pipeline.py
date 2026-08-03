from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from complexity_card_corpus.build import build_corpus
from complexity_card_corpus.package import package_for_hugging_face
from complexity_card_corpus.source import discover_datasets
from complexity_card_corpus.tokenize import load_encoding, tokenize_documents


ROOT = Path(__file__).parents[1]


def test_load_encoding_supports_huggingface_tokenizer_directories(
    tmp_path: Path,
) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            vocab={"</s>": 0, "<unk>": 1, "hello": 2, "world": 3},
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "</s>", "tokenizer_class": "PreTrainedTokenizerFast"})
    )

    encoding, config = load_encoding(tmp_path)

    assert encoding.n_vocab == 4
    assert encoding.eot_token == 0
    assert encoding.encode_single_token("</s>") == 0
    assert encoding.encode("hello world", disallowed_special=()) == [2, 3]
    assert encoding.decode([2, 3, 0]) == "hello world </s>"
    assert config["backend"] == "huggingface_tokenizers"
    assert config["encoding_name"].startswith("hf:")


def test_sources_form_a_valid_graph() -> None:
    datasets = discover_datasets(ROOT / "data/source")
    assert {dataset.metadata.domain for dataset in datasets} == {
        "computing",
        "fantasy",
        "speculative natural history",
    }
    assert sum(len(dataset.cards) for dataset in datasets) == 21_602


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
