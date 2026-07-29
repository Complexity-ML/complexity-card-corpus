from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq

from .build import file_sha256

DATASET_CARD = """---
language:
- en
license: other
pretty_name: Complexity Card Corpus
task_categories:
- text-generation
configs:
- config_name: documents
  data_files:
  - split: train
    path: data/train.parquet
  - split: validation
    path: data/validation.parquet
- config_name: cards
  data_files:
  - split: train
    path: tables/cards_train.parquet
  - split: validation
    path: tables/cards_validation.parquet
- config_name: relations
  data_files:
  - split: train
    path: tables/relations_train.parquet
  - split: validation
    path: tables/relations_validation.parquet
{alignment_configs}
---

# Complexity Card Corpus

An English, multi-domain corpus compiled from linked knowledge cards.

The `documents` configuration is intended for language-model corpus inspection.
`cards` and `relations` preserve the normalized source graph. Derived o200k
token streams are available under `tokenized/o200k/`.
{alignment_section}

## Current status

This is a private pilot dataset. It validates the schema, graph rendering,
Parquet packaging and token-shard pipeline; it is not yet large enough to serve
as a standalone pretraining corpus.

## Splits

Splits are assigned at source-dataset level, so connected cards from one graph
are not divided between training and validation.

## Tokenization

The derived shards use the `o200k_base` tokenizer, append the end-of-text token
after every document and store little-endian `uint32` token IDs. Parquet remains
the canonical, tokenizer-independent artifact.

## License

No public redistribution license has been granted for the current source
content. Keep this dataset private until a data license is selected.
"""

ALIGNMENT_CONFIGS = """- config_name: instruct
  data_files:
  - split: train
    path: alignment/instruct_train.parquet
  - split: validation
    path: alignment/instruct_validation.parquet
- config_name: chat
  data_files:
  - split: train
    path: alignment/chat_train.parquet
  - split: validation
    path: alignment/chat_validation.parquet"""

ALIGNMENT_SECTION = """

## Alignment cards

The optional `instruct` configuration contains one-turn user/assistant pairs.
The `chat` configuration contains selected multi-turn paths. Both are filtered
English subsets derived from the human-authored OpenAssistant OASST1 trees and
retain source IDs, quality scores, the pinned revision and Apache-2.0
provenance.
"""


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination)


def _write_splits(
    table,
    destination: Path,
    *,
    stem: str,
) -> None:
    for split in ("train", "validation"):
        split_table = table.filter(pc.equal(table["split"], split))
        if len(split_table) == 0:
            continue
        pq.write_table(
            split_table,
            destination / f"{stem}_{split}.parquet",
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )


def package_for_hugging_face(
    corpus_root: Path,
    tokenized_root: Path,
    output_root: Path,
    *,
    alignment_root: Path | None = None,
) -> dict[str, Any]:
    corpus_manifest = json.loads((corpus_root / "manifest.json").read_text())
    tokenized_manifest = json.loads((tokenized_root / "manifest.json").read_text())

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    data_root = temporary / "data"
    table_root = temporary / "tables"
    data_root.mkdir()
    table_root.mkdir()
    alignment_manifest = None

    documents = pq.read_table(corpus_root / "documents.parquet")
    for split, filename in (("train", "train.parquet"), ("validation", "validation.parquet")):
        split_table = documents.filter(pc.equal(documents["split"], split))
        if len(split_table):
            pq.write_table(
                split_table,
                data_root / filename,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )

    _write_splits(
        pq.read_table(corpus_root / "cards.parquet"),
        table_root,
        stem="cards",
    )
    _write_splits(
        pq.read_table(corpus_root / "relations.parquet"),
        table_root,
        stem="relations",
    )

    if alignment_root is not None:
        alignment_manifest = json.loads((alignment_root / "manifest.json").read_text())
        alignment_table = pq.read_table(alignment_root / "alignment.parquet")
        alignment_output = temporary / "alignment"
        alignment_output.mkdir()
        for mode in ("instruct", "chat"):
            mode_table = alignment_table.filter(pc.equal(alignment_table["mode"], mode))
            _write_splits(mode_table, alignment_output, stem=mode)

    _copy_tree(tokenized_root, temporary / "tokenized" / "o200k")
    (temporary / "README.md").write_text(
        DATASET_CARD.format(
            alignment_configs=ALIGNMENT_CONFIGS if alignment_root else "",
            alignment_section=ALIGNMENT_SECTION if alignment_root else "",
        )
    )

    files = {
        str(path.relative_to(temporary)): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(temporary.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    package_manifest = {
        "format": "complexity-card-corpus-hf-package-v1",
        "corpus": corpus_manifest,
        "tokenized": tokenized_manifest,
        "alignment": alignment_manifest,
        "files": files,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(package_manifest, indent=2, sort_keys=True) + "\n"
    )

    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return package_manifest
