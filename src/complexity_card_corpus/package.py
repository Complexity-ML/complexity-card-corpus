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
license: cc-by-nc-4.0
pretty_name: Complexity Atlas Pretrain
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
---

# Complexity Atlas Pretrain

An English, multi-domain corpus compiled from linked knowledge cards.

The `documents` configuration is intended for language-model corpus inspection.
`cards` and `relations` preserve the normalized source graph. Derived o200k
token streams are available under `tokenized/o200k/`.

## Included pilot worlds

The package includes computing and fantasy knowledge graphs plus
**Prismwilds**, an original creature atlas connecting peculiar wildlife,
habitats, research guilds, field relics, natural phenomena and creature food.
The Grand Codex contributes 10,000 generated creatures to the inspectable
linked-card graph. Prismwilds does not reuse the characters, names, designs or
lore of an existing franchise.

The **Aethoria Grand Archive** contributes another 10,000 original fantasy
characters connected to locations, factions, artifacts, rituals and omens.

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

The original Complexity cards, schemas, deterministic graph renderings and
derived artifacts are offered under CC BY-NC 4.0. Third-party datasets are
excluded from this package.
"""

PRETRAIN_DATA_LICENSE = """# Complexity Atlas Pretrain data license

The original Complexity card schema, editorial selection, curation and
original rendered prose are offered under the Creative Commons
Attribution-NonCommercial 4.0 International license (CC BY-NC 4.0), to the
extent that those contributions are copyrightable.

Attribution: **Complexity — Complexity Atlas Pretrain**

License: <https://creativecommons.org/licenses/by-nc/4.0/>

This package contains only Complexity-authored dataset material and
deterministic derivatives of that material. Third-party corpora are excluded.
No warranty is provided.
"""

INSTRUCT_DATASET_CARD = """---
language:
- en
license: cc-by-nc-4.0
pretty_name: Complexity Atlas Instruct
task_categories:
- text-generation
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
  - split: validation
    path: data/validation.parquet
---

# Complexity Atlas Instruct

An original English instruction-tuning dataset generated deterministically
from Complexity linked knowledge cards. It contains grounded entity lookup,
attribute questions, recorded facts, direct relations, multi-hop paths,
record comparisons, structured JSON extraction and multi-turn follow-ups.

Every row includes the exact `source_keys` and `evidence` used to construct the
assistant answer. No language model generated the prompts or responses.

## Splits

Complete source decks are assigned to either train or validation. Connected
cards from one knowledge graph never cross the split boundary.

## SFT shards

`tokenized/o200k/{train,eval}/input_ids.bin` stores little-endian uint32 model
inputs. The aligned `labels.bin` stores little-endian int32 causal targets and
uses `-100` for user prompts, role prefixes and padding. Loss must be computed
only where labels differ from `-100`. `examples.jsonl` records the offset and
length of each independent example.

Serialization follows the bundled `chat_template.json` contract
(`complexity-chat-v1`). It renders a fixed system instruction followed by
`User:\n<content>\n\nAssistant:\n`; only assistant content and EOS are
supervised. Training and inference must use this exact same contract.

## Intended use

This is a small, domain-focused SFT corpus. Use it after pretraining and measure
both held-out Atlas instruction performance and general-language regressions.
It is not a replacement for broad instruction tuning or safety evaluation.

## License

Original cards, deterministic templates and derived conversations are offered
under CC BY-NC 4.0. Attribution: **Complexity — Complexity Atlas Instruct**.
"""

INSTRUCT_LICENSE = """# Complexity Atlas Instruct license

The original Complexity linked cards, deterministic instruction templates,
curation and derived conversations in this package are offered under the
Creative Commons Attribution-NonCommercial 4.0 International license.

Attribution: **Complexity — Complexity Atlas Instruct**

License: <https://creativecommons.org/licenses/by-nc/4.0/>

No warranty is provided.
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


def _package_files(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _portable_pretrain_manifests(
    corpus_manifest: dict[str, Any],
    tokenized_manifest: dict[str, Any],
    tokenized_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    corpus_manifest = json.loads(json.dumps(corpus_manifest))
    tokenized_manifest = json.loads(json.dumps(tokenized_manifest))
    corpus_manifest["source_root"] = "tables/"

    for partition, metadata in tokenized_manifest["partitions"].items():
        split_path = (
            "data/train.parquet"
            if partition == "train"
            else "data/validation.parquet"
        )
        metadata["source_documents"] = split_path
        metadata["tokenizer"] = "tiktoken:o200k_base"
        index_path = tokenized_output / partition / "tokens.idx.json"
        index_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    tokenized_manifest["tokenizer"] = "tiktoken:o200k_base"
    (tokenized_output / "manifest.json").write_text(
        json.dumps(tokenized_manifest, indent=2, sort_keys=True) + "\n"
    )
    return corpus_manifest, tokenized_manifest


def package_for_hugging_face(
    corpus_root: Path,
    tokenized_root: Path,
    output_root: Path,
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

    tokenized_output = temporary / "tokenized" / "o200k"
    _copy_tree(tokenized_root, tokenized_output)
    corpus_manifest, tokenized_manifest = _portable_pretrain_manifests(
        corpus_manifest,
        tokenized_manifest,
        tokenized_output,
    )
    (temporary / "README.md").write_text(DATASET_CARD)
    (temporary / "LICENSE.md").write_text(PRETRAIN_DATA_LICENSE)

    files = _package_files(temporary)
    package_manifest = {
        "format": "complexity-atlas-pretrain-hf-package-v1",
        "corpus": corpus_manifest,
        "tokenized": tokenized_manifest,
        "files": files,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(package_manifest, indent=2, sort_keys=True) + "\n"
    )

    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return package_manifest


def package_instructions_for_hugging_face(
    instructions_root: Path,
    tokenized_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    instruction_manifest = json.loads(
        (instructions_root / "manifest.json").read_text()
    )
    tokenized_manifest = json.loads((tokenized_root / "manifest.json").read_text())
    table = pq.read_table(instructions_root / "instructions.parquet")

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    data_root = temporary / "data"
    data_root.mkdir()
    for split in ("train", "validation"):
        split_table = table.filter(pc.equal(table["split"], split))
        if len(split_table):
            pq.write_table(
                split_table,
                data_root / f"{split}.parquet",
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )

    _copy_tree(tokenized_root, temporary / "tokenized" / "o200k")
    portable_instruction_manifest = json.loads(json.dumps(instruction_manifest))
    portable_instruction_manifest["source_corpus"].pop("path", None)
    (temporary / "README.md").write_text(INSTRUCT_DATASET_CARD)
    (temporary / "LICENSE.md").write_text(INSTRUCT_LICENSE)
    files = _package_files(temporary)
    package_manifest = {
        "format": "complexity-atlas-instruct-hf-package-v1",
        "instructions": portable_instruction_manifest,
        "tokenized": tokenized_manifest,
        "files": files,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(package_manifest, indent=2, sort_keys=True) + "\n"
    )
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return package_manifest
