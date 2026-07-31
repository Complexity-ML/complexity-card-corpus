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
{alignment_configs}
---

# Complexity Atlas Pretrain

An English, multi-domain corpus compiled from linked knowledge cards.

The `documents` configuration is intended for language-model corpus inspection.
`cards` and `relations` preserve the normalized source graph. Derived o200k
token streams are available under `tokenized/o200k/`.
{alignment_section}

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

This dataset uses layered licensing:

- original Complexity schemas, editorial curation and original rendered prose
  are offered under CC BY-NC 4.0;
- imported source fields retain the license stated on each row, including CC0
  1.0 where applicable;
- source facts that are already available under CC0 remain reusable under
  CC0, including commercially.

See `LICENSE.md` for the full notice and use the row-level `license`, `source`
and `source_urls` fields when reusing individual records.
"""

PRETRAIN_DATA_LICENSE = """# Complexity Atlas Pretrain layered data license

This package combines material with different rights and licenses. No license
in this notice replaces the license or public-domain status of imported source
material.

## Complexity contributions

The original Complexity card schema, editorial selection, curation and
original rendered prose are offered under the Creative Commons
Attribution-NonCommercial 4.0 International license (CC BY-NC 4.0), to the
extent that those contributions are copyrightable.

Attribution: **Complexity — Complexity Atlas Pretrain**

License: <https://creativecommons.org/licenses/by-nc/4.0/>

## Imported source material

Imported fields retain their row-level source and license. Smithsonian Open
Access metadata and any other field explicitly marked `CC0 1.0` remain
available under CC0 1.0, including for commercial reuse. Complexity does not
apply the non-commercial restriction to those pre-existing source elements.

CC0 1.0: <https://creativecommons.org/publicdomain/zero/1.0/>

The technical specifications linked as references in source metadata remain
the property of their respective authors. Their links identify factual
references; those third-party documents are not redistributed in this dataset.
Other rights such as trademarks, privacy, publicity or moral rights may still
apply. No warranty is provided.
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

POSTTRAIN_DATASET_CARD = """---
language:
- en
license: apache-2.0
pretty_name: Complexity Atlas Posttrain
task_categories:
- text-generation
source_datasets:
- OpenAssistant/oasst1
configs:
- config_name: instruct
  data_files:
  - split: train
    path: instruct/train.parquet
  - split: validation
    path: instruct/validation.parquet
- config_name: chat
  data_files:
  - split: train
    path: chat/train.parquet
  - split: validation
    path: chat/validation.parquet
---

# Complexity Atlas Posttrain

English instruction and multi-turn conversation cards derived from the
human-authored OpenAssistant OASST1 conversation trees.

## Configurations

- `instruct`: one-turn user/assistant pairs using the highest-ranked accepted
  response to each root prompt.
- `chat`: one quality-selected multi-turn path per accepted conversation tree.

Rows preserve structured messages, deterministic `User:`/`Assistant:` text,
quality scores, source tree and message IDs, the pinned source revision and
license provenance.

## Filtering

The importer retains reviewed, non-synthetic English messages and applies
quality, helpfulness, task-failure, PII, spam, language-mismatch, content and
toxicity filters. This is a reproducible filtered view, not a claim that every
remaining response is factually correct.

## Intended use

Use these Parquet files for supervised post-training after base pretraining.
An SFT loader should calculate loss on assistant responses while masking user
tokens. The inference prompt format must match the training serialization.

## Source and license

Source: <https://huggingface.co/datasets/OpenAssistant/oasst1>

The derived alignment cards retain the source dataset's Apache-2.0 license.
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

Serialization is `User: <content>\\nAssistant: <content>\\n`. Training and
inference should use the same format.

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

    tokenized_output = temporary / "tokenized" / "o200k"
    _copy_tree(tokenized_root, tokenized_output)
    corpus_manifest, tokenized_manifest = _portable_pretrain_manifests(
        corpus_manifest,
        tokenized_manifest,
        tokenized_output,
    )
    (temporary / "README.md").write_text(
        DATASET_CARD.format(
            alignment_configs=ALIGNMENT_CONFIGS if alignment_root else "",
            alignment_section=ALIGNMENT_SECTION if alignment_root else "",
        )
    )
    (temporary / "LICENSE.md").write_text(PRETRAIN_DATA_LICENSE)

    files = _package_files(temporary)
    package_manifest = {
        "format": "complexity-atlas-pretrain-hf-package-v1",
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


def package_alignment_for_hugging_face(
    alignment_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    alignment_manifest = json.loads((alignment_root / "manifest.json").read_text())
    alignment_table = pq.read_table(alignment_root / "alignment.parquet")

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    for mode in ("instruct", "chat"):
        mode_root = temporary / mode
        mode_root.mkdir()
        mode_table = alignment_table.filter(pc.equal(alignment_table["mode"], mode))
        for split in ("train", "validation"):
            split_table = mode_table.filter(pc.equal(mode_table["split"], split))
            if len(split_table):
                pq.write_table(
                    split_table,
                    mode_root / f"{split}.parquet",
                    compression="zstd",
                    use_dictionary=True,
                    write_statistics=True,
                )

    (temporary / "README.md").write_text(POSTTRAIN_DATASET_CARD)
    files = _package_files(temporary)
    package_manifest = {
        "format": "complexity-atlas-posttrain-hf-package-v1",
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
