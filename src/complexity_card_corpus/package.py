from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow as pa
import pyarrow.parquet as pq

from .build import file_sha256
from .domain_taxonomy import domain_group

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
(`complexity-chat-v2`). It renders `User:\n<content>\n\nAssistant:\n` without
injecting a default system instruction; only assistant content and EOS are
supervised. Reasoning-task targets retain the audited
`<think>...</think><final>...</final>` protocol. Training and inference must
use this exact same contract.

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


def _sft_view_configs(release_slug: str) -> str:
    return f"""configs:
- config_name: chat
  data_files:
  - split: train
    path: data/{release_slug}/chat/train-*.parquet
- config_name: instruct
  data_files:
  - split: train
    path: data/{release_slug}/instruct/train-*.parquet
  - split: validation
    path: data/{release_slug}/instruct/validation.parquet
  - split: diagnostic
    path: data/{release_slug}/instruct/diagnostic.parquet
"""


_HF_VIEW_COLUMNS = {
    "chat": (
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
    ),
    "instruct": (
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
    ),
}

_HF_REASONING_COLUMNS = (
    "reasoning_envelope",
    "reasoning_trace",
    "final_response",
    "reasoning_card_hand",
)


def _project_hf_view(table, mode: str):
    """Return the explicit, stable schema exposed by one Hub subset.

    ``mode`` and ``split`` are routing metadata. Hugging Face already exposes
    them as the subset and split selectors, so repeating them inside every row
    makes the Viewer harder to read. ``example_id`` remains in the canonical
    projection and token indexes but is omitted here: the Hub statistics
    service currently fails its string-length histogram when hundreds of
    thousands of unique identifiers have only two adjacent lengths. Chat rows
    expose ``messages``; instruct rows expose the direct
    ``prompt``/``response`` pair.
    """

    if "domain_group" not in table.column_names:
        groups = pa.array(
            [domain_group(value) for value in table["domain"].to_pylist()],
            type=pa.string(),
        )
        table = table.append_column("domain_group", groups)
    columns = _HF_VIEW_COLUMNS[mode]
    if all(name in table.column_names for name in _HF_REASONING_COLUMNS):
        insertion = columns.index("structure_signature")
        columns = (
            *columns[:insertion],
            *_HF_REASONING_COLUMNS,
            *columns[insertion:],
        )
    missing = set(columns).difference(table.column_names)
    if missing:
        raise ValueError(
            f"Projected SFT {mode} view is missing columns: {sorted(missing)}"
        )
    return table.select(columns)


def _replace_dataset_card_configs(dataset_card: Path, release_slug: str) -> None:
    text = dataset_card.read_text()
    if not text.startswith("---\n"):
        raise ValueError(f"Dataset card has no YAML front matter: {dataset_card}")
    closing = text.find("\n---\n", 4)
    if closing < 0:
        raise ValueError(f"Dataset card has unterminated YAML front matter: {dataset_card}")

    front_matter = text[4:closing]
    lines = front_matter.splitlines()
    config_start = next(
        (index for index, line in enumerate(lines) if line == "configs:"),
        None,
    )
    if config_start is None:
        lines.append(_sft_view_configs(release_slug).rstrip())
    else:
        # Dataset cards produced by this project keep configs as the final
        # top-level YAML field. Refuse an ambiguous rewrite instead of silently
        # deleting metadata that follows it.
        later_top_level = [
            line
            for line in lines[config_start + 1 :]
            if line and not line.startswith((" ", "-"))
        ]
        if later_top_level:
            raise ValueError(
                "Dataset card configs must be the final YAML field before rewriting"
            )
        lines = lines[:config_start]
        lines.append(_sft_view_configs(release_slug).rstrip())

    updated_front_matter = "\n".join(lines).rstrip() + "\n"
    body = text[closing + len("\n---\n") :]
    dataset_card.write_text(
        f"---\n{updated_front_matter}---\n{body}"
    )


def package_sft_views_for_hugging_face(
    projected_parquet: Path,
    output_root: Path,
    *,
    release_slug: str,
    max_rows_per_shard: int = 50_000,
    row_group_size: int = 5_000,
) -> dict[str, Any]:
    """Publish one projection as separate chat and instruct Hub subsets.

    Rows are stored exactly once. Hugging Face exposes ``chat`` and
    ``instruct`` as subsets, with train/validation/diagnostic beneath them as
    splits. Each subset has a small, explicit column schema instead of a union
    of every internal projection field.
    """

    if max_rows_per_shard <= 0:
        raise ValueError("max_rows_per_shard must be positive")
    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")
    dataset_card = output_root / "README.md"
    if not dataset_card.exists():
        raise FileNotFoundError(dataset_card)

    table = pq.read_table(projected_parquet)
    required_columns = {"mode", "split"}
    missing_columns = required_columns.difference(table.column_names)
    if missing_columns:
        raise ValueError(
            f"Projected SFT table is missing columns: {sorted(missing_columns)}"
        )

    modes = set(pc.unique(table["mode"]).to_pylist())
    unsupported_modes = modes.difference({"chat", "instruct"})
    if unsupported_modes:
        raise ValueError(f"Unsupported SFT modes: {sorted(unsupported_modes)}")

    temporary = output_root / f".{release_slug}.partial"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    parquet_shards: list[dict[str, Any]] = []
    view_counts: dict[str, dict[str, int]] = {
        "chat": {},
        "instruct": {},
    }
    for mode in ("chat", "instruct"):
        mode_root = temporary / mode
        mode_root.mkdir()
        for split in ("train", "validation", "diagnostic"):
            split_table = table.filter(
                pc.and_(
                    pc.equal(table["mode"], mode),
                    pc.equal(table["split"], split),
                )
            )
            row_count = len(split_table)
            if row_count == 0:
                continue
            view_counts[mode][split] = row_count
            split_table = _project_hf_view(split_table, mode)

            shard_count = math.ceil(row_count / max_rows_per_shard)
            for shard_index in range(shard_count):
                shard = split_table.slice(
                    shard_index * max_rows_per_shard,
                    max_rows_per_shard,
                )
                if shard_count == 1:
                    filename = f"{split}.parquet"
                else:
                    filename = (
                        f"{split}-{shard_index:05d}-of-{shard_count:05d}.parquet"
                    )
                destination = mode_root / filename
                pq.write_table(
                    shard,
                    destination,
                    compression="zstd",
                    use_dictionary=True,
                    write_statistics=True,
                    row_group_size=row_group_size,
                    write_page_index=True,
                )
                parquet_shards.append(
                    {
                        "path": str(
                            Path("data") / release_slug / mode / filename
                        ),
                        "mode": mode,
                        "split": split,
                        "rows": len(shard),
                        "bytes": destination.stat().st_size,
                        "sha256": file_sha256(destination),
                    }
                )

    destination_root = output_root / "data" / release_slug
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    if destination_root.exists():
        shutil.rmtree(destination_root)
    temporary.replace(destination_root)
    _replace_dataset_card_configs(dataset_card, release_slug)

    release_path = output_root / "release.json"
    release = json.loads(release_path.read_text()) if release_path.exists() else {}
    release.update(
        {
            "examples": len(table),
            "source_projected_sha256": file_sha256(projected_parquet),
            "parquet_shards": parquet_shards,
            "views": view_counts,
        }
    )
    release_path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    return release


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
