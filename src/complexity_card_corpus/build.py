from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .models import CardDataset
from .render import render_documents
from .source import discover_datasets

CARD_SCHEMA = pa.schema(
    [
        ("dataset_id", pa.string()),
        ("domain", pa.string()),
        ("themes", pa.list_(pa.string())),
        ("language", pa.string()),
        ("split", pa.string()),
        ("key", pa.string()),
        ("kind", pa.string()),
        ("name", pa.string()),
        ("aliases", pa.list_(pa.string())),
        ("summary", pa.string()),
        ("description", pa.string()),
        ("facts", pa.list_(pa.string())),
        ("tags", pa.list_(pa.string())),
        ("attributes_json", pa.string()),
        ("source", pa.string()),
        ("source_urls", pa.list_(pa.string())),
        ("license", pa.string()),
        ("version", pa.string()),
    ]
)

RELATION_SCHEMA = pa.schema(
    [
        ("dataset_id", pa.string()),
        ("split", pa.string()),
        ("from_key", pa.string()),
        ("relation", pa.string()),
        ("to_dataset_id", pa.string()),
        ("to_key", pa.string()),
        ("detail", pa.string()),
    ]
)

DOCUMENT_SCHEMA = pa.schema(
    [
        ("document_id", pa.string()),
        ("dataset_id", pa.string()),
        ("domain", pa.string()),
        ("themes", pa.list_(pa.string())),
        ("language", pa.string()),
        ("split", pa.string()),
        ("template", pa.string()),
        ("source_keys", pa.list_(pa.string())),
        ("text", pa.string()),
        ("source", pa.string()),
        ("source_urls", pa.list_(pa.string())),
        ("license", pa.string()),
        ("version", pa.string()),
    ]
)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _card_rows(datasets: list[CardDataset]) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        metadata = dataset.metadata
        for card in dataset.cards:
            rows.append(
                {
                    "dataset_id": metadata.dataset_id,
                    "domain": metadata.domain,
                    "themes": metadata.themes,
                    "language": metadata.language,
                    "split": metadata.split,
                    "key": card.key,
                    "kind": card.kind,
                    "name": card.name,
                    "aliases": card.aliases,
                    "summary": card.summary,
                    "description": card.description or card.summary,
                    "facts": card.facts,
                    "tags": card.tags,
                    "attributes_json": json.dumps(
                        card.attributes, sort_keys=True, ensure_ascii=False
                    ),
                    "source": metadata.source,
                    "source_urls": metadata.source_urls,
                    "license": metadata.license,
                    "version": metadata.version,
                }
            )
    return sorted(rows, key=lambda row: (row["dataset_id"], row["key"]))


def _relation_rows(datasets: list[CardDataset]) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        metadata = dataset.metadata
        for card in dataset.cards:
            for relation in card.relations:
                rows.append(
                    {
                        "dataset_id": metadata.dataset_id,
                        "split": metadata.split,
                        "from_key": card.key,
                        "relation": relation.type,
                        "to_dataset_id": relation.target_dataset_id
                        or metadata.dataset_id,
                        "to_key": relation.target_key,
                        "detail": relation.detail,
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            row["dataset_id"],
            row["from_key"],
            row["relation"],
            row["to_dataset_id"],
            row["to_key"],
        ),
    )


def _write_table(rows: list[dict[str, Any]], schema: pa.Schema, path: Path) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


def build_corpus(
    source_root: Path,
    output_root: Path,
    *,
    max_path_depth: int = 3,
    max_paths_per_card: int = 4,
) -> dict[str, Any]:
    datasets = discover_datasets(source_root)
    cards = _card_rows(datasets)
    relations = _relation_rows(datasets)
    documents = [
        {
            "document_id": document.document_id,
            "dataset_id": document.dataset_id,
            "domain": document.domain,
            "themes": document.themes,
            "language": document.language,
            "split": document.split,
            "template": document.template,
            "source_keys": document.source_keys,
            "text": document.text,
            "source": document.source,
            "source_urls": document.source_urls,
            "license": document.license,
            "version": document.version,
        }
        for document in render_documents(
            datasets,
            max_path_depth=max_path_depth,
            max_paths_per_card=max_paths_per_card,
        )
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "cards": output_root / "cards.parquet",
        "relations": output_root / "relations.parquet",
        "documents": output_root / "documents.parquet",
    }
    _write_table(cards, CARD_SCHEMA, paths["cards"])
    _write_table(relations, RELATION_SCHEMA, paths["relations"])
    _write_table(documents, DOCUMENT_SCHEMA, paths["documents"])

    manifest = {
        "format": "complexity-card-corpus-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root.resolve()),
        "datasets": [
            dataset.metadata.model_dump(mode="json", by_alias=True)
            for dataset in datasets
        ],
        "counts": {
            "datasets": len(datasets),
            "cards": len(cards),
            "relations": len(relations),
            "documents": len(documents),
            "cards_by_domain": dict(sorted(Counter(row["domain"] for row in cards).items())),
            "documents_by_split": dict(
                sorted(Counter(row["split"] for row in documents).items())
            ),
            "documents_by_template": dict(
                sorted(Counter(row["template"] for row in documents).items())
            ),
        },
        "files": {
            name: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in paths.items()
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
