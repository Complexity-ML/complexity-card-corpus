from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from .build import DOCUMENT_SCHEMA, file_sha256
from .package import _copy_tree, _package_files, _portable_pretrain_manifests

MOSAIC_FORMAT = "complexity-atlas-mosaic-v1"
ALLOWED_LICENSES = {
    "Apache-2.0",
    "CC BY 4.0",
    "CC BY-NC 4.0",
    "CC BY-SA 3.0",
    "CC0 1.0",
    "ODC-By 1.0",
    "Public Domain",
}
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

SOURCE_SCHEMA = pa.schema(
    [
        ("dataset_id", pa.string()),
        ("kind", pa.string()),
        ("domain", pa.string()),
        ("language", pa.string()),
        ("license", pa.string()),
        ("repo_id", pa.string()),
        ("revision", pa.string()),
        ("config", pa.string()),
        ("files", pa.list_(pa.string())),
        ("source_url", pa.string()),
        ("redistribution", pa.bool_()),
    ]
)

MOSAIC_DATASET_CARD = """---
language:
- en
license: other
pretty_name: Complexity Atlas Mosaic Pretrain
task_categories:
- text-generation
configs:
- config_name: documents
  data_files:
  - split: train
    path: data/train.parquet
  - split: validation
    path: data/validation.parquet
- config_name: sources
  data_files:
  - split: train
    path: catalog/sources.parquet
---

# Complexity Atlas Mosaic Pretrain

A provenance-first, multi-source English pretraining corpus. This repository
is intentionally separate from the original-only `Complexity Atlas Pretrain`.

## License model

This is a collection with mixed licenses. The `license` field on every
document and the `sources` configuration are authoritative. Inclusion in this
collection does not replace an upstream source license.

## Current pilot

The pilot combines original Complexity Atlas documents with a pinned,
filtered sample from Hugging Face Cosmopedia. The build rejects any source
without a supported license, immutable revision and explicit redistribution
flag.

## Processing

- deterministic train/validation assignment;
- exact-content deduplication across sources;
- length and basic email-address filters;
- source URL, revision, row key and license retained per document;
- o200k token shards derived from the canonical Parquet documents.

This filtering reduces risk but does not guarantee factual correctness,
absence of personal data or suitability for every use.
"""


def validate_mosaic_registry(registry: dict[str, Any]) -> list[dict[str, Any]]:
    if registry.get("format") != "complexity-atlas-source-registry-v1":
        raise ValueError("Unsupported Mosaic source registry format")
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Mosaic source registry must contain sources")

    identifiers: set[str] = set()
    for source in sources:
        required = {
            "dataset_id",
            "kind",
            "domain",
            "language",
            "license",
            "repo_id",
            "revision",
            "config",
            "files",
            "text_column",
            "redistribution",
        }
        missing = sorted(required - source.keys())
        if missing:
            raise ValueError(
                f"Source {source.get('dataset_id', '<unknown>')} missing {missing}"
            )
        dataset_id = source["dataset_id"]
        if dataset_id in identifiers:
            raise ValueError(f"Duplicate source dataset_id: {dataset_id}")
        identifiers.add(dataset_id)
        if source["license"] not in ALLOWED_LICENSES:
            raise ValueError(
                f"Source {dataset_id} has unsupported license {source['license']!r}"
            )
        if source["redistribution"] is not True:
            raise ValueError(f"Source {dataset_id} does not allow redistribution")
        if not source["revision"] or source["revision"] in {"main", "latest"}:
            raise ValueError(f"Source {dataset_id} must pin an immutable revision")
        if source["kind"] != "huggingface_parquet":
            raise ValueError(f"Unsupported source kind: {source['kind']}")
    return sources


def _split(document_id: str, validation_per_mille: int) -> str:
    bucket = int(hashlib.sha256(document_id.encode()).hexdigest()[:8], 16) % 1000
    return "validation" if bucket < validation_per_mille else "train"


def _external_document(
    source: dict[str, Any],
    *,
    source_file: str,
    row_number: int,
    text: str,
    validation_per_mille: int,
) -> dict[str, Any]:
    source_key = f"{source_file}:{row_number}"
    digest = hashlib.sha256(
        f"{source['repo_id']}:{source['revision']}:{source_key}".encode()
    ).hexdigest()
    document_id = f"{source['dataset_id']}:{digest[:24]}"
    source_url = (
        f"https://huggingface.co/datasets/{source['repo_id']}/blob/"
        f"{source['revision']}/{source_file}"
    )
    return {
        "document_id": document_id,
        "dataset_id": source["dataset_id"],
        "domain": source["domain"],
        "language": source["language"],
        "split": _split(document_id, validation_per_mille),
        "template": source["config"],
        "source_keys": [source_key],
        "text": text.strip(),
        "source": source["repo_id"],
        "source_urls": [source_url],
        "license": source["license"],
        "version": source["revision"],
    }


def _source_catalog_row(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": source["dataset_id"],
        "kind": source["kind"],
        "domain": source["domain"],
        "language": source["language"],
        "license": source["license"],
        "repo_id": source["repo_id"],
        "revision": source["revision"],
        "config": source["config"],
        "files": source["files"],
        "source_url": f"https://huggingface.co/datasets/{source['repo_id']}",
        "redistribution": source["redistribution"],
    }


def _atlas_source_catalog_rows(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_id in sorted({row["dataset_id"] for row in documents}):
        subset = [row for row in documents if row["dataset_id"] == dataset_id]
        licenses = sorted({row["license"] for row in subset})
        versions = sorted({row["version"] for row in subset})
        rows.append(
            {
                "dataset_id": dataset_id,
                "kind": "atlas_original",
                "domain": subset[0]["domain"],
                "language": subset[0]["language"],
                "license": " OR ".join(licenses),
                "repo_id": "Pacific-i64/complexity-atlas-pretrain",
                "revision": " + ".join(versions),
                "config": "documents",
                "files": ["data/train.parquet", "data/validation.parquet"],
                "source_url": (
                    "https://huggingface.co/datasets/"
                    "Pacific-i64/complexity-atlas-pretrain"
                ),
                "redistribution": True,
            }
        )
    return rows


def _write_table(table: pa.Table, path: Path) -> None:
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


def _read_external_source(
    source: dict[str, Any],
    raw_root: Path,
    *,
    max_rows: int | None,
    validation_per_mille: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    for source_file in source["files"]:
        local_path = Path(
            hf_hub_download(
                repo_id=source["repo_id"],
                filename=source_file,
                repo_type="dataset",
                revision=source["revision"],
                local_dir=raw_root / source["dataset_id"],
            )
        )
        parquet_file = pq.ParquetFile(local_path)
        row_offset = 0
        for batch in parquet_file.iter_batches(
            batch_size=2048,
            columns=[source["text_column"]],
        ):
            for offset, value in enumerate(
                batch.column(0).to_pylist(),
                start=row_offset,
            ):
                if max_rows is not None and len(candidates) >= max_rows:
                    break
                if not isinstance(value, str):
                    rejections["missing_text"] += 1
                    continue
                text = value.strip()
                if len(text) < source.get("minimum_characters", 200):
                    rejections["too_short"] += 1
                    continue
                if len(text) > source.get("maximum_characters", 32_000):
                    rejections["too_long"] += 1
                    continue
                if EMAIL.search(text):
                    rejections["email"] += 1
                    continue
                candidates.append(
                    _external_document(
                        source,
                        source_file=source_file,
                        row_number=offset,
                        text=text,
                        validation_per_mille=validation_per_mille,
                    )
                )
            row_offset += len(batch)
            if max_rows is not None and len(candidates) >= max_rows:
                break
        if max_rows is not None and len(candidates) >= max_rows:
            break
    return candidates, rejections


def build_mosaic(
    registry_path: Path,
    atlas_documents_path: Path,
    raw_root: Path,
    output_root: Path,
    *,
    max_rows_per_source: int | None = None,
    validation_per_mille: int = 5,
    workers: int = 4,
) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text())
    sources = validate_mosaic_registry(registry)
    if not 0 < validation_per_mille < 1000:
        raise ValueError("validation_per_mille must be between 1 and 999")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    atlas_table = pq.read_table(atlas_documents_path, schema=DOCUMENT_SCHEMA)
    documents = atlas_table.to_pylist()
    content_hashes = {
        hashlib.sha256(row["text"].strip().encode()).hexdigest() for row in documents
    }
    rejections: Counter[str] = Counter()
    accepted_by_source: Counter[str] = Counter(
        row["dataset_id"] for row in documents
    )

    worker_count = min(workers, len(sources))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(
            executor.map(
                lambda source: _read_external_source(
                    source,
                    raw_root,
                    max_rows=max_rows_per_source,
                    validation_per_mille=validation_per_mille,
                ),
                sources,
            )
        )

    for source, (candidates, source_rejections) in zip(sources, results, strict=True):
        rejections.update(source_rejections)
        for document in candidates:
            content_hash = hashlib.sha256(document["text"].encode()).hexdigest()
            if content_hash in content_hashes:
                rejections["duplicate"] += 1
                continue
            content_hashes.add(content_hash)
            documents.append(document)
            accepted_by_source[source["dataset_id"]] += 1

    documents.sort(key=lambda row: row["document_id"])
    document_table = pa.Table.from_pylist(documents, schema=DOCUMENT_SCHEMA)
    atlas_catalog = _atlas_source_catalog_rows(atlas_table.to_pylist())
    source_table = pa.Table.from_pylist(
        atlas_catalog + [_source_catalog_row(source) for source in sources],
        schema=SOURCE_SCHEMA,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    documents_path = output_root / "documents.parquet"
    sources_path = output_root / "sources.parquet"
    _write_table(document_table, documents_path)
    _write_table(source_table, sources_path)

    manifest = {
        "format": MOSAIC_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "registry": {
            "path": registry_path.name,
            "sha256": file_sha256(registry_path),
        },
        "atlas_documents": {
            "path": atlas_documents_path.name,
            "sha256": file_sha256(atlas_documents_path),
        },
        "counts": {
            "documents": len(documents),
            "documents_by_source": dict(sorted(accepted_by_source.items())),
            "documents_by_split": dict(
                sorted(Counter(row["split"] for row in documents).items())
            ),
            "rejections": dict(sorted(rejections.items())),
        },
        "build": {
            "workers": worker_count,
            "max_rows_per_source": max_rows_per_source,
            "validation_per_mille": validation_per_mille,
        },
        "sources": sources,
        "files": {
            "documents": {
                "path": documents_path.name,
                "bytes": documents_path.stat().st_size,
                "sha256": file_sha256(documents_path),
            },
            "sources": {
                "path": sources_path.name,
                "bytes": sources_path.stat().st_size,
                "sha256": file_sha256(sources_path),
            },
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def package_mosaic_for_hugging_face(
    mosaic_root: Path,
    tokenized_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    mosaic_manifest = json.loads((mosaic_root / "manifest.json").read_text())
    tokenized_manifest = json.loads((tokenized_root / "manifest.json").read_text())
    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        import shutil

        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    data_root = temporary / "data"
    catalog_root = temporary / "catalog"
    data_root.mkdir()
    catalog_root.mkdir()

    documents = pq.read_table(mosaic_root / "documents.parquet")
    for split, filename in (
        ("train", "train.parquet"),
        ("validation", "validation.parquet"),
    ):
        split_table = documents.filter(pc.equal(documents["split"], split))
        if len(split_table):
            _write_table(split_table, data_root / filename)
    _write_table(
        pq.read_table(mosaic_root / "sources.parquet"),
        catalog_root / "sources.parquet",
    )

    tokenized_output = temporary / "tokenized" / "o200k"
    _copy_tree(tokenized_root, tokenized_output)
    portable_mosaic = json.loads(json.dumps(mosaic_manifest))
    portable_mosaic["registry"]["path"] = "catalog/sources.parquet"
    portable_mosaic["atlas_documents"]["path"] = (
        "Pacific-i64/complexity-atlas-pretrain"
    )
    _, tokenized_manifest = _portable_pretrain_manifests(
        {"source_root": "catalog/"},
        tokenized_manifest,
        tokenized_output,
    )

    (temporary / "README.md").write_text(MOSAIC_DATASET_CARD)
    files = _package_files(temporary)
    package_manifest = {
        "format": "complexity-atlas-mosaic-hf-package-v1",
        "mosaic": portable_mosaic,
        "tokenized": tokenized_manifest,
        "files": files,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(package_manifest, indent=2, sort_keys=True) + "\n"
    )
    if output_root.exists():
        import shutil

        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return package_manifest
