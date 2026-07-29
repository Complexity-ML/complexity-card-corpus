from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from .build import DOCUMENT_SCHEMA, file_sha256
from .mosaic import (
    SOURCE_SCHEMA,
    _atlas_source_catalog_rows,
    _external_document,
    _source_catalog_row,
    validate_mosaic_registry,
)
from .tokenize import DTYPE, directory_sha256, load_encoding

STREAM_FORMAT = "complexity-atlas-mosaic-stream-v1"
STREAM_DATASET_CARD = """---
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
    path: data/train/*.parquet
  - split: validation
    path: data/validation/*.parquet
- config_name: sources
  data_files:
  - split: train
    path: catalog/sources.parquet
---

# Complexity Atlas Mosaic Pretrain

A provenance-first, multi-source English pretraining corpus built toward a
four-billion-token o200k training stream. It remains separate from the
original-only `Complexity Atlas Pretrain`.

## License model

This is a mixed-license collection. Each document retains its upstream
`license`, source dataset, immutable revision, file and row key. The source
catalog under `catalog/sources.parquet` is authoritative. Inclusion does not
replace or weaken an upstream license.

## Reproducible processing

- immutable source revisions and explicit redistribution declarations;
- bounded-memory Parquet processing with resumable file checkpoints;
- exact-content deduplication backed by an on-disk index;
- deterministic train/validation assignment;
- length and basic email-address filters;
- round-robin source-shard tokenization with o200k;
- a 4B-token training target, with the final complete document allowed to
  cross the target by a small amount.

Filtering reduces risk but does not guarantee factual correctness, absence of
personal information or suitability for every use.
"""


def _open_state(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS seen "
        "(content_hash BLOB PRIMARY KEY) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS processed "
        "(source_file TEXT PRIMARY KEY, accepted INTEGER NOT NULL, "
        "rejections_json TEXT NOT NULL, train_path TEXT, validation_path TEXT)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS build_meta "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.commit()
    return connection


def _verify_state_identity(
    connection: sqlite3.Connection,
    identity: dict[str, Any],
) -> None:
    stored = dict(connection.execute("SELECT key, value FROM build_meta"))
    expected = {key: json.dumps(value, sort_keys=True) for key, value in identity.items()}
    if stored and stored != expected:
        raise ValueError(
            "Mosaic resume state belongs to a different registry or build profile"
        )
    if not stored:
        connection.executemany(
            "INSERT INTO build_meta(key, value) VALUES (?, ?)",
            sorted(expected.items()),
        )
        connection.commit()


def _safe_stem(dataset_id: str, source_file: str) -> str:
    digest = hashlib.sha256(source_file.encode()).hexdigest()[:12]
    stem = Path(source_file).stem.replace(".", "-")
    return f"{dataset_id}-{stem}-{digest}"


def _download(
    source: dict[str, Any],
    source_file: str,
    raw_root: Path,
) -> Path:
    return Path(
        hf_hub_download(
            repo_id=source["repo_id"],
            filename=source_file,
            repo_type="dataset",
            revision=source["revision"],
            local_dir=raw_root / source["dataset_id"],
        )
    )


def _new_writer(path: Path) -> pq.ParquetWriter:
    return pq.ParquetWriter(
        path,
        DOCUMENT_SCHEMA,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


def _insert_if_new(connection: sqlite3.Connection, text: str) -> bool:
    digest = hashlib.sha256(text.encode()).digest()
    cursor = connection.execute(
        "INSERT OR IGNORE INTO seen(content_hash) VALUES (?)",
        (digest,),
    )
    return cursor.rowcount == 1


def _write_atlas_once(
    connection: sqlite3.Connection,
    atlas_documents_path: Path,
    output_root: Path,
) -> None:
    key = "__complexity_atlas_original__"
    if connection.execute(
        "SELECT 1 FROM processed WHERE source_file = ?", (key,)
    ).fetchone():
        return
    rows = pq.read_table(atlas_documents_path, schema=DOCUMENT_SCHEMA).to_pylist()
    writers: dict[str, pq.ParquetWriter] = {}
    paths = {
        "train": output_root / "data/train/complexity-atlas-original.parquet",
        "validation": (
            output_root / "data/validation/complexity-atlas-original.parquet"
        ),
    }
    connection.execute("BEGIN")
    try:
        for row in rows:
            _insert_if_new(connection, row["text"].strip())
            split = row["split"]
            if split not in writers:
                paths[split].parent.mkdir(parents=True, exist_ok=True)
                partial = paths[split].with_suffix(".partial")
                partial.unlink(missing_ok=True)
                writers[split] = _new_writer(partial)
            writers[split].write_table(
                pa.Table.from_pylist([row], schema=DOCUMENT_SCHEMA)
            )
        for writer in writers.values():
            writer.close()
        for split in writers:
            paths[split].with_suffix(".partial").replace(paths[split])
        connection.execute(
            "INSERT INTO processed VALUES (?, ?, ?, ?, ?)",
            (
                key,
                len(rows),
                "{}",
                str(paths["train"].relative_to(output_root)),
                str(paths["validation"].relative_to(output_root)),
            ),
        )
        connection.commit()
    except BaseException:
        for writer in writers.values():
            writer.close()
        connection.rollback()
        for path in paths.values():
            path.with_suffix(".partial").unlink(missing_ok=True)
        raise


def _process_source_file(
    connection: sqlite3.Connection,
    source: dict[str, Any],
    source_file: str,
    local_path: Path,
    output_root: Path,
    *,
    validation_per_mille: int,
    batch_size: int,
) -> None:
    key = f"{source['dataset_id']}::{source_file}"
    if connection.execute(
        "SELECT 1 FROM processed WHERE source_file = ?", (key,)
    ).fetchone():
        return

    stem = _safe_stem(source["dataset_id"], source_file)
    paths = {
        "train": output_root / f"data/train/{stem}.parquet",
        "validation": output_root / f"data/validation/{stem}.parquet",
    }
    for path in paths.values():
        path.unlink(missing_ok=True)
        path.with_suffix(".partial").unlink(missing_ok=True)

    writers: dict[str, pq.ParquetWriter] = {}
    accepted = 0
    rejections: Counter[str] = Counter()
    row_offset = 0
    connection.execute("BEGIN")
    try:
        parquet_file = pq.ParquetFile(local_path)
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=[source["text_column"]],
        ):
            output_rows: dict[str, list[dict[str, Any]]] = {
                "train": [],
                "validation": [],
            }
            for offset, value in enumerate(
                batch.column(0).to_pylist(),
                start=row_offset,
            ):
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
                if "@" in text:
                    from .mosaic import EMAIL

                    if EMAIL.search(text):
                        rejections["email"] += 1
                        continue
                if not _insert_if_new(connection, text):
                    rejections["duplicate"] += 1
                    continue
                document = _external_document(
                    source,
                    source_file=source_file,
                    row_number=offset,
                    text=text,
                    validation_per_mille=validation_per_mille,
                )
                output_rows[document["split"]].append(document)
                accepted += 1
            row_offset += len(batch)
            for split, rows in output_rows.items():
                if not rows:
                    continue
                if split not in writers:
                    paths[split].parent.mkdir(parents=True, exist_ok=True)
                    writers[split] = _new_writer(paths[split].with_suffix(".partial"))
                writers[split].write_table(
                    pa.Table.from_pylist(rows, schema=DOCUMENT_SCHEMA)
                )

        for writer in writers.values():
            writer.close()
        for split in writers:
            paths[split].with_suffix(".partial").replace(paths[split])
        connection.execute(
            "INSERT INTO processed VALUES (?, ?, ?, ?, ?)",
            (
                key,
                accepted,
                json.dumps(dict(rejections), sort_keys=True),
                (
                    str(paths["train"].relative_to(output_root))
                    if "train" in writers
                    else None
                ),
                (
                    str(paths["validation"].relative_to(output_root))
                    if "validation" in writers
                    else None
                ),
            ),
        )
        connection.commit()
    except BaseException:
        for writer in writers.values():
            writer.close()
        connection.rollback()
        for path in paths.values():
            path.with_suffix(".partial").unlink(missing_ok=True)
        raise


def build_mosaic_shards(
    registry_path: Path,
    atlas_documents_path: Path,
    raw_root: Path,
    output_root: Path,
    *,
    validation_per_mille: int = 5,
    workers: int = 4,
    batch_size: int = 8192,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if not 0 < validation_per_mille < 1000:
        raise ValueError("validation_per_mille must be between 1 and 999")
    registry = json.loads(registry_path.read_text())
    sources = validate_mosaic_registry(registry)
    output_root.mkdir(parents=True, exist_ok=True)
    registry_sha256 = file_sha256(registry_path)
    state_root = raw_root / ".state" / registry_sha256[:16]
    state_root.mkdir(parents=True, exist_ok=True)
    connection = _open_state(state_root / "mosaic.sqlite")
    _verify_state_identity(
        connection,
        {
            "registry_sha256": registry_sha256,
            "atlas_documents_sha256": file_sha256(atlas_documents_path),
            "validation_per_mille": validation_per_mille,
            "format": STREAM_FORMAT,
        },
    )
    _write_atlas_once(connection, atlas_documents_path, output_root)

    tasks = [
        (source, source_file)
        for source in sources
        for source_file in source["files"]
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        downloads = executor.map(
            lambda task: _download(task[0], task[1], raw_root),
            tasks,
        )
        for (source, source_file), local_path in zip(
            tasks, downloads, strict=True
        ):
            _process_source_file(
                connection,
                source,
                source_file,
                local_path,
                output_root,
                validation_per_mille=validation_per_mille,
                batch_size=batch_size,
            )

    atlas_rows = pq.read_table(atlas_documents_path, schema=DOCUMENT_SCHEMA).to_pylist()
    catalog = _atlas_source_catalog_rows(atlas_rows) + [
        _source_catalog_row(source) for source in sources
    ]
    catalog_root = output_root / "catalog"
    catalog_root.mkdir(exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(catalog, schema=SOURCE_SCHEMA),
        catalog_root / "sources.parquet",
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    processed_rows = connection.execute(
        "SELECT source_file, accepted, rejections_json, train_path, "
        "validation_path FROM processed ORDER BY source_file"
    ).fetchall()
    connection.close()
    rejections: Counter[str] = Counter()
    for row in processed_rows:
        rejections.update(json.loads(row[2]))
    manifest = {
        "format": STREAM_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "registry_sha256": registry_sha256,
        "atlas_documents_sha256": file_sha256(atlas_documents_path),
        "counts": {
            "documents": sum(row[1] for row in processed_rows),
            "source_files": len(processed_rows) - 1,
            "rejections": dict(sorted(rejections.items())),
        },
        "build": {
            "workers": workers,
            "batch_size": batch_size,
            "validation_per_mille": validation_per_mille,
            "resumable": True,
        },
        "sources": sources,
    }
    (output_root / "corpus-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output_root / "README.md").write_text(STREAM_DATASET_CARD)
    return manifest


def _round_robin_batches(
    paths: list[Path],
    *,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    queue = deque(
        pq.ParquetFile(path).iter_batches(
            batch_size=batch_size,
            columns=["document_id", "dataset_id", "text"],
        )
        for path in paths
    )
    while queue:
        iterator = queue.popleft()
        try:
            batch = next(iterator)
        except StopIteration:
            continue
        yield batch
        queue.append(iterator)


def _tokenize_partition(
    paths: list[Path],
    output_root: Path,
    encoding,
    eos_id: int,
    *,
    target_tokens: int | None,
    workers: int,
    batch_size: int,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    bin_path = output_root / "tokens.bin"
    digest = hashlib.sha256()
    token_count = 0
    document_count = 0
    max_token_id = -1
    with bin_path.open("wb") as handle:
        for batch in _round_robin_batches(paths, batch_size=batch_size):
            rows = batch.to_pylist()
            encoded = encoding.encode_ordinary_batch(
                [row["text"] for row in rows],
                num_threads=workers,
            )
            for token_ids in encoded:
                token_ids.append(eos_id)
                payload = np.asarray(token_ids, dtype=DTYPE).tobytes()
                handle.write(payload)
                digest.update(payload)
                token_count += len(token_ids)
                document_count += 1
                max_token_id = max(max_token_id, max(token_ids))
                if target_tokens is not None and token_count >= target_tokens:
                    break
            if target_tokens is not None and token_count >= target_tokens:
                break
    return {
        "bin": bin_path.name,
        "dtype": DTYPE.str,
        "documents": document_count,
        "num_tokens": token_count,
        "target_tokens": target_tokens,
        "max_token_id": max_token_id,
        "sha256": digest.hexdigest(),
        "source_files": len(paths),
    }


def tokenize_mosaic_shards(
    corpus_root: Path,
    tokenizer_root: Path,
    *,
    target_train_tokens: int = 4_000_000_000,
    target_eval_tokens: int = 20_000_000,
    workers: int = 8,
    batch_size: int = 256,
) -> dict[str, Any]:
    encoding, tokenizer_config = load_encoding(tokenizer_root)
    eos_token = tokenizer_config.get("eos_token", "<|endoftext|>")
    try:
        eos_id = encoding.encode_single_token(eos_token)
    except KeyError:
        eos_id = encoding.eot_token
    tokenized_root = corpus_root / "tokenized/o200k"
    partitions = {}
    for split, partition, target in (
        ("train", "train", target_train_tokens),
        ("validation", "eval", target_eval_tokens),
    ):
        paths = sorted((corpus_root / f"data/{split}").glob("*.parquet"))
        if not paths:
            continue
        metadata = _tokenize_partition(
            paths,
            tokenized_root / partition,
            encoding,
            eos_id,
            target_tokens=target,
            workers=workers,
            batch_size=batch_size,
        )
        metadata.update(
            {
                "format": "complexity-token-shard-v1",
                "partition": partition,
                "vocab_size": encoding.n_vocab,
                "eos_token_id": eos_id,
                "tokenizer": "tiktoken:o200k_base",
                "tokenizer_name": tokenizer_config["encoding_name"],
                "tokenizer_sha256": directory_sha256(tokenizer_root),
                "source_documents": f"data/{split}/*.parquet",
            }
        )
        (tokenized_root / partition / "tokens.idx.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        partitions[partition] = metadata
    manifest = {
        "format": "complexity-tokenized-mosaic-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": tokenizer_config["encoding_name"],
        "partitions": partitions,
        "total_documents": sum(row["documents"] for row in partitions.values()),
        "total_tokens": sum(row["num_tokens"] for row in partitions.values()),
    }
    (tokenized_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
