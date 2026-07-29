from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import tiktoken

DTYPE = np.dtype("<u4")
FORMAT = "complexity-token-shard-v1"


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(child for child in path.rglob("*") if child.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def load_encoding(tokenizer_root: Path):
    config_path = tokenizer_root / "tiktoken_config.json"
    config = json.loads(config_path.read_text())
    encoding_name = config["encoding_name"]
    cache_dir = tokenizer_root / config.get("cache_dir", ".")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir.resolve())
    return tiktoken.get_encoding(encoding_name), config


def _target_partition(split: str) -> str:
    return {"train": "train", "validation": "eval", "test": "test"}[split]


def tokenize_documents(
    documents_path: Path,
    tokenizer_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    encoding, tokenizer_config = load_encoding(tokenizer_root)
    eos_token = tokenizer_config.get("eos_token", "<|endoftext|>")
    try:
        eos_id = encoding.encode_single_token(eos_token)
    except KeyError:
        eos_id = encoding.eot_token

    table = pq.read_table(
        documents_path,
        columns=["document_id", "dataset_id", "split", "text"],
    )
    rows = sorted(table.to_pylist(), key=lambda row: row["document_id"])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        partition = _target_partition(row["split"])
        grouped.setdefault(partition, []).append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    partition_manifests = {}
    for partition, partition_rows in sorted(grouped.items()):
        partition_root = output_root / partition
        partition_root.mkdir(parents=True, exist_ok=True)
        bin_path = partition_root / "tokens.bin"
        provenance_path = partition_root / "documents.jsonl"
        digest = hashlib.sha256()
        token_count = 0
        max_token_id = -1
        with bin_path.open("wb") as token_handle, provenance_path.open(
            "w", encoding="utf-8"
        ) as provenance_handle:
            for row in partition_rows:
                token_ids = encoding.encode(row["text"], disallowed_special=())
                token_ids.append(eos_id)
                array = np.asarray(token_ids, dtype=DTYPE)
                payload = array.tobytes()
                token_handle.write(payload)
                digest.update(payload)
                start = token_count
                token_count += len(token_ids)
                max_token_id = max(max_token_id, max(token_ids))
                provenance_handle.write(
                    json.dumps(
                        {
                            "document_id": row["document_id"],
                            "dataset_id": row["dataset_id"],
                            "start_token": start,
                            "num_tokens": len(token_ids),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

        metadata = {
            "format": FORMAT,
            "bin": bin_path.name,
            "dtype": DTYPE.str,
            "partition": partition,
            "documents": len(partition_rows),
            "num_tokens": token_count,
            "max_token_id": max_token_id,
            "vocab_size": encoding.n_vocab,
            "eos_token_id": eos_id,
            "tokenizer": str(tokenizer_root.resolve()),
            "tokenizer_name": tokenizer_config["encoding_name"],
            "tokenizer_sha256": directory_sha256(tokenizer_root),
            "source_documents": str(documents_path.resolve()),
            "source_documents_sha256": file_sha256(documents_path),
            "sha256": digest.hexdigest(),
        }
        index_path = partition_root / "tokens.idx.json"
        index_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        partition_manifests[partition] = metadata

    manifest = {
        "format": "complexity-tokenized-card-corpus-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": tokenizer_config["encoding_name"],
        "partitions": partition_manifests,
        "total_documents": sum(item["documents"] for item in partition_manifests.values()),
        "total_tokens": sum(item["num_tokens"] for item in partition_manifests.values()),
        "documents_by_partition": dict(
            Counter(
                partition
                for partition, items in grouped.items()
                for _ in items
            )
        ),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest

