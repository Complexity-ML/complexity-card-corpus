from __future__ import annotations

import csv
import io
import json
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from ..build import file_sha256
from ..conversation_mine import load_conversation_registry
from .lexical_schema import (
    CONSTRAINT_CUES,
    INTENT_CUES,
    OUTCOME_CUES,
    STATE_CUES,
    TRANSITIONS,
)


def _taskmaster_documents(path: Path) -> Iterator[str]:
    payload = json.loads(path.read_text())
    conversations = payload if isinstance(payload, list) else [payload]
    for conversation in conversations:
        for utterance in conversation.get("utterances", []):
            text = str(utterance.get("text", "")).strip()
            if text:
                yield text


def _empathetic_documents(path: Path) -> Iterator[str]:
    with tarfile.open(path, "r:gz") as archive:
        for split in ("train", "valid", "test"):
            member = archive.extractfile(f"empatheticdialogues/{split}.csv")
            if member is None:
                raise FileNotFoundError(f"empatheticdialogues/{split}.csv")
            reader = csv.DictReader(io.TextIOWrapper(member, encoding="utf-8"))
            for record in reader:
                text = str(record.get("utterance", "")).strip()
                if text:
                    yield text


def _matches_filters(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    for field, expected in filters.items():
        actual = row.get(field)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _row_role_documents(
    row: dict[str, Any], source: dict[str, Any]
) -> Iterator[tuple[str, str]]:
    for field in source.get("text_fields", []):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            yield str(source.get("text_field_role", "unspecified")), value.strip()
    messages_field = source.get("messages_field")
    if messages_field:
        for message in row.get(messages_field) or []:
            if not isinstance(message, dict):
                continue
            role = str(message.get(source.get("role_field", "role"), ""))
            allowed_roles = source.get("allowed_roles")
            if allowed_roles and role not in allowed_roles:
                continue
            content = message.get(source.get("content_field", "content"))
            if isinstance(content, str) and content.strip():
                yield role or "unspecified", content.strip()


def _row_documents(row: dict[str, Any], source: dict[str, Any]) -> Iterator[str]:
    for _, text in _row_role_documents(row, source):
        yield text


def _parquet_documents(path: Path, source: dict[str, Any]) -> Iterator[str]:
    filters = source.get("filters", {})
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2_048):
        for row in batch.to_pylist():
            if _matches_filters(row, filters):
                yield from _row_documents(row, source)


def _parquet_role_documents(
    path: Path, source: dict[str, Any]
) -> Iterator[tuple[str, str]]:
    filters = source.get("filters", {})
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2_048):
        for row in batch.to_pylist():
            if _matches_filters(row, filters):
                yield from _row_role_documents(row, source)


def _jsonl_documents(path: Path, source: dict[str, Any]) -> Iterator[str]:
    opener: Any
    if path.suffix == ".gz":
        import gzip

        opener = gzip.open
    else:
        opener = Path.open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if _matches_filters(row, source.get("filters", {})):
                yield from _row_documents(row, source)


def _jsonl_role_documents(
    path: Path, source: dict[str, Any]
) -> Iterator[tuple[str, str]]:
    opener: Any
    if path.suffix == ".gz":
        import gzip

        opener = gzip.open
    else:
        opener = Path.open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if _matches_filters(row, source.get("filters", {})):
                yield from _row_role_documents(row, source)


def _source_documents(path: Path, source: dict[str, Any]) -> Iterator[str]:
    kind = source["kind"]
    if kind == "taskmaster_json":
        yield from _taskmaster_documents(path)
    elif kind == "empathetic_tar":
        yield from _empathetic_documents(path)
    elif kind == "parquet_fields":
        yield from _parquet_documents(path, source)
    elif kind == "jsonl_fields":
        yield from _jsonl_documents(path, source)
    else:
        raise ValueError(f"unsupported lexical source kind: {kind}")


def _source_role_documents(
    path: Path, source: dict[str, Any]
) -> Iterator[tuple[str, str]]:
    kind = source["kind"]
    if kind in {"taskmaster_json", "empathetic_tar"}:
        for text in _source_documents(path, source):
            yield "unspecified", text
    elif kind == "parquet_fields":
        yield from _parquet_role_documents(path, source)
    elif kind == "jsonl_fields":
        yield from _jsonl_role_documents(path, source)
    else:
        raise ValueError(f"unsupported lexical source kind: {kind}")


def load_lexical_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("version") == 1:
        legacy = load_conversation_registry(path)
        return {
            "version": 2,
            "sources": [
                {
                    **source,
                    "legacy_layout": True,
                    "artifacts": [
                        {
                            "filename": source["filename"],
                            "download_url": source["download_url"],
                            "sha256": source["sha256"],
                        }
                    ],
                }
                for source in legacy["sources"]
            ],
        }
    if payload.get("version") != 2 or not payload.get("sources"):
        raise ValueError("lexical source registry must be version 1 or 2 and non-empty")
    required = {"dataset_id", "kind", "source_url", "revision", "license", "artifacts"}
    supported = {
        "taskmaster_json",
        "empathetic_tar",
        "parquet_fields",
        "jsonl_fields",
    }
    for source in payload["sources"]:
        missing = sorted(required - source.keys())
        if missing:
            raise ValueError(
                f"source {source.get('dataset_id', '?')} is missing {missing}"
            )
        if source["kind"] not in supported:
            raise ValueError(f"unsupported lexical source kind: {source['kind']}")
        if not re.fullmatch(
            r"[0-9a-f]{40}|artifact-sha256:[0-9a-f]{64}", source["revision"]
        ):
            raise ValueError(f"source {source['dataset_id']} is not pinned")
        if not source["artifacts"]:
            raise ValueError(f"source {source['dataset_id']} has no artifacts")
        for artifact in source["artifacts"]:
            if {"filename", "download_url", "sha256"} - set(artifact):
                raise ValueError(
                    f"source {source['dataset_id']} has an incomplete artifact"
                )
            if not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]):
                raise ValueError(f"source {source['dataset_id']} has an invalid sha256")
    return payload


def _artifact_path(
    raw_root: Path, source: dict[str, Any], artifact: dict[str, Any]
) -> Path:
    if source.get("legacy_layout"):
        return raw_root / artifact["filename"]
    dataset_dir = re.sub(r"[^a-zA-Z0-9_.-]+", "__", source["dataset_id"])
    return raw_root / dataset_dir / Path(artifact["filename"]).name


def fetch_lexical_sources(registry_path: Path, raw_root: Path) -> dict[str, Any]:
    registry = load_lexical_registry(registry_path)
    files: dict[str, Any] = {}
    for source in registry["sources"]:
        for artifact in source["artifacts"]:
            destination = _artifact_path(raw_root, source, artifact)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (
                not destination.exists()
                or file_sha256(destination) != artifact["sha256"]
            ):
                partial = destination.with_suffix(f"{destination.suffix}.partial")
                partial.unlink(missing_ok=True)
                request = urllib.request.Request(
                    artifact["download_url"],
                    headers={"User-Agent": "complexity-card-corpus/0.1"},
                )
                with (
                    urllib.request.urlopen(request) as response,
                    partial.open("wb") as output,
                ):
                    shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
                actual = file_sha256(partial)
                if actual != artifact["sha256"]:
                    partial.unlink(missing_ok=True)
                    raise ValueError(
                        f"sha256 mismatch for {source['dataset_id']}/{artifact['filename']}: {actual}"
                    )
                partial.replace(destination)
            key = f"{source['dataset_id']}:{artifact['filename']}"
            files[key] = {
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": file_sha256(destination),
            }
    return {"files": files}


def _roles(tokens: list[str], index: int) -> set[str]:
    token = tokens[index]
    roles = {"vocabulary"}
    previous = tokens[index - 1] if index else ""
    if token in TRANSITIONS:
        roles.add("transition")
    if previous in INTENT_CUES:
        roles.add("intent_term")
    if previous in STATE_CUES:
        roles.add("state_term")
    if previous in CONSTRAINT_CUES:
        roles.add("constraint_term")
    if previous in OUTCOME_CUES:
        roles.add("outcome_term")
    return roles
