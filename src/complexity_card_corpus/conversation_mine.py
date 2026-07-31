from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import tarfile
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from .build import file_sha256


EXTRACTION_VERSION = "conversation-structure-v1"

CONVERSATION_MINE_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("source_dataset", pa.string()),
        ("source_record_id", pa.string()),
        ("source_record_sha256", pa.string()),
        ("source_revision", pa.string()),
        ("source_url", pa.string()),
        ("source_license", pa.string()),
        ("source_file_sha256", pa.string()),
        ("source_split", pa.string()),
        ("corpus_kind", pa.string()),
        ("domain", pa.string()),
        ("emotion", pa.string()),
        ("turn_count", pa.int32()),
        ("speaker_pattern", pa.list_(pa.string())),
        ("turn_signal_sequence", pa.list_(pa.string())),
        ("slot_types", pa.list_(pa.string())),
        ("question_pattern", pa.list_(pa.bool_())),
        ("utterance_length_buckets", pa.list_(pa.string())),
        ("source_text_retained", pa.bool_()),
        ("extraction_version", pa.string()),
    ]
)

_PROSE_FIELD_NAMES = {"text", "utterance", "prompt", "response", "messages"}
_SAFE_LABEL = re.compile(r"[^a-z0-9_.:-]+")


def _json_canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            "__extra__" if key is None else str(key): _json_canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [_json_canonical(item) for item in value]
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _stable_order(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _normalize_label(value: str) -> str:
    return _SAFE_LABEL.sub("_", value.strip().lower()).strip("_.:-")


def _length_bucket(text: str) -> str:
    words = len(text.split())
    if words <= 4:
        return "very_short"
    if words <= 12:
        return "short"
    if words <= 30:
        return "medium"
    return "long"


def _is_question(text: str) -> bool:
    return text.rstrip().endswith("?")


def _taskmaster_annotation_names(utterance: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    for segment in utterance.get("segments", []):
        for annotation in segment.get("annotations", []):
            name = _normalize_label(str(annotation.get("name", "")))
            if name:
                names.add(name)
    return sorted(names)


def _taskmaster_turn_signal(names: Iterable[str]) -> str:
    names = tuple(names)
    if not names:
        return "none"
    states = []
    if any(name.endswith(".accept") for name in names):
        states.append("accept")
    if any(name.endswith(".reject") for name in names):
        states.append("reject")
    if any(not name.endswith((".accept", ".reject")) for name in names):
        states.append("slot")
    return "+".join(states) or "slot"


def _taskmaster_slot(name: str) -> str:
    parts = name.split(".")
    if parts and parts[-1] in {"accept", "reject"}:
        parts.pop()
    if parts:
        parts.pop(0)
    return ".".join(parts)


def _taskmaster_domain(
    instruction_id: str,
    annotation_names: Iterable[str],
) -> str:
    domains = sorted({name.split(".", 1)[0] for name in annotation_names if "." in name})
    if domains:
        return {"uber_lyft": "ride_booking"}.get(domains[0], domains[0])
    prefix = _normalize_label(instruction_id).replace("_", "-")
    aliases = {
        "auto": "auto_repair",
        "coffee": "coffee_ordering",
        "movie": "movie_ticket",
        "pizza": "pizza_ordering",
        "restaurant": "restaurant_reservation",
        "uber": "ride_booking",
        "lyft": "ride_booking",
        "ride": "ride_booking",
    }
    return next((domain for key, domain in aliases.items() if prefix.startswith(key)), "task_oriented")


def extract_taskmaster_records(
    path: Path,
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    conversations = payload if isinstance(payload, list) else [payload]
    source_sha = file_sha256(path)
    rows: list[dict[str, Any]] = []
    for conversation in conversations:
        utterances = sorted(conversation.get("utterances", []), key=lambda row: row.get("index", 0))
        if len(utterances) < 2:
            continue
        text_values = [str(utterance.get("text", "")) for utterance in utterances]
        annotations = [_taskmaster_annotation_names(utterance) for utterance in utterances]
        flat_annotations = sorted({name for turn in annotations for name in turn})
        source_record_id = str(
            conversation.get("conversation_id")
            or conversation.get("conversationId")
            or _canonical_sha256(conversation)[:20]
        )
        instruction_id = str(
            conversation.get("instruction_id")
            or conversation.get("instructionId")
            or ""
        )
        rows.append(
            {
                "record_id": f"conversation-mine:taskmaster1:{source_record_id}",
                "source_dataset": source["dataset_id"],
                "source_record_id": source_record_id,
                "source_record_sha256": _canonical_sha256(conversation),
                "source_revision": source["revision"],
                "source_url": source["source_url"],
                "source_license": source["license"],
                "source_file_sha256": source_sha,
                "source_split": str(source.get("split", "unassigned")),
                "corpus_kind": "task_oriented",
                "domain": _taskmaster_domain(instruction_id, flat_annotations),
                "emotion": "",
                "turn_count": len(utterances),
                "speaker_pattern": [
                    _normalize_label(str(utterance.get("speaker", "unknown")))
                    for utterance in utterances
                ],
                "turn_signal_sequence": [
                    _taskmaster_turn_signal(turn) for turn in annotations
                ],
                "slot_types": sorted(
                    {slot for name in flat_annotations if (slot := _taskmaster_slot(name))}
                ),
                "question_pattern": [_is_question(text) for text in text_values],
                "utterance_length_buckets": [_length_bucket(text) for text in text_values],
                "source_text_retained": False,
                "extraction_version": EXTRACTION_VERSION,
            }
        )
    return rows


def _empathetic_speaker_pattern(speaker_ids: list[str]) -> list[str]:
    aliases: dict[str, str] = {}
    pattern = []
    for speaker_id in speaker_ids:
        if speaker_id not in aliases:
            aliases[speaker_id] = f"speaker_{chr(ord('a') + len(aliases))}"
        pattern.append(aliases[speaker_id])
    return pattern


def _empathetic_conversation(
    records: list[dict[str, str]],
    *,
    source: dict[str, Any],
    source_sha: str,
    split: str,
) -> dict[str, Any] | None:
    if len(records) < 2:
        return None
    records.sort(key=lambda row: int(row.get("utterance_idx", "0") or 0))
    utterances = [row.get("utterance", "") for row in records]
    source_record_id = records[0]["conv_id"]
    return {
        "record_id": f"conversation-mine:empathetic:{split}:{source_record_id}",
        "source_dataset": source["dataset_id"],
        "source_record_id": source_record_id,
        "source_record_sha256": _canonical_sha256(records),
        "source_revision": source["revision"],
        "source_url": source["source_url"],
        "source_license": source["license"],
        "source_file_sha256": source_sha,
        "source_split": split,
        "corpus_kind": "empathetic_conversation",
        "domain": "everyday_emotion",
        "emotion": _normalize_label(records[0].get("context", "")),
        "turn_count": len(records),
        "speaker_pattern": _empathetic_speaker_pattern(
            [row.get("speaker_idx", "unknown") for row in records]
        ),
        "turn_signal_sequence": [
            "question" if _is_question(text) else "statement" for text in utterances
        ],
        "slot_types": [],
        "question_pattern": [_is_question(text) for text in utterances],
        "utterance_length_buckets": [_length_bucket(text) for text in utterances],
        "source_text_retained": False,
        "extraction_version": EXTRACTION_VERSION,
    }


def extract_empathetic_records(
    path: Path,
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    source_sha = file_sha256(path)
    rows: list[dict[str, Any]] = []
    with tarfile.open(path, "r:gz") as archive:
        for split in ("train", "valid", "test"):
            member = archive.extractfile(f"empatheticdialogues/{split}.csv")
            if member is None:
                raise FileNotFoundError(f"empatheticdialogues/{split}.csv")
            reader = csv.DictReader(io.TextIOWrapper(member, encoding="utf-8"))
            current_id = ""
            records: list[dict[str, str]] = []
            for record in reader:
                conv_id = record["conv_id"]
                if current_id and conv_id != current_id:
                    row = _empathetic_conversation(
                        records,
                        source=source,
                        source_sha=source_sha,
                        split=split,
                    )
                    if row is not None:
                        rows.append(row)
                    records = []
                current_id = conv_id
                records.append(record)
            row = _empathetic_conversation(
                records,
                source=source,
                source_sha=source_sha,
                split=split,
            )
            if row is not None:
                rows.append(row)
    return rows


def load_conversation_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text())
    if registry.get("version") != 1 or not registry.get("sources"):
        raise ValueError("conversation source registry must be version 1 and non-empty")
    required = {
        "dataset_id",
        "kind",
        "filename",
        "download_url",
        "source_url",
        "revision",
        "license",
        "sha256",
    }
    for source in registry["sources"]:
        missing = sorted(required - source.keys())
        if missing:
            raise ValueError(f"source {source.get('dataset_id', '?')} is missing {missing}")
        if not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
            raise ValueError(f"source {source['dataset_id']} has an invalid sha256")
        if not re.fullmatch(r"[0-9a-f]{40}|artifact-sha256:[0-9a-f]{64}", source["revision"]):
            raise ValueError(f"source {source['dataset_id']} is not pinned")
    return registry


def fetch_conversation_sources(
    registry_path: Path,
    raw_root: Path,
) -> dict[str, Any]:
    registry = load_conversation_registry(registry_path)
    raw_root.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for source in registry["sources"]:
        destination = raw_root / source["filename"]
        if not destination.exists() or file_sha256(destination) != source["sha256"]:
            partial = destination.with_suffix(f"{destination.suffix}.partial")
            partial.unlink(missing_ok=True)
            request = urllib.request.Request(
                source["download_url"],
                headers={"User-Agent": "complexity-card-corpus/0.1"},
            )
            with urllib.request.urlopen(request) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            actual = file_sha256(partial)
            if actual != source["sha256"]:
                partial.unlink(missing_ok=True)
                raise ValueError(
                    f"sha256 mismatch for {source['dataset_id']}: {actual}"
                )
            partial.replace(destination)
        files[source["dataset_id"]] = {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": file_sha256(destination),
        }
    return {"files": files}


def _audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    schema_names = set(CONVERSATION_MINE_SCHEMA.names)
    prohibited_columns = sorted(schema_names & _PROSE_FIELD_NAMES)
    if prohibited_columns:
        raise ValueError(f"prose columns are forbidden: {prohibited_columns}")
    if any(row["source_text_retained"] for row in rows):
        raise ValueError("normalized conversation mine must not retain source text")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("conversation mine contains duplicate record IDs")
    turn_counts = Counter(row["turn_count"] for row in rows)
    length_buckets = Counter(
        bucket for row in rows for bucket in row["utterance_length_buckets"]
    )
    question_turns = sum(
        question for row in rows for question in row["question_pattern"]
    )
    total_turns = sum(row["turn_count"] for row in rows)
    speaker_patterns = Counter(" > ".join(row["speaker_pattern"]) for row in rows)
    return {
        "source_text_rows": 0,
        "prose_columns": [],
        "unique_record_ids": len(rows),
        "licenses": dict(sorted(Counter(row["source_license"] for row in rows).items())),
        "domains": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "emotions": dict(
            sorted(Counter(row["emotion"] for row in rows if row["emotion"]).items())
        ),
        "turn_counts": {
            str(turn_count): count for turn_count, count in sorted(turn_counts.items())
        },
        "utterance_length_buckets": dict(sorted(length_buckets.items())),
        "question_turn_rate": round(question_turns / total_turns, 6),
        "unique_speaker_patterns": len(speaker_patterns),
        "most_common_speaker_patterns": dict(speaker_patterns.most_common(10)),
    }


def build_conversation_mine(
    registry_path: Path,
    raw_root: Path,
    output_root: Path,
    *,
    max_rows_per_source: int | None = None,
) -> dict[str, Any]:
    if max_rows_per_source is not None and max_rows_per_source < 1:
        raise ValueError("max_rows_per_source must be positive")
    registry = load_conversation_registry(registry_path)
    rows: list[dict[str, Any]] = []
    source_files: dict[str, Any] = {}
    extractors = {
        "taskmaster_json": extract_taskmaster_records,
        "empathetic_tar": extract_empathetic_records,
    }
    for source in registry["sources"]:
        path = raw_root / source["filename"]
        if not path.exists():
            raise FileNotFoundError(path)
        actual_sha = file_sha256(path)
        if actual_sha != source["sha256"]:
            raise ValueError(f"sha256 mismatch for {source['dataset_id']}: {actual_sha}")
        if source["kind"] not in extractors:
            raise ValueError(f"unsupported conversation source kind: {source['kind']}")
        source_rows = extractors[source["kind"]](path, source)
        source_rows.sort(key=lambda row: _stable_order(row["record_id"]))
        if max_rows_per_source is not None:
            source_rows = source_rows[:max_rows_per_source]
        rows.extend(source_rows)
        source_files[source["dataset_id"]] = {
            "filename": source["filename"],
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
            "revision": source["revision"],
            "license": source["license"],
            "selected_records": len(source_rows),
        }
    rows.sort(key=lambda row: row["record_id"])
    audit = _audit_rows(rows)

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    table_path = temporary / "raw_records.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=CONVERSATION_MINE_SCHEMA),
        table_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    counts = {
        "records": len(rows),
        "records_by_source": dict(
            sorted(Counter(row["source_dataset"] for row in rows).items())
        ),
        "records_by_kind": dict(
            sorted(Counter(row["corpus_kind"] for row in rows).items())
        ),
    }
    manifest = {
        "format": EXTRACTION_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "structure-only source mine; not a training dataset",
        "generation_enabled": False,
        "counts": counts,
        "audit": audit,
        "sources": source_files,
        "files": {
            "raw_records.parquet": {
                "bytes": table_path.stat().st_size,
                "sha256": file_sha256(table_path),
            }
        },
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return manifest
