from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import statistics
import tarfile
import urllib.request
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from .build import file_sha256
from .conversation_mine import load_conversation_registry
from .surface_reference import (
    SurfaceStructureAccumulator,
    compare_surface_structures,
)


LEXICAL_MINE_VERSION = "aggregate-lexical-mine-v1"
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
VALID_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")

LEXICON_SCHEMA = pa.schema(
    [
        ("token", pa.string()),
        ("role", pa.string()),
        ("source_dataset", pa.string()),
        ("source_license", pa.string()),
        ("source_revision", pa.string()),
        ("occurrences", pa.int64()),
        ("document_count", pa.int64()),
        ("mined_unit", pa.string()),
        ("source_text_retained", pa.bool_()),
        ("release_ready", pa.bool_()),
        ("extraction_version", pa.string()),
    ]
)

TRANSITIONS = {
    "after",
    "although",
    "before",
    "because",
    "finally",
    "first",
    "following",
    "however",
    "instead",
    "meanwhile",
    "next",
    "once",
    "otherwise",
    "then",
    "therefore",
    "unless",
    "until",
    "when",
    "while",
}
INTENT_CUES = {"can", "could", "help", "let", "must", "please", "should", "to", "would"}
STATE_CUES = {"am", "are", "became", "become", "becomes", "feel", "feels", "is", "remain", "remains", "seem", "seems", "was", "were"}
CONSTRAINT_CUES = {"avoid", "cannot", "can't", "must", "never", "should", "without"}
OUTCOME_CUES = {"achieve", "ensure", "result", "results", "so", "successful", "successfully"}
BLOCKED_TOKENS = {
    "assistant",
    "http",
    "https",
    "speaker",
    "system",
    "unknown",
    "user",
    "www",
}


class _ApproxDistinct:
    """Fixed-memory linear counter for aggregate diversity statistics."""

    def __init__(self, bit_power: int = 24) -> None:
        self._bits = bytearray(1 << (bit_power - 3))
        self._mask = (1 << bit_power) - 1
        self._set_bits = 0

    def add(self, value: str) -> None:
        index = int.from_bytes(
            hashlib.blake2b(value.encode(), digest_size=8).digest(), "little"
        ) & self._mask
        byte_index, bit_index = divmod(index, 8)
        bit = 1 << bit_index
        if not self._bits[byte_index] & bit:
            self._bits[byte_index] |= bit
            self._set_bits += 1

    def estimate(self) -> int:
        slots = self._mask + 1
        empty = slots - self._set_bits
        if empty <= 0:
            return slots
        return round(-slots * math.log(empty / slots))


def _words(text: str) -> list[tuple[str, bool]]:
    return [
        (match.group(0).replace("’", "'").lower(), match.group(0)[:1].isupper())
        for match in TOKEN_PATTERN.finditer(text)
    ]


def _valid_word(token: str) -> bool:
    return (
        3 <= len(token) <= 24
        and token not in BLOCKED_TOKENS
        and VALID_TOKEN.fullmatch(token) is not None
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
            raise ValueError(f"source {source.get('dataset_id', '?')} is missing {missing}")
        if source["kind"] not in supported:
            raise ValueError(f"unsupported lexical source kind: {source['kind']}")
        if not re.fullmatch(r"[0-9a-f]{40}|artifact-sha256:[0-9a-f]{64}", source["revision"]):
            raise ValueError(f"source {source['dataset_id']} is not pinned")
        if not source["artifacts"]:
            raise ValueError(f"source {source['dataset_id']} has no artifacts")
        for artifact in source["artifacts"]:
            if {"filename", "download_url", "sha256"} - set(artifact):
                raise ValueError(f"source {source['dataset_id']} has an incomplete artifact")
            if not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]):
                raise ValueError(f"source {source['dataset_id']} has an invalid sha256")
    return payload


def _artifact_path(raw_root: Path, source: dict[str, Any], artifact: dict[str, Any]) -> Path:
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
            if not destination.exists() or file_sha256(destination) != artifact["sha256"]:
                partial = destination.with_suffix(f"{destination.suffix}.partial")
                partial.unlink(missing_ok=True)
                request = urllib.request.Request(
                    artifact["download_url"],
                    headers={"User-Agent": "complexity-card-corpus/0.1"},
                )
                with urllib.request.urlopen(request) as response, partial.open("wb") as output:
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


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return float(ordered[index])


def _source_stats(
    lengths: list[int],
    vocabulary: Counter[str],
    questions: int,
    retained: set[str],
    unique_documents_estimate: int,
    sentence_count: int,
    unique_sentences_estimate: int,
    surface_structure: dict[str, Any],
) -> dict[str, Any]:
    occurrences = sum(vocabulary.values())
    retained_occurrences = sum(vocabulary[token] for token in retained)
    return {
        "documents": len(lengths),
        "word_occurrences": occurrences,
        "observed_vocabulary": len(vocabulary),
        "mean_words": round(statistics.fmean(lengths), 3) if lengths else 0.0,
        "median_words": round(statistics.median(lengths), 3) if lengths else 0.0,
        "p95_words": _percentile(lengths, 0.95),
        "question_rate": round(questions / len(lengths), 6) if lengths else 0.0,
        "unique_document_rate_estimate": round(
            unique_documents_estimate / len(lengths), 6
        )
        if lengths
        else 0.0,
        "unique_sentence_rate_estimate": round(
            unique_sentences_estimate / sentence_count, 6
        )
        if sentence_count
        else 0.0,
        "distinct_counter": "linear_counting_2^24_bits",
        "type_token_ratio": round(len(vocabulary) / occurrences, 6)
        if occurrences
        else 0.0,
        "retained_vocabulary": len(retained),
        "retained_occurrence_coverage": round(retained_occurrences / occurrences, 6)
        if occurrences
        else 0.0,
        "surface_structure": surface_structure,
    }


_REPETITION_LEVELS = (
    ("unique", 1, 1),
    ("2-4", 2, 4),
    ("5-9", 5, 9),
    ("10-24", 10, 24),
    ("25+", 25, None),
)


def _unit_digest(value: str) -> bytes:
    return hashlib.blake2b(value.encode(), digest_size=16).digest()


def _repetition_profile(frequencies: Counter[bytes]) -> dict[str, Any]:
    total_occurrences = sum(frequencies.values())
    levels: dict[str, dict[str, int | float]] = {}
    for label, minimum, maximum in _REPETITION_LEVELS:
        values = [
            count
            for count in frequencies.values()
            if count >= minimum and (maximum is None or count <= maximum)
        ]
        occurrences = sum(values)
        levels[label] = {
            "units": len(values),
            "occurrences": occurrences,
            "occurrence_share": round(occurrences / total_occurrences, 6)
            if total_occurrences
            else 0.0,
        }
    repeated_occurrences = sum(max(0, count - 1) for count in frequencies.values())
    return {
        "counting_unit": "blake2b_128_digest_in_memory",
        "hashes_retained": False,
        "units": len(frequencies),
        "occurrences": total_occurrences,
        "maximum_occurrences": max(frequencies.values(), default=0),
        "repeated_occurrences": repeated_occurrences,
        "repeated_occurrence_share": round(
            repeated_occurrences / total_occurrences, 6
        )
        if total_occurrences
        else 0.0,
        "levels": levels,
    }


def _new_stats_accumulator() -> dict[str, Any]:
    return {
        "lengths": [],
        "vocabulary": Counter(),
        "questions": 0,
        "document_counter": _ApproxDistinct(),
        "sentence_counter": _ApproxDistinct(),
        "sentence_count": 0,
        "surface_structure": SurfaceStructureAccumulator(window_tokens=8),
        "document_frequencies": Counter(),
        "sentence_frequencies": Counter(),
    }


def _accumulate_stats(
    accumulator: dict[str, Any], text: str, tokens: list[str]
) -> None:
    accumulator["surface_structure"].add(text)
    accumulator["lengths"].append(len(tokens))
    accumulator["questions"] += int(text.rstrip().endswith("?"))
    normalized_document = " ".join(tokens)
    accumulator["document_counter"].add(normalized_document)
    accumulator["document_frequencies"][_unit_digest(normalized_document)] += 1
    accumulator["vocabulary"].update(tokens)
    for sentence in re.split(r"[.!?]+", text):
        if not sentence.strip():
            continue
        sentence_tokens = _normalized_tokens(sentence)
        if sentence_tokens:
            accumulator["sentence_count"] += 1
            normalized_sentence = " ".join(sentence_tokens)
            accumulator["sentence_counter"].add(normalized_sentence)
            accumulator["sentence_frequencies"][_unit_digest(normalized_sentence)] += 1


def _finalize_stats(
    accumulator: dict[str, Any], retained: set[str]
) -> dict[str, Any]:
    return {
        **_source_stats(
            accumulator["lengths"],
            accumulator["vocabulary"],
            accumulator["questions"],
            retained,
            accumulator["document_counter"].estimate(),
            accumulator["sentence_count"],
            accumulator["sentence_counter"].estimate(),
            accumulator["surface_structure"].summary(),
        ),
        "document_repetition": _repetition_profile(
            accumulator["document_frequencies"]
        ),
        "sentence_repetition": _repetition_profile(
            accumulator["sentence_frequencies"]
        ),
    }


def build_lexical_mine(
    registry_path: Path,
    raw_root: Path,
    output_root: Path,
    *,
    min_count: int = 8,
    max_capitalized_ratio: float = 0.65,
    delete_raw: bool = False,
    scenarios_path: Path | None = None,
) -> dict[str, Any]:
    if min_count < 1:
        raise ValueError("min_count must be positive")
    if not 0 <= max_capitalized_ratio <= 1:
        raise ValueError("max_capitalized_ratio must be between zero and one")
    registry = load_lexical_registry(registry_path)
    rows: list[dict[str, Any]] = []
    source_stats: dict[str, Any] = {}
    source_files: dict[str, Any] = {}
    reference_structure = SurfaceStructureAccumulator(window_tokens=8)
    for source in registry["sources"]:
        occurrences: Counter[tuple[str, str]] = Counter()
        document_counts: Counter[tuple[str, str]] = Counter()
        capitalized: Counter[str] = Counter()
        aggregate_stats = _new_stats_accumulator()
        role_stats: dict[str, dict[str, Any]] = {}
        artifact_rows: list[dict[str, Any]] = []
        for artifact in source["artifacts"]:
            path = _artifact_path(raw_root, source, artifact)
            if not path.exists():
                raise FileNotFoundError(path)
            actual_sha = file_sha256(path)
            if actual_sha != artifact["sha256"]:
                raise ValueError(
                    f"sha256 mismatch for {source['dataset_id']}/{artifact['filename']}: {actual_sha}"
                )
            artifact_rows.append(
                {
                    "filename": artifact["filename"],
                    "bytes": path.stat().st_size,
                    "sha256": actual_sha,
                }
            )
            for conversation_role, text in _source_role_documents(path, source):
                reference_structure.add(text)
                word_pairs = _words(text)
                tokens = [token for token, _ in word_pairs if _valid_word(token)]
                if not tokens:
                    continue
                _accumulate_stats(aggregate_stats, text, tokens)
                if conversation_role not in role_stats:
                    role_stats[conversation_role] = _new_stats_accumulator()
                role_accumulator = role_stats[conversation_role]
                _accumulate_stats(role_accumulator, text, tokens)
                for token, is_capitalized in word_pairs:
                    if _valid_word(token) and is_capitalized:
                        capitalized[token] += 1
                seen: set[tuple[str, str]] = set()
                for index, token in enumerate(tokens):
                    for role in _roles(tokens, index):
                        key = (token, role)
                        occurrences[key] += 1
                        seen.add(key)
                document_counts.update(seen)

        retained = {
            token
            for token, count in aggregate_stats["vocabulary"].items()
            if count >= min_count
            and capitalized[token] / count <= max_capitalized_ratio
        }
        for (token, role), count in sorted(occurrences.items()):
            if token not in retained or count < min_count:
                continue
            rows.append(
                {
                    "token": token,
                    "role": role,
                    "source_dataset": source["dataset_id"],
                    "source_license": source["license"],
                    "source_revision": source["revision"],
                    "occurrences": count,
                    "document_count": document_counts[(token, role)],
                    "mined_unit": "single_normalized_token",
                    "source_text_retained": False,
                    "release_ready": False,
                    "extraction_version": LEXICAL_MINE_VERSION,
                }
            )
        source_stats[source["dataset_id"]] = {
            **_finalize_stats(aggregate_stats, retained),
            "conversation_roles": {
                role: _finalize_stats(accumulator, retained)
                for role, accumulator in sorted(role_stats.items())
            },
        }
        source_files[source["dataset_id"]] = {
            "artifacts": artifact_rows,
            "revision": source["revision"],
            "license": source["license"],
            "origin": source.get("origin", "unspecified"),
        }

    rows.sort(key=lambda row: (row["source_dataset"], row["role"], row["token"]))
    if any(row["source_text_retained"] for row in rows):
        raise ValueError("lexical mine cannot retain source text")
    if any(" " in row["token"] for row in rows):
        raise ValueError("lexical mine cannot retain phrases")

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    table_path = temporary / "lexicon.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=LEXICON_SCHEMA),
        table_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    preview_path = temporary / "lexicon.json"
    preview_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    audit = {
        "source_text_retained": False,
        "max_retained_ngram_tokens": 1,
        "release_ready": False,
        "human_approval_required": True,
        "rows": len(rows),
        "unique_tokens": len({row["token"] for row in rows}),
        "roles": dict(sorted(Counter(row["role"] for row in rows).items())),
        "source_stats": source_stats,
        "reference_surface_structure": reference_structure.summary(),
    }
    if scenarios_path is not None:
        candidate_structure = SurfaceStructureAccumulator(window_tokens=8)
        candidate_structure.extend(
            str(row.get("situation", "")) for row in _scenario_rows(scenarios_path)
        )
        audit["surface_structure_comparison"] = compare_surface_structures(
            reference_structure, candidate_structure
        )
    (temporary / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in (table_path, preview_path, temporary / "audit.json")
    }
    manifest = {
        "format": LEXICAL_MINE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "private aggregate lexical candidates; not a training dataset",
        "min_count": min_count,
        "max_capitalized_ratio": max_capitalized_ratio,
        "sources": source_files,
        "audit": audit,
        "files": files,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    if delete_raw:
        for source in registry["sources"]:
            for artifact in source["artifacts"]:
                _artifact_path(raw_root, source, artifact).unlink(missing_ok=True)
        for directory in sorted(raw_root.glob("*")):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
    return manifest


def _normalized_tokens(text: str) -> list[str]:
    return [token for token, _ in _words(text)]


def _window_hashes(tokens: list[str], size: int) -> set[bytes]:
    return {
        hashlib.sha256(" ".join(tokens[index : index + size]).encode()).digest()
        for index in range(max(0, len(tokens) - size + 1))
    }


def _scenario_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        if (path / "scenarios.parquet").exists():
            path = path / "scenarios.parquet"
        else:
            path = path / "scenarios.jsonl"
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_source_overlap(
    registry_path: Path,
    raw_root: Path,
    scenarios_path: Path,
    *,
    window_tokens: int = 8,
    fail_on_match: bool = True,
) -> dict[str, Any]:
    if window_tokens < 6:
        raise ValueError("overlap windows shorter than six tokens are too noisy")
    registry = load_lexical_registry(registry_path)
    source_windows: set[bytes] = set()
    source_documents = 0
    for source in registry["sources"]:
        for artifact in source["artifacts"]:
            path = _artifact_path(raw_root, source, artifact)
            if file_sha256(path) != artifact["sha256"]:
                raise ValueError(f"sha256 mismatch for {source['dataset_id']}")
            for text in _source_documents(path, source):
                source_documents += 1
                source_windows.update(
                    _window_hashes(_normalized_tokens(text), window_tokens)
                )

    matched_ids: list[str] = []
    generated_windows = 0
    for row in _scenario_rows(scenarios_path):
        matched = False
        for field in ("title", "trigger", "situation", "goal"):
            hashes = _window_hashes(
                _normalized_tokens(str(row.get(field, ""))), window_tokens
            )
            generated_windows += len(hashes)
            if hashes & source_windows:
                matched = True
        if matched:
            matched_ids.append(str(row["scenario_id"]))
    report = {
        "window_tokens": window_tokens,
        "source_documents_scanned": source_documents,
        "source_windows_hashed_transiently": len(source_windows),
        "generated_windows_checked": generated_windows,
        "matched_scenarios": len(matched_ids),
        "matched_scenario_ids": sorted(matched_ids),
        "source_text_retained": False,
        "source_window_hashes_retained": False,
        "passed": not matched_ids,
    }
    if matched_ids and fail_on_match:
        raise ValueError(
            f"found source overlap in {len(matched_ids)} generated scenarios"
        )
    return report
