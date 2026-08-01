from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ..build import file_sha256
from .lexical_schema import _words
from .lexical_sources import (
    _artifact_path,
    _source_documents,
    load_lexical_registry,
)


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
