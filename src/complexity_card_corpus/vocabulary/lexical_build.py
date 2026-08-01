from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..build import file_sha256
from ..surface_reference import SurfaceStructureAccumulator, compare_surface_structures
from .lexical_audit import _scenario_rows
from .lexical_schema import LEXICAL_MINE_VERSION, LEXICON_SCHEMA, _valid_word, _words
from .lexical_sources import (
    _artifact_path,
    _source_role_documents,
    _roles,
    load_lexical_registry,
)
from .lexical_statistics import (
    _accumulate_stats,
    _finalize_stats,
    _new_stats_accumulator,
)


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
