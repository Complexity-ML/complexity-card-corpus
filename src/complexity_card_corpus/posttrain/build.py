from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..build import file_sha256
from ..sft.schema import INSTRUCTION_SCHEMA
from .capacity import post_training_capacity_report
from .constants import DATASET_ID, DATASET_LICENSE, DATASET_SOURCE
from .metrics import _audit
from .rendering import (
    _apply_vocabulary_placements,
    _balance_conversation_families,
    _conversation_rows,
    _deduplicate_conversation_rows,
    _load_vocabulary_placements,
    _render_conversation_rows,
)
from .review import _review_sample


def build_post_training_corpus(
    scenarios_path: Path,
    output_root: Path,
    *,
    variants_per_scenario: int = 8,
    review_scenarios: int = 140,
    seed: int = 42,
    vocabulary_placement_path: Path | None = None,
    workers: int = 1,
    target_rows: int | None = None,
    max_examples_per_family: int | None = None,
) -> dict[str, Any]:
    if variants_per_scenario < 1:
        raise ValueError("variants_per_scenario must be positive")
    if review_scenarios and variants_per_scenario < 2:
        raise ValueError(
            "variants_per_scenario must be at least 2 when building the human "
            "review set because every reviewed scenario requires one instruct "
            "and one chat projection"
        )
    if workers < 1:
        raise ValueError("workers must be positive")
    scenarios = pq.read_table(scenarios_path).to_pylist()
    placements = (
        _load_vocabulary_placements(vocabulary_placement_path)
        if vocabulary_placement_path is not None
        else []
    )
    rows = _parallel_conversation_rows(
        scenarios,
        variants_per_scenario,
        vocabulary_placements=placements,
        workers=workers,
    )
    rows, family_balance = _balance_conversation_families(
        rows,
        max_examples_per_family=max_examples_per_family,
    )
    audit = _audit(rows)
    audit["family_balance"] = family_balance
    audit["scale_100k"] = post_training_capacity_report(
        source_cards=len(scenarios),
        configured_variants_per_source_card=variants_per_scenario,
        audit=audit,
        target_rows=target_rows,
    )
    observed_lexical_focus = {
        json.loads(row["answer_json"])["lexical_focus"]
        for row in rows
        if json.loads(row["answer_json"])["lexical_focus"]
    }
    requested_lexical_focus = {row["token"] for row in placements}
    if observed_lexical_focus != requested_lexical_focus:
        missing = sorted(requested_lexical_focus - observed_lexical_focus)
        raise ValueError(f"vocabulary placement coverage is incomplete: {missing[:5]}")
    realized_placements: dict[str, tuple[str, str]] = {}
    for row in rows:
        answer = json.loads(row["answer_json"])
        if answer["lexical_focus"]:
            realized_placements[answer["lexical_focus"]] = (
                answer["lexical_assignment_method"],
                answer["family"],
            )
    lexical_methods = Counter(
        method for method, _family in realized_placements.values()
    )
    lexical_families = Counter(
        family for _method, family in realized_placements.values()
    )
    audit["vocabulary_placement"] = {
        "requested_tokens": len(requested_lexical_focus),
        "observed_tokens": len(observed_lexical_focus),
        "coverage_ratio": 1.0 if placements else None,
        "mapped_conversation_rows": sum(
            bool(json.loads(row["answer_json"])["lexical_focus"]) for row in rows
        ),
        "surfaced_conversation_rows": 0,
        "assignment_methods": dict(sorted(lexical_methods.items())),
        "family_counts": dict(sorted(lexical_families.items())),
        "surface_policy": "metadata_only" if placements else None,
        "automatic_definition_generation": False,
    }
    review = _review_sample(rows, review_scenarios=review_scenarios, seed=seed)

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    conversations_path = temporary / "conversations.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=INSTRUCTION_SCHEMA),
        conversations_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    audit_path = temporary / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    review_path = temporary / "human_review.csv"
    with review_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    manifest = {
        "format": "complexity-post-training-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "license": DATASET_LICENSE,
        "source": DATASET_SOURCE,
        "input": {"path": str(scenarios_path), "sha256": file_sha256(scenarios_path)},
        "vocabulary_placement_input": (
            {
                "path": str(vocabulary_placement_path),
                "sha256": file_sha256(vocabulary_placement_path),
            }
            if vocabulary_placement_path is not None
            else None
        ),
        "variants_per_scenario": variants_per_scenario,
        "max_examples_per_family": max_examples_per_family,
        "workers": workers,
        "audit": audit,
        "human_review": {
            "rows": len(review),
            "source_scenarios": len({row["scenario_id"] for row in review}),
            "sample_fraction_of_source_scenarios": round(
                len({row["scenario_id"] for row in review}) / audit["source_scenarios"],
                6,
            ),
            "modes_per_source_scenario": ["instruct", "chat"],
            "status": "pending",
            "strata": ["family", "risk_level", "split", "domain"],
            "sampling_scope": (
                "deterministic stratified quality-control sample; not a simple "
                "random estimate of corpus-wide defect prevalence"
            ),
            "required_before_training": True,
        },
        "training_ready": False,
        "release_ready": False,
    }
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return {
        **manifest,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in (
                output_root / "conversations.parquet",
                output_root / "audit.json",
                output_root / "human_review.csv",
                output_root / "manifest.json",
            )
        },
    }


def _render_conversation_chunk(
    arguments: tuple[list[dict[str, Any]], int],
) -> list[dict[str, Any]]:
    scenarios, variants_per_scenario = arguments
    return _render_conversation_rows(scenarios, variants_per_scenario)


def _parallel_conversation_rows(
    scenarios: list[dict[str, Any]],
    variants_per_scenario: int,
    *,
    vocabulary_placements: list[dict[str, str]],
    workers: int,
) -> list[dict[str, Any]]:
    """Render in processes while retaining canonical global selection."""
    if workers == 1 or len(scenarios) < 2:
        return _conversation_rows(
            scenarios,
            variants_per_scenario,
            vocabulary_placements=vocabulary_placements,
        )

    enriched = (
        _apply_vocabulary_placements(scenarios, vocabulary_placements)
        if vocabulary_placements
        else scenarios
    )
    worker_count = min(workers, len(enriched), os.cpu_count() or 1)
    chunk_size = (len(enriched) + worker_count - 1) // worker_count
    chunks = [
        enriched[start : start + chunk_size]
        for start in range(0, len(enriched), chunk_size)
    ]
    arguments = [(chunk, variants_per_scenario) for chunk in chunks]

    def collect(executor: ProcessPoolExecutor | ThreadPoolExecutor) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for chunk_rows in executor.map(_render_conversation_chunk, arguments):
            collected.extend(chunk_rows)
        return collected

    try:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            raw_rows = collect(executor)
    except (NotImplementedError, OSError, PermissionError):
        # Some sandboxed macOS runtimes deny POSIX semaphore introspection.
        # Normal local builds still take the process path above.
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            raw_rows = collect(executor)
    return _deduplicate_conversation_rows(raw_rows)
