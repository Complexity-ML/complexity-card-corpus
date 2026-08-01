from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..build import file_sha256
from ..scenario_language import NARRATIVE_FRAME_IDS
from .audit import audit_scenarios
from .compiler import compile_scenarios
from .schema import (
    SCENARIO_FORGE_VERSION,
    SCENARIO_PROVENANCE,
    SCENARIO_SCHEMA,
    ScenarioForgeRegistry,
)


def load_scenario_registry(path: Path) -> ScenarioForgeRegistry:
    payload = json.loads(path.read_text())
    families = list(payload.get("families", []))
    for include in payload.get("includes", []):
        include_path = path.parent / include
        include_payload = json.loads(include_path.read_text())
        if include_payload.get("format") != "scenario-family-pack-v1":
            raise ValueError(f"unsupported Scenario Forge family pack: {include_path}")
        included_families = include_payload.get("families")
        if not isinstance(included_families, list) or not included_families:
            raise ValueError(f"empty Scenario Forge family pack: {include_path}")
        families.extend(included_families)
    payload["families"] = families
    return ScenarioForgeRegistry.model_validate(payload)


def build_scenario_forge(
    registry_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    registry = load_scenario_registry(registry_path)
    rows = compile_scenarios(registry)
    audit = audit_scenarios(rows, registry)

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    parquet_path = temporary / "scenarios.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCENARIO_SCHEMA),
        parquet_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    jsonl_path = temporary / "scenarios.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    audit_path = temporary / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    files = {}
    for path in (parquet_path, jsonl_path, audit_path):
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    manifest = {
        "format": SCENARIO_FORGE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": registry.metadata.model_dump(mode="json", by_alias=True),
        "seed": registry.seed,
        "validation_percent": registry.validation_percent,
        "input": {
            "registry": {
                "path": str(registry_path),
                "sha256": file_sha256(registry_path),
            },
            "family_packs": [
                {
                    "path": str(registry_path.parent / include),
                    "sha256": file_sha256(registry_path.parent / include),
                }
                for include in registry.includes
            ],
        },
        "counts": {
            "scenarios": len(rows),
            "families": len(registry.families),
            "by_family": audit["family_counts"],
            "by_split": audit["split_counts"],
        },
        "generation": {
            "model_generated_dialogue": False,
            "third_party_utterances_accessed": False,
            "language_selection": "seeded_dynamic_least_used",
            "narrative_frames": len(NARRATIVE_FRAME_IDS),
            "provenance": SCENARIO_PROVENANCE,
        },
        "audit": audit,
        "files": files,
    }
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return manifest
