from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from complexity_card_corpus.scenario_forge import (
    SCENARIO_PROVENANCE,
    ScenarioForgeRegistry,
    _verification_hash,
    audit_scenarios,
    build_scenario_forge,
    compile_scenarios,
    load_scenario_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/scenario-forge/scenario-forge-v1.json"


def test_scenario_forge_compiles_two_thousand_semantic_cards() -> None:
    registry = load_scenario_registry(REGISTRY)
    rows = compile_scenarios(registry)
    audit = audit_scenarios(rows, registry)

    assert len(rows) == 2_000
    assert audit["unique_ids"] == 2_000
    assert audit["unique_semantic_signatures"] == 2_000
    assert audit["unique_semantic_payloads"] == 2_000
    assert audit["unique_situations"] == 2_000
    assert audit["unique_creation_hashes"] == 2_000
    assert audit["unique_verification_hashes"] == 2_000
    assert audit["family_counts"] == {
        "conversation_empathy": 150,
        "explanation_learning": 400,
        "planning_comparison": 200,
        "practical_action": 600,
        "safety_uncertainty": 100,
        "troubleshooting": 300,
        "writing_transformation": 250,
    }
    assert audit["surface_text_rows"] == 0
    assert audit["payload_contract_match_ratio"] == 1.0
    assert audit["compatibility_match_ratio"] == 1.0
    assert set(audit["split_counts"]) == {"train", "validation"}

    assert all(row["provenance"] == SCENARIO_PROVENANCE for row in rows)
    assert all(not row["surface_text_generated"] for row in rows)
    assert all(row["response_contract"] for row in rows)
    assert all(len(row["source_structure_keys"]) == 3 for row in rows)
    assert all(len(row["creation_hash"]) == 64 for row in rows)
    assert all(len(row["verification_hash"]) == 64 for row in rows)
    assert all(row["situation"] for row in rows)


def test_scenario_forge_output_is_deterministic_and_inspectable(tmp_path: Path) -> None:
    first = build_scenario_forge(REGISTRY, tmp_path / "first")
    second = build_scenario_forge(REGISTRY, tmp_path / "second")

    for filename in ("scenarios.parquet", "scenarios.jsonl", "audit.json"):
        assert first["files"][filename]["sha256"] == second["files"][filename]["sha256"]

    rows = pq.read_table(tmp_path / "first/scenarios.parquet").to_pylist()
    jsonl_rows = [
        json.loads(line)
        for line in (tmp_path / "first/scenarios.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len(jsonl_rows) == 2_000
    assert rows[0]["scenario_id"] == jsonl_rows[0]["scenario_id"]
    assert json.loads(rows[0]["semantic_payload"])["subject"]


def test_scenario_forge_balances_every_family_across_domains() -> None:
    registry = load_scenario_registry(REGISTRY)
    audit = audit_scenarios(compile_scenarios(registry), registry)

    for domain_counts in audit["domain_counts"].values():
        counts = list(domain_counts.values())
        assert max(counts) - min(counts) <= 1
    for family in registry.families:
        assert audit["axis_coverage"][family.family_id] == {
            "domains": len(family.domains),
            "intents": len(family.intents),
            "constraints": len(family.constraints),
            "states": len(family.states),
            "outcomes": len(family.outcomes),
        }


def test_scenario_registry_rejects_insufficient_semantic_capacity() -> None:
    payload = json.loads(REGISTRY.read_text())
    family = payload["families"][0]
    family["target"] = (
        len(family["domains"])
        * len(family["intents"])
        * len(family["constraints"])
        * len(family["states"])
        * len(family["outcomes"])
        + 1
    )

    with pytest.raises(ValueError, match="semantic signature capacity"):
        ScenarioForgeRegistry.model_validate(payload)


def test_scenario_registry_rejects_duplicate_axis_ids() -> None:
    payload = json.loads(REGISTRY.read_text())
    payload["families"][0]["constraints"][1]["id"] = payload["families"][0][
        "constraints"
    ][0]["id"]

    with pytest.raises(ValueError, match="duplicate identifiers"):
        ScenarioForgeRegistry.model_validate(payload)


def test_scenario_registry_rejects_incomplete_compatibility_matrix() -> None:
    payload = json.loads(REGISTRY.read_text())
    del payload["families"][0]["compatibility"]["intentOutcomes"]["arrange"]

    with pytest.raises(ValueError, match="intentOutcomes must cover exactly"):
        ScenarioForgeRegistry.model_validate(payload)


def test_scenario_registry_rejects_unknown_compatibility_reference() -> None:
    payload = json.loads(REGISTRY.read_text())
    payload["families"][0]["compatibility"]["intentOutcomes"]["arrange"] = [
        "invented_outcome"
    ]

    with pytest.raises(ValueError, match="references unknown IDs"):
        ScenarioForgeRegistry.model_validate(payload)


def test_scenario_registry_requires_fallback_for_every_risk_state_pair() -> None:
    payload = json.loads(REGISTRY.read_text())
    payload["families"][0]["compatibility"]["riskFallbacks"]["medium"] = [
        "official_channel"
    ]

    with pytest.raises(ValueError, match="has no fallback for domain"):
        ScenarioForgeRegistry.model_validate(payload)


def test_audit_rejects_tampered_scenario_content() -> None:
    registry = load_scenario_registry(REGISTRY)
    rows = compile_scenarios(registry)
    rows[0]["situation"] = "Tampered after compilation."

    with pytest.raises(ValueError, match="verification hash mismatch"):
        audit_scenarios(rows, registry)


def test_audit_rejects_semantically_incompatible_outcome() -> None:
    registry = load_scenario_registry(REGISTRY)
    rows = compile_scenarios(registry)
    row = next(
        value
        for value in rows
        if value["family"] == "planning_comparison"
        and value["intent"] == "compare"
    )
    row["desired_outcome"] = "Steps are ordered by dependency with clear checkpoints."
    payload = json.loads(row["semantic_payload"])
    payload["success_condition"] = row["desired_outcome"]
    payload["decision_rule"] = row["desired_outcome"]
    row["semantic_payload"] = json.dumps(payload, sort_keys=True)
    row["verification_hash"] = _verification_hash(row)

    with pytest.raises(ValueError, match="compatibility violations"):
        audit_scenarios(rows, registry)
