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
from complexity_card_corpus.scenario_language import DynamicNarrativeComposer


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
    assert audit["unique_titles"] == 2_000
    assert audit["unique_goals"] == 2_000
    assert audit["unique_triggers"] >= 100
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
    assert audit["model_generated_dialogue_rows"] == 0
    assert audit["payload_contract_match_ratio"] == 1.0
    assert audit["compatibility_match_ratio"] == 1.0
    assert audit["split_counts"] == {"train": 1_900, "validation": 100}
    assert audit["split_holdout_unit"] == "family+domain+intent"
    assert audit["split_group_overlap"] == 0
    assert sum(audit["validation_family_counts"].values()) == 100
    assert audit["surface_stats"]["documents"] == 2_000
    assert audit["surface_stats"]["unique_document_rate"] == 1.0
    assert audit["surface_stats"]["unique_sentence_rate"] >= 0.45
    assert 14 <= audit["surface_stats"]["mean_sentence_words"] <= 20
    assert 0.10 <= audit["surface_stats"]["transitions_per_sentence"] <= 0.30
    assert 0.25 <= audit["surface_stats"]["question_rate"] <= 0.30
    assert 0 < audit["surface_stats"]["raw_type_token_ratio"] <= 1
    assert 0 < audit["surface_stats"]["mattr_100"] <= 1
    assert "type_token_ratio" not in audit["surface_stats"]
    assert audit["surface_language_audit"]["issue_count"] == 0
    assert audit["surface_language_audit"]["checked_rows"] == 2_000
    assert audit["surface_language_audit"]["semantic_anchor_match_rate"] == 1.0
    assert audit["surface_language_audit"]["frame_family_cells"] == 84
    assert audit["morphology_audit"] == {
        "intent_phrases": 35,
        "unique_lemmas": 34,
        "forms_per_intent": 5,
        "forms_generated": 175,
        "unique_realized_forms": 140,
    }

    assert all(row["provenance"] == SCENARIO_PROVENANCE for row in rows)
    assert all(not row["model_generated_dialogue"] for row in rows)
    assert all(row["response_contract"] for row in rows)
    assert all(len(row["source_structure_keys"]) == 6 for row in rows)
    assert all(len(row["creation_hash"]) == 64 for row in rows)
    assert all(len(row["verification_hash"]) == 64 for row in rows)
    assert all(row["situation"] for row in rows)
    assert len({row["narrative_frame"] for row in rows}) == 12
    assert all(row["trigger"] for row in rows)
    assert all(
        row["situation"].lower().count(row["state"].rstrip(".").lower()) == 1
        for row in rows
    )
    assert all(
        row["situation"].lower().count(row["constraint"].rstrip(".").lower()) == 1
        for row in rows
    )


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


@pytest.mark.parametrize("matrix", ["domainIntents", "stateOutcomes"])
def test_scenario_registry_rejects_missing_new_compatibility_rows(
    matrix: str,
) -> None:
    payload = json.loads(REGISTRY.read_text())
    family = payload["families"][0]
    del family["compatibility"][matrix][next(iter(family["compatibility"][matrix]))]

    with pytest.raises(ValueError, match=f"{matrix} must cover exactly"):
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
        if value["family"] == "planning_comparison" and value["intent"] == "compare"
    )
    row["desired_outcome"] = "Steps are ordered by dependency with clear checkpoints."
    payload = json.loads(row["semantic_payload"])
    payload["success_condition"] = row["desired_outcome"]
    payload["decision_rule"] = row["desired_outcome"]
    row["semantic_payload"] = json.dumps(payload, sort_keys=True)
    row["verification_hash"] = _verification_hash(row)

    with pytest.raises(ValueError, match="compatibility violations"):
        audit_scenarios(rows, registry)


def test_domain_intent_and_state_outcome_rules_are_realized() -> None:
    registry = load_scenario_registry(REGISTRY)
    rows = compile_scenarios(registry)

    assert not any(
        row["family"] == "safety_uncertainty"
        and row["domain"] == "physical_safety"
        and row["intent"] == "preserve_privacy"
        for row in rows
    )
    urgent = [
        row
        for row in rows
        if row["family"] == "safety_uncertainty"
        and row["desired_outcome"]
        == "An urgent protective action is stated plainly when warranted."
    ]
    assert urgent
    assert all(
        row["state"] == "The available facts indicate an active risk." for row in urgent
    )


def test_validation_holds_out_complete_domain_intent_groups() -> None:
    registry = load_scenario_registry(REGISTRY)
    rows = compile_scenarios(registry)

    train_groups = {
        (row["family"], row["domain"], row["intent"])
        for row in rows
        if row["split"] == "train"
    }
    validation_groups = {
        (row["family"], row["domain"], row["intent"])
        for row in rows
        if row["split"] == "validation"
    }
    assert not train_groups & validation_groups
    assert sum(row["split"] == "validation" for row in rows) == 100


def test_safety_constraints_are_domain_specific() -> None:
    registry = load_scenario_registry(REGISTRY)
    rows = compile_scenarios(registry)
    safety = [row for row in rows if row["family"] == "safety_uncertainty"]

    assert not any(
        row["domain"] in {"financial_decision", "physical_safety"}
        and "diagnosis" in row["constraint"].lower()
        for row in safety
    )
    assert any(
        row["domain"] == "financial_decision"
        and "financial advice" in row["constraint"].lower()
        for row in safety
    )
    assert any(
        row["domain"] == "physical_safety"
        and "immediate hazards" in row["constraint"].lower()
        for row in safety
    )


def test_fallback_selection_uses_registry_priority() -> None:
    registry = load_scenario_registry(REGISTRY)
    rows = compile_scenarios(registry)

    critical_active = [
        row
        for row in rows
        if row["family"] == "safety_uncertainty"
        and row["risk_level"] == "critical"
        and row["state"] == "The available facts indicate an active risk."
    ]
    high_active = [
        row
        for row in rows
        if row["family"] == "safety_uncertainty"
        and row["risk_level"] == "high"
        and row["state"] == "The available facts indicate an active risk."
    ]
    assert critical_active and high_active
    assert {row["fallback"] for row in critical_active} == {
        "Recommend immediate local emergency help when there is imminent danger."
    }
    assert {row["fallback"] for row in high_active} == {
        "Direct the user to an official or qualified support channel."
    }


def test_dynamic_language_is_seeded_and_balances_frame_usage() -> None:
    registry = load_scenario_registry(REGISTRY)
    family = registry.families[0]
    domain = family.domains[0]
    intent = family.intents[0]
    constraint = family.constraints[0]
    state = family.states[0]
    outcome = family.outcomes[0]

    def sequence(seed: int) -> list[str]:
        composer = DynamicNarrativeComposer(seed)
        return [
            composer.compose(
                family.family_id,
                domain,
                intent,
                constraint,
                state,
                outcome,
                family.fallbacks[0],
            )[2]
            for _ in range(24)
        ]

    first = sequence(42)
    assert first == sequence(42)
    assert first != sequence(43)
    assert set(first) == {f"frame_{index:02d}" for index in range(1, 13)}
    assert {first.count(frame) for frame in set(first)} == {2}
