from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from complexity_card_corpus.scenario_integrity import (
    verification_hash as _verification_hash,
)
from complexity_card_corpus.scenarios import (
    SCENARIO_PROVENANCE,
    ScenarioForgeRegistry,
    audit_scenarios,
    build_scenario_forge,
    compile_scenarios,
    load_scenario_registry,
    audit_scenario_tanks,
)
from complexity_card_corpus.scenario_language import DynamicNarrativeComposer


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/scenario-forge/scenario-forge-v1.json"
EXPECTED_FAMILY_COUNTS = {
    "brainstorming_creativity": 5_000,
    "conversation_empathy": 2_500,
    "context_clarification": 5_000,
    "critique_revision": 6_000,
    "explanation_learning": 4_000,
    "extraction_classification": 7_000,
    "grounded_qa": 7_000,
    "planning_comparison": 3_000,
    "practical_action": 3_500,
    "reasoning_verification": 5_000,
    "safety_uncertainty": 3_500,
    "summarization_synthesis": 6_000,
    "troubleshooting": 3_500,
    "writing_transformation": 4_000,
}
EXPECTED_SCENARIOS = sum(EXPECTED_FAMILY_COUNTS.values())


def _expanded_registry_payload() -> dict:
    return load_scenario_registry(REGISTRY).model_dump(mode="json", by_alias=True)


def test_registry_keeps_one_editable_raw_data_tank_per_family() -> None:
    root_payload = json.loads(REGISTRY.read_text())

    assert root_payload["families"] == []
    assert len(root_payload["includes"]) == len(EXPECTED_FAMILY_COUNTS)
    assert all(include.startswith("tanks/") for include in root_payload["includes"])

    tank_ids = []
    domain_contexts = []
    intent_templates = []
    for include in root_payload["includes"]:
        tank_path = REGISTRY.parent / include
        tank = json.loads(tank_path.read_text())
        assert tank["format"] == "scenario-family-pack-v1"
        assert len(tank["families"]) == 1
        family = tank["families"][0]
        assert tank["tankId"] == family["id"] == tank_path.stem
        assert family["weight"] == EXPECTED_FAMILY_COUNTS[family["id"]]
        # A tank must contain authored subject matter, not merely enough
        # Cartesian combinations to reach its generated-row target.
        assert len(family["domains"]) >= 8
        assert len(family["intents"]) >= 5
        assert len(family["constraints"]) >= 5
        assert len(family["states"]) >= 4
        assert len(family["outcomes"]) >= 5
        assert len(family["fallbacks"]) >= 3
        tank_ids.append(tank["tankId"])
        domain_contexts.extend(domain["context"] for domain in family["domains"])
        intent_templates.extend(
            intent["goalTemplate"] for intent in family["intents"]
        )

    assert set(tank_ids) == set(EXPECTED_FAMILY_COUNTS)
    assert len(domain_contexts) == len(set(domain_contexts))
    assert len(intent_templates) == len(set(intent_templates))


def test_tank_hydration_audit_exposes_capacity_without_inventing_rows() -> None:
    audit = audit_scenario_tanks(REGISTRY)

    assert audit["tank_count"] == len(EXPECTED_FAMILY_COUNTS)
    assert set(audit["tanks"]) == set(EXPECTED_FAMILY_COUNTS)
    for family_id, tank in audit["tanks"].items():
        assert tank["path"] == f"tanks/{family_id}.json"
        assert tank["allocation_weight"] == EXPECTED_FAMILY_COUNTS[family_id]
        assert tank["compatible_signature_capacity"] >= tank["allocation_weight"]
        assert tank["unused_signature_capacity_at_baseline"] >= 0
        assert tank["raw_atom_count"] == sum(tank["raw_atom_counts"].values())
        assert tank["raw_atom_count"] >= 40
        assert tank["raw_atom_counts"]["domains"] >= 8
        assert tank["capacity_reserve_ratio"] >= 1.5
        assert tank["hydrated_for_scale"] is True


def test_scenario_forge_compiles_semantic_cards_from_family_tanks() -> None:
    registry = load_scenario_registry(REGISTRY)
    rows = compile_scenarios(registry, target_scenarios=EXPECTED_SCENARIOS)
    audit = audit_scenarios(rows, registry)

    assert len(rows) == EXPECTED_SCENARIOS
    assert audit["unique_ids"] == EXPECTED_SCENARIOS
    assert audit["unique_semantic_signatures"] == EXPECTED_SCENARIOS
    assert audit["unique_semantic_payloads"] == EXPECTED_SCENARIOS
    assert audit["unique_situations"] == EXPECTED_SCENARIOS
    assert audit["unique_titles"] == EXPECTED_SCENARIOS
    assert audit["unique_goals"] == EXPECTED_SCENARIOS
    assert audit["unique_triggers"] >= 100
    assert audit["unique_creation_hashes"] == EXPECTED_SCENARIOS
    assert audit["unique_verification_hashes"] == EXPECTED_SCENARIOS
    assert audit["family_counts"] == EXPECTED_FAMILY_COUNTS
    assert audit["model_generated_dialogue_rows"] == 0
    assert audit["payload_contract_match_ratio"] == 1.0
    assert audit["compatibility_match_ratio"] == 1.0
    assert sum(audit["split_counts"].values()) == EXPECTED_SCENARIOS
    assert audit["split_holdout_unit"] == "family+domain+intent"
    assert audit["split_group_overlap"] == 0
    assert abs(audit["validation_row_delta"]) <= audit["validation_tolerance_rows"]
    assert 4.9 <= audit["validation_percent_actual"] <= 5.1
    assert audit["surface_stats"]["documents"] == EXPECTED_SCENARIOS
    assert audit["surface_stats"]["unique_document_rate"] == 1.0
    assert audit["surface_stats"]["unique_sentence_rate"] >= 0.45
    assert 14 <= audit["surface_stats"]["mean_sentence_words"] <= 20
    assert 0.10 <= audit["surface_stats"]["transitions_per_sentence"] <= 0.30
    assert 0.25 <= audit["surface_stats"]["question_rate"] <= 0.30
    assert 0 < audit["surface_stats"]["raw_type_token_ratio"] <= 1
    assert 0 < audit["surface_stats"]["mattr_100"] <= 1
    assert "type_token_ratio" not in audit["surface_stats"]
    assert audit["surface_language_audit"]["issue_count"] == 0
    assert audit["surface_language_audit"]["checked_rows"] == EXPECTED_SCENARIOS
    assert audit["surface_language_audit"]["semantic_anchor_match_rate"] == 1.0
    assert audit["surface_language_audit"]["frame_family_cells"] == 336
    assert audit["morphology_audit"] == {
        "intent_phrases": 84,
        "unique_lemmas": 62,
        "forms_per_intent": 5,
        "forms_generated": 420,
        "unique_realized_forms": 337,
    }
    card_staticity = audit["card_staticity"]
    assert card_staticity["hands"] == EXPECTED_SCENARIOS
    assert card_staticity["unique_hands"] == EXPECTED_SCENARIOS
    assert card_staticity["exact_hand_uniqueness_ratio"] == 1.0
    assert card_staticity["static_axes"] == []
    assert set(card_staticity["axes"]) == {
        "family",
        "domain",
        "intent",
        "constraint",
        "state",
        "outcome",
        "fallback",
        "risk",
    }
    assert all(
        metrics["unique_values"] >= 2
        for metrics in card_staticity["axes"].values()
    )

    assert all(row["provenance"] == SCENARIO_PROVENANCE for row in rows)
    assert all(not row["model_generated_dialogue"] for row in rows)
    assert all(row["response_contract"] for row in rows)
    assert all(len(row["source_structure_keys"]) == 8 for row in rows)
    assert all(len(row["source_structure_links"]) == 12 for row in rows)
    assert all(
        set(row["source_structure_keys"])
        == {
            card
            for link in row["source_structure_links"]
            for card in link.split("->", maxsplit=1)
        }
        for row in rows
    )
    assert all(
        f"family:{row['family']}->domain:{row['domain']}"
        in row["source_structure_links"]
        for row in rows
    )
    assert audit["source_graph"] == {
        "cards_per_scenario": 8,
        "links_per_scenario": 12,
        "orphan_cards": 0,
        "connected_scenarios": EXPECTED_SCENARIOS,
        "minimum_card_degree": 2,
        "unique_cards": 629,
        "unique_links": 6_941,
    }
    assert all(len(row["creation_hash"]) == 64 for row in rows)
    assert all(len(row["verification_hash"]) == 64 for row in rows)
    assert all(row["situation"] for row in rows)
    assert len({row["narrative_frame"] for row in rows}) == 24
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
    first = build_scenario_forge(
        REGISTRY,
        tmp_path / "first",
        target_scenarios=EXPECTED_SCENARIOS,
    )
    second = build_scenario_forge(
        REGISTRY,
        tmp_path / "second",
        target_scenarios=EXPECTED_SCENARIOS,
    )

    for filename in ("scenarios.parquet", "scenarios.jsonl", "audit.json"):
        assert first["files"][filename]["sha256"] == second["files"][filename]["sha256"]

    rows = pq.read_table(tmp_path / "first/scenarios.parquet").to_pylist()
    jsonl_rows = [
        json.loads(line)
        for line in (tmp_path / "first/scenarios.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len(jsonl_rows) == EXPECTED_SCENARIOS
    assert rows[0]["scenario_id"] == jsonl_rows[0]["scenario_id"]
    assert json.loads(rows[0]["semantic_payload"])["subject"]


def test_scenario_forge_balances_every_family_across_domains() -> None:
    registry = load_scenario_registry(REGISTRY)
    audit = audit_scenarios(
        compile_scenarios(registry, target_scenarios=EXPECTED_SCENARIOS),
        registry,
    )

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


def test_weak_generalist_families_have_broad_linked_subject_decks() -> None:
    registry = load_scenario_registry(REGISTRY)
    families = {family.family_id: family for family in registry.families}

    for family_id in ("grounded_qa", "explanation_learning", "critique_revision"):
        family = families[family_id]
        assert len(family.domains) >= 14
        assert all(
            len(family.compatibility.domain_intents[domain.domain_id]) >= 5
            for domain in family.domains
        )
        assert all(
            len(family.compatibility.domain_constraints[domain.domain_id]) >= 4
            for domain in family.domains
        )


def test_scenario_registry_rejects_insufficient_semantic_capacity() -> None:
    payload = _expanded_registry_payload()
    family = payload["families"][0]
    family["weight"] = (
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
    payload = _expanded_registry_payload()
    payload["families"][0]["constraints"][1]["id"] = payload["families"][0][
        "constraints"
    ][0]["id"]

    with pytest.raises(ValueError, match="duplicate identifiers"):
        ScenarioForgeRegistry.model_validate(payload)


def test_scenario_registry_rejects_incomplete_compatibility_matrix() -> None:
    payload = _expanded_registry_payload()
    del payload["families"][0]["compatibility"]["intentOutcomes"]["arrange"]

    with pytest.raises(ValueError, match="intentOutcomes must cover exactly"):
        ScenarioForgeRegistry.model_validate(payload)


@pytest.mark.parametrize("matrix", ["domainIntents", "stateOutcomes"])
def test_scenario_registry_rejects_missing_new_compatibility_rows(
    matrix: str,
) -> None:
    payload = _expanded_registry_payload()
    family = payload["families"][0]
    del family["compatibility"][matrix][next(iter(family["compatibility"][matrix]))]

    with pytest.raises(ValueError, match=f"{matrix} must cover exactly"):
        ScenarioForgeRegistry.model_validate(payload)


def test_scenario_registry_rejects_unknown_compatibility_reference() -> None:
    payload = _expanded_registry_payload()
    payload["families"][0]["compatibility"]["intentOutcomes"]["arrange"] = [
        "invented_outcome"
    ]

    with pytest.raises(ValueError, match="references unknown IDs"):
        ScenarioForgeRegistry.model_validate(payload)


def test_scenario_registry_requires_fallback_for_every_risk_state_pair() -> None:
    payload = _expanded_registry_payload()
    payload["families"][0]["compatibility"]["riskFallbacks"]["medium"] = [
        "official_channel"
    ]

    with pytest.raises(ValueError, match="has no fallback for domain"):
        ScenarioForgeRegistry.model_validate(payload)


def test_audit_rejects_tampered_scenario_content() -> None:
    registry = load_scenario_registry(REGISTRY)
    rows = compile_scenarios(registry, target_scenarios=EXPECTED_SCENARIOS)
    rows[0]["situation"] = "Tampered after compilation."

    with pytest.raises(ValueError, match="verification hash mismatch"):
        audit_scenarios(rows, registry)


def test_audit_rejects_a_well_formed_but_false_source_graph() -> None:
    registry = load_scenario_registry(REGISTRY)
    rows = compile_scenarios(registry, target_scenarios=EXPECTED_SCENARIOS)
    row = rows[0]
    row["source_structure_links"][1] = row["source_structure_links"][1].replace(
        "->intent:",
        "->intent:false_",
    )
    row["verification_hash"] = _verification_hash(row)

    with pytest.raises(
        ValueError,
        match="source graph (mismatch|references an unknown card)",
    ):
        audit_scenarios(rows, registry)


def test_audit_rejects_semantically_incompatible_outcome() -> None:
    registry = load_scenario_registry(REGISTRY)
    rows = compile_scenarios(registry, target_scenarios=EXPECTED_SCENARIOS)
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
    old_outcome_key = next(
        key for key in row["source_structure_keys"] if key.startswith("outcome:")
    )
    row["source_structure_keys"] = [
        "outcome:ordered_steps" if key == old_outcome_key else key
        for key in row["source_structure_keys"]
    ]
    row["source_structure_links"] = [
        link.replace(f"->{old_outcome_key}", "->outcome:ordered_steps")
        for link in row["source_structure_links"]
    ]
    row["verification_hash"] = _verification_hash(row)

    with pytest.raises(ValueError, match="compatibility violations"):
        audit_scenarios(rows, registry)


def test_domain_intent_and_state_outcome_rules_are_realized() -> None:
    registry = load_scenario_registry(REGISTRY)
    rows = compile_scenarios(registry, target_scenarios=EXPECTED_SCENARIOS)

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
    rows = compile_scenarios(registry, target_scenarios=EXPECTED_SCENARIOS)

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
    actual_validation = sum(row["split"] == "validation" for row in rows)
    expected_validation = round(len(rows) * registry.validation_percent / 100)
    assert abs(actual_validation - expected_validation) <= len(registry.families)


def test_safety_constraints_are_domain_specific() -> None:
    registry = load_scenario_registry(REGISTRY)
    rows = compile_scenarios(registry, target_scenarios=EXPECTED_SCENARIOS)
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
    rows = compile_scenarios(registry, target_scenarios=EXPECTED_SCENARIOS)

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
    assert set(first) == {f"frame_{index:02d}" for index in range(1, 25)}
    assert {first.count(frame) for frame in set(first)} == {1}
