from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from complexity_card_corpus.quality_audit import resolve_quality_audit_policy
from complexity_card_corpus.posttrain.rendering import (
    _balance_conversation_families,
)
from complexity_card_corpus.scenarios import (
    compile_scenarios,
    load_scenario_registry,
    resolve_family_targets,
)
from complexity_card_corpus.sft.selection import (
    _balance_response_card_hands,
    _balance_task_domains,
    _balance_task_families,
    _deduplicate_structural_rows,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/scenario-forge/scenario-forge-v1.json"


def _rows(count: int) -> list[dict[str, object]]:
    return [
        {
            "example_id": f"example:{index:06d}",
            "task": "grounded_qa",
            "domain": "dominant" if index < count - 100 else "minority",
            "_projected_target": "The same normalized response shape.",
            "_conditioning_cards": SimpleNamespace(
                response_structure_signature=(
                    "dominant_hand" if index < count - 100 else "minority_hand"
                )
            ),
        }
        for index in range(count)
    ]


def test_default_sft_selection_does_not_drop_a_family_above_fifteen_thousand() -> None:
    rows = _rows(15_001)

    family_rows, family_audit = _balance_task_families(
        rows, max_examples_per_family=None
    )
    structure_rows, structure_audit = _deduplicate_structural_rows(rows)
    domain_rows, domain_audit = _balance_task_domains(rows, maximum_share=None)
    hand_rows, hand_audit = _balance_response_card_hands(rows, maximum_share=None)

    assert len(family_rows) == len(rows)
    assert len(structure_rows) == len(rows)
    assert len(domain_rows) == len(rows)
    assert len(hand_rows) == len(rows)
    assert family_audit["maximum_examples_per_family"] is None
    assert structure_audit["maximum_retained_per_structure"] is None
    assert domain_audit["requested_maximum_share"] is None
    assert hand_audit["requested_maximum_share"] is None


def test_default_post_training_selection_does_not_drop_above_old_cap() -> None:
    rows = [
        {
            "example_id": f"post:{index:06d}",
            "task": "grounded_qa",
            "answer_json": '{"lexical_focus": null}',
        }
        for index in range(40_001)
    ]

    kept, audit = _balance_conversation_families(rows)

    assert len(kept) == len(rows)
    assert audit["maximum_examples_per_family"] is None
    assert audit["policy"] == "preserve_all_valid_rows"
    assert audit["dropped"] == 0


def test_manual_recovery_caps_remain_explicit_opt_in_controls() -> None:
    rows = _rows(200)

    family_rows, _ = _balance_task_families(rows, max_examples_per_family=40)
    structure_rows, _ = _deduplicate_structural_rows(rows, max_per_structure=16)

    assert len(family_rows) == 40
    # Structural units include semantic domain, so both domain buckets retain 16.
    assert len(structure_rows) == 32


def test_statistical_resources_grow_but_every_row_remains_scored() -> None:
    current = resolve_quality_audit_policy(443_782)
    future = resolve_quality_audit_policy(1_000_000)

    assert future["sample_size"] > current["sample_size"]
    assert future["max_features"] >= current["max_features"]
    assert future["cluster_count"] >= current["cluster_count"]
    assert current["score_batch_size"] <= 10_000
    assert future["score_batch_size"] <= 10_000


def test_scenario_targets_scale_from_weights_instead_of_family_caps() -> None:
    registry = load_scenario_registry(REGISTRY)
    baseline = {
        family.family_id: family.weight for family in registry.families
    }
    scaled = resolve_family_targets(registry, target_scenarios=80_000)
    capacities = {
        family.family_id: family.semantic_signature_capacity()
        for family in registry.families
    }

    assert sum(baseline.values()) == 65_000
    assert sum(scaled.values()) == 80_000
    assert any(scaled[key] > baseline[key] for key in baseline)
    assert all(scaled[key] <= capacities[key] for key in scaled)


def test_scenario_compilation_requires_an_explicit_total() -> None:
    registry = load_scenario_registry(REGISTRY)

    try:
        compile_scenarios(registry)  # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("scenario compilation accepted an implicit target")
