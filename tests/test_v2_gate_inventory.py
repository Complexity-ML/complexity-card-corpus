from __future__ import annotations

from complexity_card_corpus.v2 import V2_RELEASE_GATES, v2_gate_progress


def test_v2_gate_inventory_has_unique_stable_ids() -> None:
    gate_ids = [gate.gate_id for gate in V2_RELEASE_GATES]

    assert len(gate_ids) == len(set(gate_ids))
    assert len(gate_ids) >= 40


def test_v2_gate_inventory_is_fully_implemented() -> None:
    progress = v2_gate_progress()

    assert progress["complete"] is True
    assert progress["implemented_count"] == 49
    assert "surface.prompt_thinking_copy" in progress["implemented"]
    assert "correctness.arithmetic_recomputation" in progress["implemented"]
    assert "composition.compatibility_graph_validity" in progress["implemented"]
    assert "composition.compatibility_graph_coverage" in progress["implemented"]
    assert "composition.contextual_history_dependency" in progress["implemented"]
    assert "tokenization.assistant_loss_mask" in progress["implemented"]
    assert progress["missing"] == ()
