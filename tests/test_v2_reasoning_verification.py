from __future__ import annotations

import json

import pytest

from complexity_card_corpus.v2.behavior_audit import audit_v2_behavior
from complexity_card_corpus.v2.integrity_audit import audit_v2_integrity
from complexity_card_corpus.v2.families import (
    reasoning_verification_capacity,
    render_reasoning_verification_rows,
)


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def reasoning_rows() -> list[dict[str, object]]:
    return render_reasoning_verification_rows()


def test_v2_reasoning_has_enough_original_scenario_capacity() -> None:
    assert reasoning_verification_capacity() == 108_000


def test_v2_reasoning_full_family_uses_its_complete_capacity(
    reasoning_rows: list[dict[str, object]],
) -> None:
    first = reasoning_rows

    assert len(first) == reasoning_verification_capacity()
    assert len({row["example_id"] for row in first}) == len(first)
    assert all(len(row["messages"]) == 2 for row in first)


def test_v2_reasoning_full_family_passes_behavior_metrics(
    reasoning_rows: list[dict[str, object]],
) -> None:
    rows = reasoning_rows
    audit = audit_v2_behavior(
        rows,
        thresholds={
            "required_train_examples": len(rows),
            "minimum_direct_casual_examples": 0,
            "minimum_direct_casual_share": 0.0,
            "minimum_short_direct_casual_share": 0.0,
        },
    )
    task = audit["tasks"]["reasoning_verification"]

    assert task["internal_repetition_share"] <= 0.02
    assert task["prompt_copy_share"] <= 0.03
    assert task["abstract_function_share"] <= 0.03
    assert task["top_closing_sentence_share"] <= 0.05
    assert task["thinking_share"] == 1.0
    assert task["thinking_internal_repetition_share"] == 0.0
    assert task["top_exact_thinking_signature_share"] <= 0.01
    assert task["top_thinking_fivegram_share"] <= 0.10
    assert task["thinking_prompt_copy_share"] == 0.0
    assert task["thinking_final_overlap_share"] == 0.0
    assert audit["failing_tasks"].get("reasoning_verification", []) == []

    integrity = audit_v2_integrity(rows)
    assert integrity["passed"] is True
    assert integrity["arithmetic_error_count"] == 0
    assert integrity["rendering_error_counts"] == {}
    for row in rows:
        facts = json.loads(row["source_representation"])["facts"]
        if facts["operation"] == "division":
            assert f"{facts['items']} per station" in row["final_response"]
