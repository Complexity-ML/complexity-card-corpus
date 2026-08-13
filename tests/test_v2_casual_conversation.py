from __future__ import annotations

import pytest

from complexity_card_corpus.v2 import audit_v2_behavior, audit_v2_family_roadmap
from complexity_card_corpus.v2.families import (
    casual_conversation_capacity,
    render_casual_conversation_rows,
)


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def rows() -> list[dict[str, object]]:
    return render_casual_conversation_rows()


def test_v2_casual_renders_every_valid_direct_case(
    rows: list[dict[str, object]],
) -> None:
    assert len(rows) == casual_conversation_capacity() == 30_294
    assert len({row["example_id"] for row in rows}) == len(rows)
    assert all(len(row["messages"]) == 2 for row in rows)


def test_v2_casual_passes_every_family_gate(
    rows: list[dict[str, object]],
) -> None:
    family = audit_v2_family_roadmap(rows)["families"]["casual_conversation"]

    assert family["priority"] == "PASS"
    assert family["behavior_failures"] == []
    assert family["integrity_violations"] == []
    assert family["distribution_failures"] == []
    assert family["near_duplicate_failures"] == []
    assert family["near_duplicates"]["prompt"]["collision_share"] <= 0.10
    assert family["near_duplicates"]["final"]["collision_share"] <= 0.10


def test_v2_casual_satisfies_global_direct_and_anchor_contract(
    rows: list[dict[str, object]],
) -> None:
    audit = audit_v2_behavior(rows)

    assert audit["casual_conversation"]["direct_rows"] >= 25_000
    assert audit["casual_conversation"]["short_direct_share"] >= 0.90
    assert audit["missing_anchors"] == []
    assert audit["incorrect_anchors"] == []
    assert audit["passed"] is True
