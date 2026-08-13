from __future__ import annotations

import re

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
    assert len(rows) == casual_conversation_capacity() == 49_794
    assert len({row["example_id"] for row in rows}) == len(rows)
    assert all(len(row["messages"]) in {2, 4} for row in rows)
    assert sum(len(row["messages"]) == 4 for row in rows) == 10_000


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
    assert audit["casual_conversation"]["natural_social_rows"] >= 5_000
    assert audit["missing_anchors"] == []
    assert audit["incorrect_anchors"] == []
    assert audit["passed"] is True


def test_v2_natural_social_surfaces_preserve_speaker_grammar(
    rows: list[dict[str, object]],
) -> None:
    social = [
        row for row in rows if str(row["domain"]).startswith("social_")
    ]
    assert social
    assert all("my notes" not in str(row["final_response"]) for row in social)
    assert all(
        "the rest of my work" not in str(row["final_response"])
        for row in social
    )
    assert all(
        "so I can verify" not in str(row["final_response"])
        for row in social
    )
    assert all(
        "a paragraph I am revising" not in str(row["final_response"])
        for row in social
    )
    assert all(
        re.search(r"(?<=[.!?])\s+[a-z]", str(row["prompt"])) is None
        for row in social
    )
