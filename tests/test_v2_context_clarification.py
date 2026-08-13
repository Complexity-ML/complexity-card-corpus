from __future__ import annotations

import pytest

from complexity_card_corpus.v2 import audit_v2_family_roadmap
from complexity_card_corpus.v2.families import (
    context_clarification_capacity,
    render_context_clarification_rows,
)


@pytest.fixture(scope="module")
def rows() -> list[dict[str, object]]:
    return render_context_clarification_rows()


def test_v2_context_clarification_renders_every_scenario(
    rows: list[dict[str, object]],
) -> None:
    assert len(rows) == context_clarification_capacity() == 9_216
    assert len({row["example_id"] for row in rows}) == len(rows)


def test_v2_context_clarification_passes_every_family_gate(
    rows: list[dict[str, object]],
) -> None:
    family = audit_v2_family_roadmap(rows)["families"]["context_clarification"]

    assert family["priority"] == "PASS"
    assert family["behavior"]["prompt_copy_share"] <= 0.10
    assert family["near_duplicates"]["prompt"]["collision_share"] <= 0.10
    assert family["near_duplicates"]["final"]["collision_share"] <= 0.10
