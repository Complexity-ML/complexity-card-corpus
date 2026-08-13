from __future__ import annotations

from complexity_card_corpus.v2 import audit_v2_family_roadmap
from complexity_card_corpus.v2.families import planning_comparison_capacity, render_planning_comparison_rows


def test_v2_planning_uses_full_capacity_and_passes() -> None:
    rows = render_planning_comparison_rows()
    family = audit_v2_family_roadmap(rows)["families"]["planning_comparison"]

    assert len(rows) == planning_comparison_capacity() == 384
    assert family["priority"] == "PASS"
