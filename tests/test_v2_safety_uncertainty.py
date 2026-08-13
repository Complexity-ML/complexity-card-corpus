from __future__ import annotations

from complexity_card_corpus.v2 import audit_v2_family_roadmap
from complexity_card_corpus.v2.families import render_safety_uncertainty_rows, safety_uncertainty_capacity


def test_v2_safety_uses_full_capacity_and_passes() -> None:
    rows = render_safety_uncertainty_rows()
    family = audit_v2_family_roadmap(rows)["families"]["safety_uncertainty"]

    assert len(rows) == safety_uncertainty_capacity() == 384
    assert family["priority"] == "PASS"
