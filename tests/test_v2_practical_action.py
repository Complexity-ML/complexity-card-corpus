from __future__ import annotations

from complexity_card_corpus.v2 import audit_v2_family_roadmap
from complexity_card_corpus.v2.families import practical_action_capacity, render_practical_action_rows


def test_v2_practical_action_uses_full_capacity_and_passes() -> None:
    rows = render_practical_action_rows()
    family = audit_v2_family_roadmap(rows)["families"]["practical_action"]

    assert len(rows) == practical_action_capacity() == 384
    assert family["priority"] == "PASS"
