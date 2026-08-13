from __future__ import annotations

from complexity_card_corpus.v2 import audit_v2_family_roadmap
from complexity_card_corpus.v2.families import brainstorming_creativity_capacity, render_brainstorming_creativity_rows


def test_v2_brainstorming_uses_full_capacity_and_passes() -> None:
    rows = render_brainstorming_creativity_rows()
    family = audit_v2_family_roadmap(rows)["families"]["brainstorming_creativity"]

    assert len(rows) == brainstorming_creativity_capacity() == 384
    assert family["priority"] == "PASS"
