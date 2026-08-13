from __future__ import annotations

from complexity_card_corpus.v2 import audit_v2_family_roadmap
from complexity_card_corpus.v2.families import grounded_qa_capacity, render_grounded_qa_rows


def test_v2_grounded_qa_uses_full_capacity_and_passes() -> None:
    rows = render_grounded_qa_rows()
    family = audit_v2_family_roadmap(rows)["families"]["grounded_qa"]

    assert len(rows) == grounded_qa_capacity() == 4_608
    assert family["priority"] == "PASS"
