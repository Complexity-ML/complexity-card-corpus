from __future__ import annotations

from complexity_card_corpus.v2 import audit_v2_family_roadmap
from complexity_card_corpus.v2.families import conversation_empathy_capacity, render_conversation_empathy_rows


def test_v2_empathy_uses_full_capacity_and_passes() -> None:
    rows = render_conversation_empathy_rows()
    family = audit_v2_family_roadmap(rows)["families"]["conversation_empathy"]

    assert len(rows) == conversation_empathy_capacity() == 2_048
    assert family["priority"] == "PASS"
