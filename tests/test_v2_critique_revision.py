from __future__ import annotations

import re

from complexity_card_corpus.v2 import audit_v2_family_roadmap
from complexity_card_corpus.v2.families import critique_revision_capacity, render_critique_revision_rows


def test_v2_critique_uses_full_capacity_and_exactly_two_sentences() -> None:
    rows = render_critique_revision_rows()
    family = audit_v2_family_roadmap(rows)["families"]["critique_revision"]

    assert len(rows) == critique_revision_capacity() == 64
    assert all(len(re.findall(r"[.!?](?:\s|$)", row["final_response"])) == 2 for row in rows)
    assert family["priority"] == "PASS"
