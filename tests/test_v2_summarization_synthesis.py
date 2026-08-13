from __future__ import annotations

import re

from complexity_card_corpus.v2 import audit_v2_family_roadmap
from complexity_card_corpus.v2.families import render_summarization_synthesis_rows, summarization_synthesis_capacity


def test_v2_summarization_uses_full_capacity_and_passes() -> None:
    rows = render_summarization_synthesis_rows()
    family = audit_v2_family_roadmap(rows)["families"]["summarization_synthesis"]

    assert len(rows) == summarization_synthesis_capacity() == 64
    assert all(len(re.findall(r"[.!?](?:\s|$)", row["final_response"])) == 2 for row in rows)
    assert family["priority"] == "PASS"
