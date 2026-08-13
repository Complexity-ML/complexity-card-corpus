from __future__ import annotations

from complexity_card_corpus.v2 import audit_v2_family_roadmap
from complexity_card_corpus.v2.families import render_troubleshooting_rows, troubleshooting_capacity


def test_v2_troubleshooting_uses_full_capacity_and_passes() -> None:
    rows = render_troubleshooting_rows()
    family = audit_v2_family_roadmap(rows)["families"]["troubleshooting"]

    assert len(rows) == troubleshooting_capacity() == 384
    assert family["priority"] == "PASS"
