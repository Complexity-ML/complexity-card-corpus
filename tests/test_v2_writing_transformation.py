from __future__ import annotations

from complexity_card_corpus.v2 import audit_v2_family_roadmap
from complexity_card_corpus.v2.families import render_writing_transformation_rows, writing_transformation_capacity


def test_v2_writing_uses_full_capacity_and_passes() -> None:
    rows = render_writing_transformation_rows()
    family = audit_v2_family_roadmap(rows)["families"]["writing_transformation"]

    assert len(rows) == writing_transformation_capacity() == 4_096
    assert family["priority"] == "PASS"
