from __future__ import annotations

from complexity_card_corpus.v2 import audit_v2_family_roadmap
from complexity_card_corpus.v2.families import (
    extraction_classification_capacity,
    render_extraction_classification_rows,
)


def test_v2_extraction_uses_full_capacity_and_passes() -> None:
    rows = render_extraction_classification_rows()
    family = audit_v2_family_roadmap(rows)["families"]["extraction_classification"]

    assert len(rows) == extraction_classification_capacity() == 3_456
    assert family["priority"] == "PASS"
    assert family["integrity_counts"]["validator_error_count"] == 0
