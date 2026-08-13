from __future__ import annotations

import pytest

from complexity_card_corpus.v2 import audit_v2_family_roadmap
from complexity_card_corpus.v2.families import explanation_learning_capacity, render_explanation_learning_rows


pytestmark = pytest.mark.slow


def test_v2_explanation_uses_full_capacity_and_passes() -> None:
    rows = render_explanation_learning_rows()
    family = audit_v2_family_roadmap(rows)["families"]["explanation_learning"]

    assert len(rows) == explanation_learning_capacity() == 32_776
    assert family["priority"] == "PASS"
    assert family["lengths"]["final_band_shares"]["concise_26_80"] >= 0.25
    assert family["lengths"]["final_band_shares"]["detailed_81_200"] >= 0.25
    assert family["lengths"]["final_band_shares"]["extended_201_512"] >= 0.25

    finals = [str(row["final_response"]) for row in rows]
    assert all("teach_back" not in final for final in finals)
    assert all("to removing" not in final for final in finals)
    assert all("learner removing" not in final for final in finals)
    assert all("kitchen table Keep" not in final for final in finals)
