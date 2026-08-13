from __future__ import annotations

import pytest

from complexity_card_corpus.v2 import render_complete_v2, v2_generation_progress
from complexity_card_corpus.v2 import audit_v2_splits


pytestmark = pytest.mark.slow


def test_v2_progress_reports_full_registered_capacity_without_a_cap() -> None:
    progress = v2_generation_progress()

    assert progress["example_limit"] is None
    assert progress["registered_capacity"] == 206_090
    assert len(progress["registered_families"]) == 15
    assert progress["missing_families"] == ()


def test_v2_full_build_renders_every_registered_example() -> None:
    rows = render_complete_v2()

    assert len(rows) == 206_090
    assert len({row["example_id"] for row in rows}) == len(rows)
    assert set(row["split"] for row in rows) == {"train", "validation", "test"}
    assert audit_v2_splits(rows)["passed"] is True
