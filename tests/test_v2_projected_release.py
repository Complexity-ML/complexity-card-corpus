from __future__ import annotations

from pathlib import Path

import pytest

from complexity_card_corpus.v2 import audit_projected_parquet


PROJECTED_V2 = Path("build/card-corpus-v2/projected.parquet")


def test_projected_release_passes_v2_learned_behavior_gate() -> None:
    """Red release gate: it stays failing until the V2 corpus is repaired."""

    if not PROJECTED_V2.exists():
        pytest.skip("build the projected V2 candidate before running its release gate")
    audit = audit_projected_parquet(PROJECTED_V2)

    assert audit["passed"] is True, {
        "violations": audit["violations"],
        "casual_conversation": audit["casual_conversation"],
        "missing_anchors": audit["missing_anchors"],
        "incorrect_anchors": audit["incorrect_anchors"],
        "failing_tasks": audit["failing_tasks"],
    }
