from __future__ import annotations

import os
from pathlib import Path

import pytest

from complexity_card_corpus.v2 import audit_projected_parquet


pytestmark = pytest.mark.slow


def test_projected_release_passes_v2_learned_behavior_gate() -> None:
    """Opt-in release gate for an explicitly selected projected artifact."""

    selected = os.environ.get("CARD_CORPUS_V2_PROJECTED")
    if not selected:
        pytest.skip("set CARD_CORPUS_V2_PROJECTED to audit a release candidate")
    projected = Path(selected)
    if not projected.exists():
        pytest.fail(f"selected projected V2 artifact does not exist: {projected}")
    audit = audit_projected_parquet(projected)

    assert audit["passed"] is True, {
        "violations": audit["violations"],
        "casual_conversation": audit["casual_conversation"],
        "missing_anchors": audit["missing_anchors"],
        "incorrect_anchors": audit["incorrect_anchors"],
        "failing_tasks": audit["failing_tasks"],
    }
