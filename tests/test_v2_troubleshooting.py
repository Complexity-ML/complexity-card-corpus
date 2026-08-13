from __future__ import annotations

import json

from complexity_card_corpus.v2 import audit_v2_family_roadmap
from complexity_card_corpus.v2.families import render_troubleshooting_rows, troubleshooting_capacity


def test_v2_troubleshooting_uses_full_capacity_and_passes() -> None:
    rows = render_troubleshooting_rows()
    family = audit_v2_family_roadmap(rows)["families"]["troubleshooting"]

    assert len(rows) == troubleshooting_capacity() == 384
    assert family["priority"] == "PASS"
    assert all(
        "for the reported" not in str(row["final_response"])
        for row in rows
    )
    for row in rows:
        source = json.loads(str(row["source_representation"]))
        site = source["facts"]["site"]
        assert str(row["final_response"]).count(site) <= 1
