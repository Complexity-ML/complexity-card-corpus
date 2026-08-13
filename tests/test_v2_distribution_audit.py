from __future__ import annotations

import json

from complexity_card_corpus.v2 import audit_v2_distribution


def _row(index: int, *, biased: bool = False) -> dict:
    action = 0 if biased else index % 2
    approach = (index // 2) % 2
    metadata = {
        "deck_name": "reasoning:test",
        "variable_indices": {
            "prompt": {"request": index % 2},
            "thinking": {"action": action, "approach": approach},
            "answer": {"direct": index % 2},
        },
        "variable_card_counts": {
            "prompt": {"request": 2},
            "thinking": {"action": 2, "approach": 2},
            "answer": {"direct": 2},
        },
        "dependency_graph": {
            "prompt[request]": [],
            "thinking[action]": [],
            "thinking[approach]": ["thinking[action]"],
            "answer[direct]": [],
        },
    }
    return {
        "task": "reasoning_verification",
        "split": "train",
        "domain": ("math", "logic", "probability")[index % 3],
        "source_representation": json.dumps(metadata),
    }


def test_v2_distribution_accepts_balanced_cards_edges_and_domains() -> None:
    audit = audit_v2_distribution([_row(index) for index in range(40)])

    task = audit["tasks"]["reasoning_verification"]
    assert audit["passed"] is True
    assert task["reservoir_failure_count"] == 0
    assert task["edge_failure_count"] == 0


def test_v2_distribution_rejects_unused_variable_cards() -> None:
    audit = audit_v2_distribution([_row(index, biased=True) for index in range(40)])

    assert "variable_card_entropy" in audit["tasks"]["reasoning_verification"][
        "failures"
    ]


def test_v2_distribution_counts_heldout_cards_in_authoring_coverage() -> None:
    rows = []
    for index in range(20):
        train = _row(index, biased=True)
        rows.append(train)
        heldout = _row(index + 20, biased=True)
        heldout["split"] = "test"
        metadata = json.loads(heldout["source_representation"])
        metadata["variable_indices"]["thinking"]["action"] = 1
        heldout["source_representation"] = json.dumps(metadata)
        rows.append(heldout)

    audit = audit_v2_distribution(rows)

    assert audit["passed"] is True
    assert audit["tasks"]["reasoning_verification"]["train_rows"] == 20


def test_v2_distribution_requires_machine_readable_deck_provenance() -> None:
    audit = audit_v2_distribution(
        [{"task": "writing_transformation", "split": "train", "domain": "email"}]
    )

    assert "variable_provenance_unavailable" in audit["tasks"][
        "writing_transformation"
    ]["failures"]
