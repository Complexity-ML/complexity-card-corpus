from __future__ import annotations

from complexity_card_corpus.v2 import audit_v2_lengths


def _row(index: int, words: int, *, thinking_words: int = 0) -> dict:
    return {
        "task": f"task-{index % 3}",
        "split": "train",
        "final_response": " ".join(f"w{index}" for _ in range(words)),
        "reasoning_trace": " ".join("reason" for _ in range(thinking_words)),
    }


def test_v2_length_audit_requires_four_model_facing_bands() -> None:
    rows = []
    for index, words in enumerate((10, 50, 120, 250)):
        rows.extend(_row(index * 10 + offset, words) for offset in range(10))

    audit = audit_v2_lengths(rows)

    assert audit["passed"] is True
    assert all(share >= 0.05 for share in audit["final_band_shares"].values())


def test_v2_length_audit_rejects_repetitive_runaway_thinking_length() -> None:
    audit = audit_v2_lengths([_row(1, 10, thinking_words=150)], require_global_bands=False)

    assert audit["thinking_outside_contract"] == 1
    assert audit["passed"] is False
