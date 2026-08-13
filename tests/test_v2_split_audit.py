from __future__ import annotations

from complexity_card_corpus.v2 import audit_v2_splits
from complexity_card_corpus.v2.registry import _ensure_family_heldouts


def _row(example_id: str, split: str, prompt: str, final: str) -> dict:
    return {
        "example_id": example_id,
        "split": split,
        "prompt": prompt,
        "final_response": final,
    }


def test_v2_split_audit_detects_exact_and_numeric_template_leakage() -> None:
    rows = [
        _row("train-exact", "train", "Add two and three.", "The answer is five."),
        _row("eval-exact", "validation", "Add two and three.", "The answer is five."),
        _row("train-template", "train", "Add 17 and 8.", "The answer is 25."),
        _row("test-template", "test", "Add 91 and 4.", "The answer is 95."),
    ]

    audit = audit_v2_splits(rows)

    assert audit["passed"] is False
    assert audit["exact_cross_split_collision_rows"] == 2
    assert audit["structural_cross_split_collision_rows"] == 4


def test_v2_split_audit_accepts_functionally_distinct_splits() -> None:
    rows = [
        _row("train", "train", "Name a blue mineral.", "Azurite is blue."),
        _row("eval", "validation", "Why do leaves wilt?", "They lose water pressure."),
    ]

    assert audit_v2_splits(rows)["passed"] is True


def test_v2_split_audit_refuses_to_certify_one_split() -> None:
    audit = audit_v2_splits(
        [_row("train", "train", "Say hello.", "Hello!")]
    )

    assert audit["passed"] is False
    assert "at least two populated splits are required" in audit["violations"]


def test_v2_split_fallback_moves_whole_composition_groups() -> None:
    rows = []
    for index in range(6):
        row = _row(
            f"row-{index}",
            "train",
            f"Prompt {index}.",
            f"Answer {index}.",
        )
        row["task"] = "summarization_synthesis"
        row["source_representation"] = __import__("json").dumps(
            {
                "case_id": f"case-{index}",
                "composition": {
                    "intent": "summarize",
                    "domain": "operations",
                    "deck_name": f"deck-{index // 2}",
                    "prompt_plan": "prompt-0",
                    "answer_plan": "answer-0",
                    "thinking_plan": "thinking-none",
                    "user_tone": "neutral",
                },
            }
        )
        rows.append(row)

    _ensure_family_heldouts(rows)

    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    for deck_index in range(3):
        assert len({rows[deck_index * 2 + offset]["split"] for offset in range(2)}) == 1
    assert audit_v2_splits(rows)["composition_cross_split_collision_rows"] == 0
