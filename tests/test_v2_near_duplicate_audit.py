from __future__ import annotations

from complexity_card_corpus.v2 import audit_v2_near_duplicates


def _row(index: int, prompt: str, final: str) -> dict:
    return {
        "example_id": str(index),
        "task": "reasoning_verification",
        "split": "train",
        "prompt": prompt,
        "final_response": final,
    }


def test_v2_near_duplicate_audit_catches_numeric_template_recycling() -> None:
    rows = [
        _row(
            index,
            f"Calculate the sum of {index} and {index + 3} and return the number.",
            f"The result of the calculation is {index * 2 + 3}.",
        )
        for index in range(40)
    ]

    audit = audit_v2_near_duplicates(rows)
    task = audit["tasks"]["reasoning_verification"]

    assert task["prompt"]["collision_share"] == 1.0
    assert task["final"]["collision_share"] == 1.0
    assert task["failures"] == ["prompt_near_duplicates", "final_near_duplicates"]


def test_v2_near_duplicate_audit_accepts_distinct_short_functions() -> None:
    words = (
        "amber", "birch", "cobalt", "delta", "ember", "frost", "grove", "harbor",
        "iris", "jade", "kelp", "linen", "moss", "north", "opal", "pearl",
        "quartz", "reed", "slate", "tulip",
    )
    rows = [
        _row(index, f"Return {word}.", word)
        for index, word in enumerate(words)
    ]

    audit = audit_v2_near_duplicates(rows)

    assert audit["passed"] is True
