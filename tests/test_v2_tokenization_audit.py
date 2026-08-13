from __future__ import annotations

from pathlib import Path

import pytest

from complexity_card_corpus.v2 import audit_v2_tokenization, render_think_final


def test_v2_tokenization_contract_with_project_tokenizer() -> None:
    tokenizer = Path("/Users/boris/Dev/complexity-framework/tokenizer")
    if not tokenizer.exists():
        pytest.skip("the framework tokenizer is not available")
    response = render_think_final(
        "Add the two values and verify the result independently.",
        "The answer is 4.",
    )
    rows = [
        {
            "example_id": "arithmetic",
            "task": "reasoning_verification",
            "split": "train",
            "prompt": "What is 2 + 2?",
            "response": response,
            "final_response": "The answer is 4.",
        },
        {
            "example_id": "hello",
            "task": "casual_conversation",
            "split": "train",
            "prompt": "Hello",
            "response": "Hello! How can I help?",
            "final_response": "Hello! How can I help?",
        },
    ]

    audit = audit_v2_tokenization(rows, tokenizer)

    assert audit["passed"] is True
    assert audit["roundtrip_failures"] == 0
    assert audit["loss_mask_failures"] == 0
    assert audit["envelope_failures"] == 0
    assert audit["marker_failures"] == []
