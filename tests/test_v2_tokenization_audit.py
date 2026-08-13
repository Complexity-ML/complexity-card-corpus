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
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": response},
            ],
            "prompt": "What is 2 + 2?",
            "response": response,
            "final_response": "The answer is 4.",
        },
        {
            "example_id": "hello",
            "task": "casual_conversation",
            "split": "train",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello! How can I help?"},
            ],
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


def test_v2_tokenization_masks_prior_assistant_history() -> None:
    tokenizer = Path("/Users/boris/Dev/complexity-framework/tokenizer")
    if not tokenizer.exists():
        pytest.skip("the framework tokenizer is not available")
    rows = [
        {
            "example_id": "followup",
            "task": "casual_conversation",
            "split": "train",
            "messages": [
                {"role": "user", "content": "Remember five boxes."},
                {"role": "assistant", "content": "I will remember five boxes."},
                {"role": "user", "content": "Add two more."},
                {"role": "assistant", "content": "There are seven boxes now."},
            ],
            "prompt": "Add two more.",
            "response": "There are seven boxes now.",
            "final_response": "There are seven boxes now.",
        }
    ]

    audit = audit_v2_tokenization(rows, tokenizer)

    assert audit["passed"] is True
    assert audit["conversation_failures"] == 0
