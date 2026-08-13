from __future__ import annotations

import json

from complexity_card_corpus.v2 import (
    audit_v2_family_roadmap,
    render_think_final,
    roadmap_markdown,
)


def _row(example_id: str, task: str, prompt: str, final: str) -> dict:
    thinking = "Inspect the values independently. A separate check confirms the result."
    response = render_think_final(thinking, final)
    return {
        "example_id": example_id,
        "task": task,
        "domain": "general",
        "split": "train",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "prompt": prompt,
        "response": response,
        "reasoning_trace": thinking,
        "final_response": final,
        "source": "authored test",
        "license": "CC BY-NC 4.0",
        "version": "2.0.0",
        "source_representation": json.dumps(
            {
                "deck_name": f"{task}:test",
                "composition": {
                    "intent": task,
                    "domain": "general",
                    "deck_name": f"{task}:test:{example_id}",
                    "prompt_plan": "prompt-default",
                    "answer_plan": "answer-default",
                    "thinking_plan": "thinking-default",
                    "prompt_functions": ["request"],
                    "answer_functions": ["answer"],
                    "thinking_functions": ["verify"],
                    "user_tone": "neutral",
                    "thinking_budget": "short",
                },
                "variable_indices": {
                    "prompt": {"request": 0},
                    "answer": {"direct": 0},
                },
                "variable_card_counts": {
                    "prompt": {"request": 1},
                    "answer": {"direct": 1},
                },
                "dependency_graph": {
                    "prompt[request]": [],
                    "answer[direct]": [],
                },
                "validator": {"kind": "natural", "maximum_words": 100},
            }
        ),
    }


def test_v2_roadmap_audits_each_family_independently() -> None:
    rows = [
        _row("one", "writing_transformation", "Rewrite this.", "A clear rewrite."),
        _row(
            "two",
            "troubleshooting",
            "Help diagnose this.",
            "Repeat this step. Repeat this step. Repeat this step.",
        ),
    ]

    roadmap = audit_v2_family_roadmap(rows)

    assert set(roadmap["families"]) == {
        "troubleshooting",
        "writing_transformation",
    }
    assert "internal_repetition" in roadmap["families"]["troubleshooting"][
        "behavior_failures"
    ]
    assert roadmap["rows"] == 2
    assert roadmap["train_rows"] == 2
    assert "near_duplicates" in roadmap["families"]["troubleshooting"]
    assert "lengths" in roadmap["families"]["troubleshooting"]
    assert "splits" in roadmap["families"]["troubleshooting"]


def test_v2_roadmap_reports_pass_after_all_family_gates_pass() -> None:
    rows = []
    for index in range(30):
        label = chr(ord("a") + index // 26) + chr(ord("a") + index % 26)
        row = _row(
            label,
            "writing_transformation",
            f"Transform item {label}.",
            f"value{label}",
        )
        row["reasoning_trace"] = ""
        row["response"] = row["final_response"]
        row["messages"][-1]["content"] = row["final_response"]
        row["domain"] = ("email", "notice", "summary")[index % 3]
        if index % 10 == 0:
            row["split"] = "validation"
        rows.append(row)

    roadmap = audit_v2_family_roadmap(rows, require_splits=True)

    assert roadmap["families"]["writing_transformation"]["priority"] == "PASS"
    assert roadmap["priority_counts"]["PASS"] == 1
    assert roadmap["split_audit"]["passed"] is True
    assert roadmap["families"]["writing_transformation"]["split_passed"] is True


def test_v2_roadmap_renders_a_compact_priority_table() -> None:
    roadmap = audit_v2_family_roadmap(
        [_row("one", "writing_transformation", "Rewrite this.", "A clear rewrite.")]
    )

    rendered = roadmap_markdown(roadmap)

    assert "| Priority | Family | Rows | Behavior |" in rendered
    assert "Near duplicates | Length | Splits |" in rendered
    assert "Tokenizer |" in rendered
    assert "writing_transformation" in rendered
