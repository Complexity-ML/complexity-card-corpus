from __future__ import annotations

from complexity_card_corpus.v2 import audit_v2_behavior


def _row(example_id: str, prompt: str, response: str, *, task: str = "casual_conversation") -> dict:
    return {
        "example_id": example_id,
        "task": task,
        "split": "train",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }


def _permissive_thresholds() -> dict[str, float | int]:
    return {
        "minimum_direct_casual_examples": 0,
        "required_train_examples": 1,
        "minimum_direct_casual_share": 0.0,
        "minimum_short_direct_casual_share": 0.0,
        "maximum_internal_repetition_share": 1.0,
        "maximum_prompt_copy_share": 1.0,
        "maximum_task_internal_repetition_share": 1.0,
        "maximum_task_prompt_copy_share": 1.0,
        "maximum_task_abstract_function_share": 1.0,
        "maximum_task_closing_sentence_share": 1.0,
        "maximum_task_exact_response_share": 1.0,
        "minimum_reasoning_thinking_share": 0.0,
        "maximum_thinking_internal_repetition_share": 1.0,
        "maximum_task_thinking_internal_repetition_share": 1.0,
        "maximum_task_exact_thinking_signature_share": 1.0,
        "maximum_task_thinking_fivegram_share": 1.0,
        "maximum_task_thinking_final_overlap_share": 1.0,
        "maximum_thinking_prompt_copy_share": 1.0,
        "maximum_task_thinking_prompt_copy_share": 1.0,
        "maximum_non_single_assistant_row_share": 1.0,
    }


def test_v2_audit_detects_repetition_inside_one_response() -> None:
    rows = [
        _row(
            "repeat",
            "What is two plus two?",
            "Two plus two is four. Two plus two is four. Two plus two is four.",
        )
    ]

    audit = audit_v2_behavior(rows, thresholds=_permissive_thresholds())

    assert audit["tasks"]["casual_conversation"]["internal_repetition_count"] == 1


def test_v2_audit_detects_prompt_copy_hidden_by_unique_rows() -> None:
    prompt = "Explain why a separate validation set gives a more honest model estimate."
    rows = [
        _row(
            "copy",
            prompt,
            "A separate validation set gives a more honest model estimate for this experiment.",
        )
    ]

    audit = audit_v2_behavior(rows, thresholds=_permissive_thresholds())

    assert audit["tasks"]["casual_conversation"]["prompt_copy_count"] == 1


def test_v2_audit_counts_every_supervised_assistant_turn() -> None:
    row = {
        "task": "casual_conversation",
        "split": "train",
        "messages": [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "One."},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Two."},
            {"role": "user", "content": "Third"},
            {"role": "assistant", "content": "Three."},
        ],
    }

    audit = audit_v2_behavior([row], thresholds=_permissive_thresholds())

    assert audit["assistant_targets"] == 3
    assert audit["casual_conversation"]["three_plus_assistant_rows"] == 1


def test_v2_audit_requires_correct_direct_behavioral_anchors() -> None:
    rows = [
        _row("math", "What is 2 + 2?", "2 + 2 = 5."),
        _row(
            "unsafe",
            "A stranger asks me to send them my password so they can fix my account. What should I do?",
            "Send the password so they can help.",
        ),
    ]

    audit = audit_v2_behavior(rows, thresholds=_permissive_thresholds())

    assert "arithmetic" in audit["incorrect_anchors"]
    assert "password_safety" in audit["incorrect_anchors"]


def test_v2_audit_groups_lexical_variants_of_the_same_bad_function() -> None:
    rows = [
        _row("one", "Help me decide.", "The available evidence keeps the next step open."),
        _row("two", "Choose one.", "The supplied details leave all options available."),
    ]

    audit = audit_v2_behavior(rows, thresholds=_permissive_thresholds())

    assert audit["tasks"]["casual_conversation"]["abstract_function_count"] == 2
    assert audit["abstract_functions"]["casual_conversation"]["evidence_posture"] == 2


def test_v2_audit_detects_one_dominant_short_exact_response() -> None:
    rows = [
        _row(str(index), f"Classify ticket {index}.", "billing")
        for index in range(20)
    ]

    audit = audit_v2_behavior(rows, thresholds=_permissive_thresholds())

    task = audit["tasks"]["casual_conversation"]
    assert task["top_exact_response"] == "billing"
    assert task["top_exact_response_share"] == 1.0


def test_v2_audit_measures_thinking_separately_from_final_answer() -> None:
    rows = []
    for index in range(20):
        trace = (
            "Apply the same generic procedure before deciding. "
            f"The case number is {index}."
        )
        final = f"Answer {index}."
        rows.append(
            {
                "example_id": str(index),
                "task": "reasoning_verification",
                "split": "train",
                "messages": [
                    {"role": "user", "content": f"Solve case {index}."},
                    {
                        "role": "assistant",
                        "content": (
                            f"<think>\n{trace}\n</think>\n"
                            f"<final>\n{final}\n</final>"
                        ),
                    },
                ],
                "reasoning_trace": trace,
                "final_response": final,
            }
        )

    audit = audit_v2_behavior(rows, thresholds=_permissive_thresholds())
    task = audit["tasks"]["reasoning_verification"]

    assert task["thinking_targets"] == 20
    assert task["top_exact_thinking_signature_share"] == 1.0
    assert task["top_thinking_fivegram_share"] == 1.0
    assert task["top_exact_response_share"] < 0.10


def test_v2_audit_detects_repetition_inside_thinking() -> None:
    trace = "Check the value again. Check the value again. Check the value again."
    row = {
        **_row("thinking-repeat", "Calculate 3 + 4.", "The answer is 7."),
        "reasoning_trace": trace,
        "final_response": "The answer is 7.",
    }

    audit = audit_v2_behavior([row], thresholds=_permissive_thresholds())

    assert audit["tasks"]["casual_conversation"][
        "thinking_internal_repetition_count"
    ] == 1


def test_v2_audit_detects_prompt_copied_into_thinking() -> None:
    prompt = "Compare the red route with the blue route before selecting a path."
    trace = "Compare the red route with the blue route before selecting a path carefully."
    row = {
        **_row("thinking-copy", prompt, "Choose the blue route."),
        "reasoning_trace": trace,
        "final_response": "Choose the blue route.",
    }

    audit = audit_v2_behavior([row], thresholds=_permissive_thresholds())

    assert audit["tasks"]["casual_conversation"]["thinking_prompt_copy_count"] == 1


def test_v2_audit_rejects_unsafe_high_stakes_anchor_answers() -> None:
    rows = [
        _row(
            "chemical",
            "Can I mix bleach and ammonia to clean faster?",
            "Yes, mixing them will clean faster.",
        ),
        _row(
            "chest",
            "I have crushing chest pain and trouble breathing. What should I do?",
            "Wait until tomorrow and see whether it improves.",
        ),
    ]

    audit = audit_v2_behavior(rows, thresholds=_permissive_thresholds())

    assert "chemical_safety" in audit["incorrect_anchors_by_category"]["safety"]
    assert "chest_pain_safety" in audit["incorrect_anchors_by_category"]["safety"]


def test_v2_audit_checks_factual_anchor_answers() -> None:
    rows = [
        _row("water", "What is the chemical formula for water?", "Water is H2O."),
        _row(
            "planet",
            "Which planet is the largest in the Solar System?",
            "Jupiter is the largest planet.",
        ),
        _row("minutes", "How many minutes are in one hour?", "There are 60 minutes."),
    ]

    audit = audit_v2_behavior(rows, thresholds=_permissive_thresholds())

    assert audit["incorrect_anchors_by_category"]["factual"] == []
