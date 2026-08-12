from __future__ import annotations

import json

from complexity_card_corpus.sft.projection import _project_sft_exchange
from complexity_card_corpus.sft.reasoning_envelope import (
    REASONING_ENVELOPE_ACTIVE_TASKS,
    REASONING_ENVELOPE_TASKS,
    audit_reasoning_envelopes,
    parse_reasoning_envelope,
    render_reasoning_envelope,
)
from complexity_card_corpus.variable_by import reasoning_envelope_variable_by


CASES = {
    "reasoning_verification": {
        "source": (
            "Equation: 457 / (457 + 951) = 457/1408. "
            "Total: the resulting quantity is 457/1408 probability of blue. "
            "Check: the favorable and total counts are 457 and 1408."
        ),
        "final": (
            "The calculation is 457 / (457 + 951) = 457/1408. "
            "The result is 457/1408 probability of blue. "
            "The counts 457 and 1408 provide an independent check."
        ),
    },
    "planning_comparison": {
        "source": (
            "B exceeds the $125 cap and C misses a mandatory check. "
            "Choose A: a structured tutorial series. "
            "Sequence: confirm availability before payment. "
            "Fallback trigger: reopen the shortlist if A is unavailable."
        ),
        "final": (
            "Choose A: a structured tutorial series. Confirm availability "
            "before payment. Reopen the shortlist if A is unavailable."
        ),
    },
    "explanation_learning": {
        "source": (
            "Core idea: A primary source records direct evidence. "
            "Example: An original interview is primary evidence. "
            "Check: Which source contains the speaker's exact words?"
        ),
        "final": (
            "A primary source records direct evidence. An original interview "
            "is primary evidence. Which source contains the speaker's exact words?"
        ),
    },
    "critique_revision": {
        "source": (
            "Weakness: the deletion comes before backup and verification. "
            "Revision: Back up the folder, test the update in isolation, and "
            "retain the rollback copy through validation."
        ),
        "final": (
            "The deletion comes before backup and verification. Back up the "
            "folder and retain the rollback copy through validation."
        ),
    },
    "troubleshooting": {
        "source": (
            "1. Preserve the current log. 2. Bypass the hub for one test. "
            "3. Compare the new log with the baseline. Restore the original "
            "arrangement after a failed check."
        ),
        "final": (
            "1. Preserve the current log. 2. Bypass the hub for one test. "
            "3. Compare the new log with the baseline. Restore the original "
            "arrangement after a failed check."
        ),
    },
}


METADATA = {
    "state": "The current result has not been independently verified.",
    "constraint": "Use only supplied facts and keep every change reversible.",
    "desired_outcome": "The conclusion remains checkable from a second route.",
}


def test_v18_renders_balanced_variable_by_think_final_for_five_tasks() -> None:
    for task, case in CASES.items():
        envelopes = [
            render_reasoning_envelope(
                task=task,
                source_response=case["source"],
                natural_final=case["final"],
                metadata=METADATA,
                seed=f"{task}:{index}",
            )
            for index in range(240)
        ]
        assert all(envelope is not None for envelope in envelopes)
        rendered = [envelope for envelope in envelopes if envelope is not None]
        assert len({envelope.think for envelope in rendered}) >= 100
        assert len({envelope.card_hand for envelope in rendered}) >= 100
        assert all(parse_reasoning_envelope(envelope.text) for envelope in rendered)
        assert all("I should" not in envelope.think for envelope in rendered)
        assert all("<think>" not in envelope.final for envelope in rendered)
        assert all(
            label not in envelope.final.casefold()
            for envelope in rendered
            for label in ("equation:", "total:", "check:", "sequence:", "fallback trigger:")
        )


def test_v18_variable_by_has_nested_think_and_final_dependencies() -> None:
    variable_by = reasoning_envelope_variable_by(
        "reasoning_verification",
        analysis="The equation is 2 × 12 = 24.",
        analysis_inline="the equation is 2 × 12 = 24",
        verification="Adding 12 twice also gives 24.",
        verification_inline="adding 12 twice also gives 24",
        final_variants=(
            "The result is 24.",
            "The answer is 24.",
            "This gives 24.",
        ),
    )

    graph = variable_by.dependency_graph()

    assert "opening[reasoning_verification]" in graph["think[reasoning_verification]"]
    assert "scenario[analysis]" in graph["think[reasoning_verification]"]
    assert graph["final[reasoning_verification]"] == (
        "scenario[final_0]",
        "scenario[final_1]",
        "scenario[final_2]",
    )
    assert variable_by.deal_indices("one") == variable_by.deal_indices("one")


def test_v18_reasoning_final_never_invents_a_calculation_value() -> None:
    case = CASES["reasoning_verification"]

    for index in range(240):
        envelope = render_reasoning_envelope(
            task="reasoning_verification",
            source_response=case["source"],
            natural_final=case["final"],
            metadata=METADATA,
            seed=f"calculation:{index}",
        )

        assert envelope is not None
        assert "457/1408" in envelope.final
        assert "457" in envelope.think
        assert "951" in envelope.think
        assert "1408" in envelope.think


def test_v18_reasoning_final_drops_secondary_total_explanation() -> None:
    envelope = render_reasoning_envelope(
        task="reasoning_verification",
        source_response=(
            "Equation: 567 × 100 = 56700. "
            "Total: the computed value is 56700 centimetres. "
            "The unit identity 1 metre = 100 centimetres fixes the result. "
            "Check: dividing 56700 by 100 returns 567 metres."
        ),
        natural_final="56700 centimetres.",
        metadata=METADATA,
        seed="secondary-total-explanation",
    )

    assert envelope is not None
    assert "56700 centimetres" in envelope.final
    assert "unit identity" not in envelope.final.casefold()
    assert "1 metre" not in envelope.final.casefold()


def test_v18_planning_uses_last_choice_sentence_after_a_prefixed_criterion() -> None:
    envelope = render_reasoning_envelope(
        task="planning_comparison",
        source_response=(
            "A is the only compliant candidate: B exceeds the budget by $25, "
            "while C omits a mandatory element. Choose the viable option, A: "
            "a structured tutorial series. Sequence: confirm availability, "
            "then make payment. Fallback trigger: reopen the shortlist if A "
            "cannot be confirmed."
        ),
        natural_final=(
            "Choose the structured tutorial series, confirm availability, and "
            "reopen the shortlist if it cannot be confirmed."
        ),
        metadata=METADATA,
        seed="criteria-also-starts-with-choice-prefix",
    )

    assert envelope is not None
    assert "only compliant candidate" in envelope.think
    assert "structured tutorial series" in envelope.final


def test_v18_freeform_calculation_does_not_create_a_one_item_list() -> None:
    source = (
        "The total is 200 units because 5 × 40 = 200. "
        "Verify it by adding 40 exactly 5 times."
    )

    for index in range(48):
        envelope = render_reasoning_envelope(
            task="reasoning_verification",
            source_response=source,
            natural_final=source,
            metadata=METADATA,
            seed=f"freeform-calculation:{index}",
        )

        assert envelope is not None
        assert envelope.final == "The total is 200 units."
        assert not envelope.final.startswith(("- ", "1. "))
        assert (
            envelope.think.casefold().count(
                "verify it by adding 40 exactly 5 times."
            )
            == 1
        )


def test_v18_audit_checks_scope_lengths_and_collisions() -> None:
    rows = []
    for task in REASONING_ENVELOPE_ACTIVE_TASKS:
        case = CASES[task]
        for index in range(240):
            envelope = render_reasoning_envelope(
                task=task,
                source_response=case["source"],
                natural_final=case["final"],
                metadata=METADATA,
                seed=f"{task}:{index}",
            )
            assert envelope is not None
            rows.append(
                {
                    "example_id": f"{task}:{index}",
                    "task": task,
                    "_projected_target": envelope.text,
                }
            )
    rows.append(
        {
            "example_id": "casual:0",
            "task": "casual_conversation",
            "_projected_target": "That sounds relaxing. What did you enjoy most?",
        }
    )

    audit = audit_reasoning_envelopes(rows, enabled=True)

    assert audit["passed"] is True
    assert set(audit["tasks"]) == set(REASONING_ENVELOPE_ACTIVE_TASKS)
    assert all(
        task["maximum_card_hand_share"] <= 0.05
        for task in audit["tasks"].values()
    )
    assert audit["checks"]["reasoning_final_numbers_are_established_in_think"]


def test_v18_projection_is_opt_in_and_leaves_non_reasoning_natural() -> None:
    reasoning_messages = [
        {"role": "user", "content": "Calculate two groups of twelve."},
        {
            "role": "assistant",
            "content": (
                "Equation: 2 × 12 = 24. Total: 24 units. "
                "Check: adding 12 twice gives 24."
            ),
        },
    ]
    metadata = json.dumps(METADATA)

    _, v16_target, _ = _project_sft_exchange(
        reasoning_messages,
        example_id="reasoning:v16",
        task="reasoning_verification",
        answer_json=metadata,
    )
    _, v18_target, _ = _project_sft_exchange(
        reasoning_messages,
        example_id="reasoning:v18",
        task="reasoning_verification",
        answer_json=metadata,
        reasoning_envelope_version="v18",
        reasoning_seed="reasoning:v18",
    )
    _, casual_target, _ = _project_sft_exchange(
        [
            {"role": "user", "content": "I had a quiet afternoon."},
            {"role": "assistant", "content": "That sounds restorative."},
        ],
        example_id="casual:v18",
        task="casual_conversation",
        answer_json=metadata,
        reasoning_envelope_version="v18",
        reasoning_seed="casual:v18",
    )
    _, critique_target, _ = _project_sft_exchange(
        [
            {"role": "user", "content": "Revise this unsafe sequence."},
            {
                "role": "assistant",
                "content": CASES["critique_revision"]["source"],
            },
        ],
        example_id="critique:v18",
        task="critique_revision",
        answer_json=metadata,
        reasoning_envelope_version="v18",
        reasoning_seed="critique:v18",
    )

    assert parse_reasoning_envelope(v16_target) is None
    assert parse_reasoning_envelope(v18_target) is not None
    assert casual_target == "That sounds restorative."
    assert parse_reasoning_envelope(critique_target) is None
