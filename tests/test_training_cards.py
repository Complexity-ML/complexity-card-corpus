from complexity_card_corpus.card_staticity import audit_card_staticity
from complexity_card_corpus.posttrain import (
    required_distinct_surfaces_per_source_card,
)
from complexity_card_corpus.training_cards import deal_training_cards


TASKS = (
    "brainstorming_creativity",
    "context_clarification",
    "conversation_empathy",
    "critique_revision",
    "explanation_learning",
    "extraction_classification",
    "grounded_qa",
    "planning_comparison",
    "practical_action",
    "reasoning_verification",
    "safety_uncertainty",
    "summarization_synthesis",
    "troubleshooting",
    "writing_transformation",
)


def test_training_cards_are_deterministic_and_complete() -> None:
    first = deal_training_cards(
        task="grounded_qa",
        mode="instruct",
        example_id="example:1",
        metadata={"risk_level": "low", "state": "The evidence is complete."},
    )
    second = deal_training_cards(
        task="grounded_qa",
        mode="instruct",
        example_id="example:1",
        metadata={"risk_level": "low", "state": "The evidence is complete."},
    )
    assert first == second
    assert set(first.as_dict()) == {
        "surface",
        "dialogue_state",
        "output",
        "evidence",
        "reasoning",
        "style",
        "context_density",
        "noise",
        "uncertainty",
        "response_order",
        "response_bridge",
        "response_layout",
        "response_opening",
    }


def test_training_cards_respect_task_and_evidence_compatibility() -> None:
    cards = deal_training_cards(
        task="extraction_classification",
        mode="chat",
        example_id="example:conflict",
        metadata={
            "risk_level": "medium",
            "state": "Two available records contain conflicting details.",
        },
    )
    assert cards.output == "json_or_schema"
    assert cards.reasoning == "extract_then_validate"
    assert cards.evidence == "conflicting"
    assert cards.uncertainty == "preserve_conflict"
    assert cards.context_density in {"full", "focused"}


def test_constraint_tension_is_not_mislabeled_as_conflicting_evidence() -> None:
    cards = deal_training_cards(
        task="brainstorming_creativity",
        mode="instruct",
        example_id="constraint-tension",
        metadata={
            "state": "One appealing idea conflicts with a hard constraint.",
            "risk_level": "low",
        },
    )

    assert cards.evidence == "sufficient"
    assert cards.uncertainty == "answerable"


def test_chat_and_instruct_use_different_dialogue_decks() -> None:
    instruct = {
        deal_training_cards(
            task="grounded_qa",
            mode="instruct",
            example_id=f"instruct:{index}",
        ).dialogue_state
        for index in range(100)
    }
    chat = {
        deal_training_cards(
            task="grounded_qa",
            mode="chat",
            example_id=f"chat:{index}",
        ).dialogue_state
        for index in range(100)
    }
    assert instruct <= {"new_request", "bounded_request", "direct_request"}
    assert chat <= {
        "follow_up",
        "constraint_update",
        "clarification_resolved",
        "continued_request",
    }
    assert instruct.isdisjoint(chat)


def test_each_family_has_far_more_than_the_seven_surfaces_needed_for_100k() -> None:
    assert required_distinct_surfaces_per_source_card(15_000) == 7
    for task in TASKS:
        hands = {
            tuple(
                deal_training_cards(
                    task=task,
                    mode="instruct" if index % 2 == 0 else "chat",
                    example_id=f"scale:{task}:{index}",
                    metadata={"risk_level": "low"},
                ).as_dict().items()
            )
            for index in range(256)
        }
        assert len(hands) >= 100, (task, len(hands))


def test_training_card_staticity_is_measured_without_ids_or_rendered_text() -> None:
    hands = [
        deal_training_cards(
            task=TASKS[index % len(TASKS)],
            mode="instruct" if index % 2 == 0 else "chat",
            example_id=f"scenario:{index:05d}",
            metadata={"risk_level": ("low", "medium", "high")[index % 3]},
        ).as_dict()
        for index in range(15_000)
    ]
    audit = audit_card_staticity(hands)

    assert audit["hands"] == 15_000
    assert audit["unique_hands"] >= 1_000
    assert audit["maximum_hand_share"] < 0.005
    for axis in ("surface", "dialogue_state", "style", "context_density", "noise"):
        assert audit["axes"][axis]["unique_values"] >= 2
        assert audit["axes"][axis]["normalized_entropy"] > 0.45


def test_discursive_families_receive_many_response_structure_hands() -> None:
    for task in (
        "explanation_learning",
        "planning_comparison",
        "practical_action",
        "reasoning_verification",
        "summarization_synthesis",
    ):
        hands = {
            deal_training_cards(
                task=task,
                mode="instruct",
                example_id=f"response-structure:{task}:{index}",
            ).response_structure_signature
            for index in range(512)
        }
        assert len(hands) >= 40, (task, len(hands))
