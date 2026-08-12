from complexity_card_corpus.card_staticity import audit_card_staticity
from complexity_card_corpus.posttrain import (
    required_distinct_surfaces_per_source_card,
)
from complexity_card_corpus.sft.answer_development import development_card_count
from complexity_card_corpus.sft.dialogue_links import (
    LINK_MOVES,
    dialogue_link_card_count,
    dialogue_link_move,
    family_dialogue_card_count,
    preserve_linked_dialogue,
)
from complexity_card_corpus.training_cards import (
    RESPONSE_STRUCTURE_SIBLING_TASKS,
    TrainingCards,
    deal_training_cards,
    natural_dialogue_deck,
    projected_difficulty,
)


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
        "natural_opening",
        "natural_link",
        "natural_update",
        "natural_depth",
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


def test_projected_difficulty_uses_realized_conversation_complexity() -> None:
    easy = TrainingCards(
        surface="direct",
        dialogue_state="new_request",
        output="direct_prose",
        evidence="sufficient",
        reasoning="direct_response",
        style="plain",
        context_density="focused",
        noise="none",
        uncertainty="answerable",
    )
    assert projected_difficulty(
        easy,
        [{"role": "user", "content": "Explain this briefly."}],
    ) == "easy"

    medium = TrainingCards(
        **{
            **easy.as_dict(),
            "context_density": "full",
            "natural_depth": "linked",
        }
    )
    linked_messages = [
        {"role": "user", "content": "Compare the available options."},
        {"role": "assistant", "content": "Which constraint matters most?"},
        {"role": "user", "content": "Cost matters most."},
    ]
    assert projected_difficulty(medium, linked_messages) == "medium"

    hard = TrainingCards(
        **{
            **easy.as_dict(),
            "context_density": "full",
            "noise": "secondary_detail",
            "evidence": "conflicting",
            "uncertainty": "preserve_conflict",
            "natural_depth": "linked",
        }
    )
    assert projected_difficulty(hard, linked_messages) == "hard"

    partial_but_direct = TrainingCards(
        **{
            **easy.as_dict(),
            "evidence": "partial",
            "uncertainty": "state_limits",
        }
    )
    assert projected_difficulty(
        partial_but_direct,
        [{"role": "user", "content": "Answer only what the note supports."}],
    ) == "easy"

    seventy_nine_words = " ".join(["value"] * 79)
    eighty_words = " ".join(["value"] * 80)
    assert projected_difficulty(
        partial_but_direct,
        [{"role": "user", "content": seventy_nine_words}],
    ) == "easy"
    assert projected_difficulty(
        partial_but_direct,
        [{"role": "user", "content": eighty_words}],
    ) == "medium"


def test_single_role_response_tasks_are_not_given_fake_sibling_axes() -> None:
    assert "troubleshooting" not in RESPONSE_STRUCTURE_SIBLING_TASKS
    for task in RESPONSE_STRUCTURE_SIBLING_TASKS:
        assert len(
            {
                deal_training_cards(
                    task=task,
                    mode="instruct",
                    example_id=f"sibling-order:{index}",
                ).response_order
                for index in range(256)
            }
        ) >= 2


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
        "objection",
        "correction",
        "validation",
    }
    assert instruct.isdisjoint(chat)


def test_each_family_has_linkage_cards_and_no_generic_development_reservoir() -> None:
    assert dialogue_link_card_count() == 75
    assert set(natural_dialogue_deck()) == set(TASKS)
    for task in TASKS:
        assert family_dialogue_card_count(task) == 7
    previously_developed = {
        "context_clarification",
        "conversation_empathy",
        "critique_revision",
        "explanation_learning",
        "grounded_qa",
        "reasoning_verification",
        "summarization_synthesis",
    }
    for task in previously_developed:
        assert development_card_count(task) == 0


def test_grounded_qa_deals_many_visible_response_structures() -> None:
    hands = {
        deal_training_cards(
            task="grounded_qa",
            mode="instruct",
            example_id=f"grounded-response-hand:{index}",
        ).response_structure_signature
        for index in range(4_096)
    }

    assert len(hands) >= 400


def test_every_compositional_family_can_stay_below_five_percent_siblings() -> None:
    for task in sorted(RESPONSE_STRUCTURE_SIBLING_TASKS):
        dealt = [
            deal_training_cards(
                task=task,
                mode="instruct",
                example_id=f"sibling-capacity:{task}:{index}",
            )
            for index in range(8_192)
        ]
        neighbourhoods = {
            dimension: {
                cards.response_structure_sibling_signatures[dimension]
                for cards in dealt
            }
            for dimension in dealt[0].response_structure_sibling_signatures
        }
        assert min(map(len, neighbourhoods.values())) >= 20, (
            task,
            {dimension: len(values) for dimension, values in neighbourhoods.items()},
        )


def test_weak_families_deal_multiple_visible_response_layouts() -> None:
    expected = {
        "context_clarification": {
            "paragraph",
            "line_breaks",
            "spaced_lines",
            "opening_break",
        },
        "conversation_empathy": {
            "paragraph",
            "line_breaks",
            "spaced_lines",
            "opening_break",
        },
        "safety_uncertainty": {"paragraph", "line_breaks", "bullets"},
    }
    for task, layouts in expected.items():
        dealt = {
            deal_training_cards(
                task=task,
                mode="instruct",
                example_id=f"visible-layout:{task}:{index}",
            ).response_layout
            for index in range(512)
        }
        assert dealt == layouts


def test_chat_card_deals_cover_every_required_link_move() -> None:
    for task in TASKS:
        moves = {
            dialogue_link_move(
                deal_training_cards(
                    task=task,
                    mode="chat",
                    example_id=f"dialogue-link:{task}:{index}",
                ),
                f"dialogue-link:{task}:{index}",
            )
            for index in range(512)
        }
        assert moves == set(LINK_MOVES), (task, moves)


def test_linked_dialogue_sampling_is_deterministic_and_progressive() -> None:
    first = [preserve_linked_dialogue(f"linked:{index}") for index in range(1_000)]
    second = [preserve_linked_dialogue(f"linked:{index}") for index in range(1_000)]

    assert first == second
    assert 160 <= sum(first) <= 240


def test_natural_dialogue_subdecks_cover_every_family() -> None:
    for task in TASKS:
        hands = [
            deal_training_cards(
                task=task,
                mode="chat",
                example_id=f"natural-deck:{task}:{index}",
            )
            for index in range(512)
        ]
        assert len({hand.natural_opening for hand in hands}) >= 3
        assert len({hand.natural_link for hand in hands}) >= 3
        assert len({hand.natural_update for hand in hands}) >= 3
        assert {hand.natural_depth for hand in hands} == {"direct", "linked"}


def test_each_family_has_far_more_than_the_seven_surfaces_needed_for_100k() -> None:
    assert required_distinct_surfaces_per_source_card(15_000, 100_000) == 7
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
