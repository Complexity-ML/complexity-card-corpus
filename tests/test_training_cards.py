from complexity_card_corpus.training_cards import deal_training_cards


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
