from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from complexity_card_corpus.english_morphology import (
    correct_indefinite_articles,
    indefinite_article,
)
from complexity_card_corpus.posttrain import (
    REVIEW_GRADES,
    audit_human_review,
    build_post_training_corpus,
    required_distinct_surfaces_per_source_card,
)
from complexity_card_corpus.posttrain.constants import (
    _FORBIDDEN_ASSISTANT_META_PHRASES,
    _FORBIDDEN_USER_META_PHRASES,
)
from complexity_card_corpus.posttrain.build import _parallel_conversation_rows
from complexity_card_corpus.posttrain.metrics import _masked_response
from complexity_card_corpus.posttrain.rendering import (
    _apply_vocabulary_placements,
    _balance_conversation_families,
    _intent_for_subject,
)
from complexity_card_corpus.scenarios import (
    build_scenario_forge,
    compile_scenarios,
    load_scenario_registry,
)
from complexity_card_corpus.tasks import deal_task_hand
from complexity_card_corpus.tasks.core import DealtCard, LinkedSubcardDeck, SubcardPool
from complexity_card_corpus.tasks.intent_contracts import (
    intent_contract_catalog,
    intent_contract_for,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/scenario-forge/scenario-forge-v1.json"
EXPECTED_SCENARIOS = 33_320


def test_linked_subcard_deck_never_walks_an_incompatible_edge() -> None:
    deck = LinkedSubcardDeck(
        pools=(
            SubcardPool("input", ("A", "B")),
            SubcardPool("rule", ("A-rule", "B-rule")),
            SubcardPool("result", ("A-result", "B-result")),
        ),
        links=(((0, 0), (1, 1)), ((0, 0), (1, 1))),
    )
    assert deck.pool_names == ("input", "rule", "result")
    row = {"scenario_id": "scenario:linked-subcards"}
    paths = {deck.deal(row, variant, "linked-test") for variant in range(64)}
    assert paths == {
        ("A", "A-rule", "A-result"),
        ("B", "B-rule", "B-result"),
    }


def test_vocabulary_overflow_uses_compatible_capacity_without_dropping_words() -> None:
    scenarios = [
        {"scenario_id": "a", "family": "writing_transformation", "domain": "email"},
        {"scenario_id": "b", "family": "writing_transformation", "domain": "summary"},
        {"scenario_id": "c", "family": "writing_transformation", "domain": "summary"},
    ]
    placements = [
        {
            "token": token,
            "family": "writing_transformation",
            "domain": "email",
            "assignment_method": "cross_source_context",
            "statistical_usages_json": json.dumps(
                [
                    {
                        "family": "writing_transformation",
                        "domain": "summary",
                        "rank": 2,
                        "score": 8.0,
                    }
                ]
            ),
        }
        for token in ("clear", "concise", "faithful")
    ]

    enriched = _apply_vocabulary_placements(scenarios, placements)

    assert {row.get("lexical_focus") for row in enriched} == {
        "clear",
        "concise",
        "faithful",
    }
    assert (
        sum(
            "statistical_alternative" in row.get("lexical_assignment_method", "")
            for row in enriched
        )
        == 2
    )


def _balanced_row(
    scenario_id: str,
    mode: str,
    *,
    lexical_focus: str = "",
) -> dict:
    return {
        "example_id": f"{scenario_id}:{mode}",
        "task": "grounded_qa",
        "mode": mode,
        "answer_json": json.dumps(
            {
                "scenario_id": scenario_id,
                "lexical_focus": lexical_focus,
            }
        ),
    }


def test_family_cap_preserves_complete_instruct_chat_scenarios() -> None:
    rows = [
        _balanced_row(scenario, mode)
        for scenario in ("scenario:a", "scenario:b", "scenario:c")
        for mode in ("instruct", "chat")
    ]

    kept, audit = _balance_conversation_families(
        rows,
        max_examples_per_family=4,
    )

    modes_by_scenario: dict[str, set[str]] = {}
    for row in kept:
        scenario = json.loads(row["answer_json"])["scenario_id"]
        modes_by_scenario.setdefault(scenario, set()).add(row["mode"])
    assert len(kept) == 4
    assert all(modes == {"chat", "instruct"} for modes in modes_by_scenario.values())
    assert audit["policy"] == "explicit_manual_cap_by_complete_scenario"


def test_family_cap_rejects_limit_that_would_drop_lexical_coverage() -> None:
    rows = [
        _balanced_row("scenario:lexical", mode, lexical_focus="abundant")
        for mode in ("instruct", "chat")
    ] + [
        _balanced_row("scenario:plain", mode)
        for mode in ("instruct", "chat")
    ]

    with pytest.raises(ValueError, match="at least 2 rows are required"):
        _balance_conversation_families(rows, max_examples_per_family=1)

    kept, _audit = _balance_conversation_families(
        rows,
        max_examples_per_family=2,
    )
    assert {
        json.loads(row["answer_json"])["lexical_focus"] for row in kept
    } == {"abundant"}
    assert {row["mode"] for row in kept} == {"chat", "instruct"}


def test_post_training_rejects_one_variant_before_rendering_review_set(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="at least 2.*one instruct and one chat projection",
    ):
        build_post_training_corpus(
            tmp_path / "not-read.parquet",
            tmp_path / "output",
            variants_per_scenario=1,
            review_scenarios=140,
        )


def test_parallel_rendering_is_byte_for_byte_deterministic() -> None:
    scenarios = compile_scenarios(
        load_scenario_registry(REGISTRY),
        target_scenarios=EXPECTED_SCENARIOS,
    )[:16]
    serial = _parallel_conversation_rows(
        scenarios,
        3,
        vocabulary_placements=[],
        workers=1,
    )
    parallel = _parallel_conversation_rows(
        scenarios,
        3,
        vocabulary_placements=[],
        workers=4,
    )

    assert parallel == serial


def _task_row(
    family: str,
    domain: str,
    *,
    scenario: str = "abcdef123456",
    intent: str | None = None,
    constraint: str = "Keep the action bounded.",
    state: str = "",
) -> dict:
    if intent is None:
        family_contracts = intent_contract_catalog().get(family)
        intent = next(iter(family_contracts)) if family_contracts else "extract"
    return {
        "scenario_id": f"scenario:{scenario}",
        "family": family,
        "domain": domain,
        "intent": intent,
        "constraint": constraint,
        "state": state,
        "semantic_payload": json.dumps(
            {
                "subject": domain.replace("_", " "),
                "domain_context": f"Context for {domain}.",
            }
        ),
    }


def test_task_cards_do_not_invent_missing_trial_outcomes() -> None:
    hand = deal_task_hand(_task_row("critique_revision", "argument"), 0)
    assert re.search(r"\d+ of \d+ testers", hand.data.lower())
    assert "unrecorded" not in hand.answer.lower()
    assert re.search(
        r"(?:does not establish|not a universal|avoid claiming|cannot establish)",
        hand.answer.lower(),
    )


def test_every_registered_domain_deals_a_valid_task_hand() -> None:
    registry = load_scenario_registry(REGISTRY)
    scenarios = compile_scenarios(
        registry,
        target_scenarios=EXPECTED_SCENARIOS,
    )
    representatives: dict[tuple[str, str], dict] = {}
    for scenario in scenarios:
        representatives.setdefault(
            (scenario["family"], scenario["domain"]),
            scenario,
        )

    expected_pairs = {
        (family.family_id, domain.domain_id)
        for family in registry.families
        for domain in family.domains
    }
    assert set(representatives) == expected_pairs
    assert len(representatives) >= 100
    for pair, scenario in representatives.items():
        for variant in range(4):
            hand = deal_task_hand(scenario, variant)
            assert hand.data.strip(), pair
            assert hand.goal.strip(), pair
            assert hand.answer.strip(), pair
            assert hand.contract, pair
            for layer in (hand.data, hand.goal, hand.answer):
                assert isinstance(layer, DealtCard), (pair, layer)
                assert layer.deck_name
                assert layer.deck_lineage[0] == layer.deck_name
                assert layer.pool_names
                assert layer.variable_by
                assert set(layer.pool_names) <= set(layer.variable_by)
                for topology in layer.deck_topologies:
                    assert len(topology.pool_sizes) == len(topology.pool_names)
                    assert topology.variable_by[: len(topology.pool_names)] == (
                        topology.pool_names
                    )
                    assert set(topology.pool_names) <= set(topology.variable_by)
                    assert all(size >= 1 for size in topology.pool_sizes)
                assert len(layer.compatibility_links) == len(layer.pool_names) - 1
            if pair[0] != "extraction_classification":
                assert len(hand.answer.deck_lineage) >= 2


def test_every_non_json_family_intent_has_an_authored_completion_contract() -> None:
    registry = load_scenario_registry(REGISTRY)
    catalog = intent_contract_catalog()
    expected = {
        family.family_id: {intent.atom_id for intent in family.intents}
        for family in registry.families
        if family.family_id != "extraction_classification"
    }

    assert set(catalog) == set(expected)
    for family, intents in expected.items():
        assert set(catalog[family]) == intents
        assert all(len(contract) >= 3 for contract in catalog[family].values())
        assert len(set(catalog[family].values())) >= 4


def test_dealt_non_json_hands_expose_their_intent_contract() -> None:
    scenarios = compile_scenarios(
        load_scenario_registry(REGISTRY),
        target_scenarios=EXPECTED_SCENARIOS,
    )
    representatives: dict[tuple[str, str], dict] = {}
    for scenario in scenarios:
        if scenario["family"] != "extraction_classification":
            representatives.setdefault(
                (scenario["family"], scenario["intent"]),
                scenario,
            )

    for pair, scenario in representatives.items():
        assert deal_task_hand(scenario, 0).contract == intent_contract_for(
            scenario
        ), pair


def test_every_family_answer_deck_has_deep_generalist_reservoirs() -> None:
    registry = load_scenario_registry(REGISTRY)
    scenarios = compile_scenarios(
        registry,
        target_scenarios=EXPECTED_SCENARIOS,
    )
    representatives: dict[tuple[str, str], dict] = {}
    for scenario in scenarios:
        representatives.setdefault(
            (scenario["family"], scenario["domain"]),
            scenario,
        )

    covered_families: set[str] = set()
    for (family, _domain), scenario in representatives.items():
        answer = deal_task_hand(scenario, 0).answer
        semantic_decks = tuple(
            topology
            for topology in answer.deck_topologies
            if not topology.name.endswith("-surface")
        )
        assert semantic_decks, family
        for topology in semantic_decks:
            assert all(size >= 3 for size in topology.pool_sizes), (
                family,
                topology.name,
                dict(zip(topology.pool_names, topology.pool_sizes, strict=True)),
            )
        covered_families.add(family)

    assert covered_families == {family.family_id for family in registry.families}


def test_every_family_composes_task_hands_from_independent_card_decks() -> None:
    registry = load_scenario_registry(REGISTRY)
    for family in registry.families:
        for domain in family.domains:
            row = _task_row(family.family_id, domain.domain_id)
            hands = [deal_task_hand(row, variant) for variant in range(64)]
            data_cards = {hand.data for hand in hands}
            goal_cards = {hand.goal for hand in hands}
            surfaces = {(hand.data, hand.goal, hand.answer) for hand in hands}
            assert len(data_cards) >= 3, (
                family.family_id,
                domain.domain_id,
                "data",
                len(data_cards),
            )
            assert len(goal_cards) >= 3, (
                family.family_id,
                domain.domain_id,
                "goal",
                len(goal_cards),
            )
            assert len(surfaces) >= 12, (
                family.family_id,
                domain.domain_id,
                "full_hand",
                len(surfaces),
            )


def test_email_critique_does_not_invent_revision_details() -> None:
    hand = deal_task_hand(_task_row("critique_revision", "email_draft"), 0)
    assert re.search(r"\d+-page", hand.answer)
    assert re.search(r"\d+ attachments", hand.answer)
    assert all(
        detail in hand.answer.lower()
        for detail in ("recipient", "deadline", "file names")
    )
    assert "16:00" not in hand.answer
    assert "project team" not in hand.answer
    assert "two review files" not in hand.answer


def test_every_critique_provides_a_faithful_two_sentence_revision() -> None:
    domains = (
        "email_draft",
        "argument",
        "project_plan",
        "explanation",
        "instructions",
        "summary",
        "claim_evidence",
        "interface_copy",
        "status_update",
        "survey_report",
        "policy_notice",
        "data_caption",
        "release_note",
        "support_macro",
        "risk_assessment",
    )
    for domain in domains:
        revisions = set()
        for variant in range(16):
            hand = deal_task_hand(_task_row("critique_revision", domain), variant)
            revision = hand.answer.split("Revision:", 1)[1].strip()
            revisions.add(revision)
            sentences = re.findall(r"[^.!?]+[.!?](?:\s|$)", revision)
            assert len(sentences) == 2, (domain, variant, revision)
        assert len(revisions) >= 4, (domain, revisions)
    combined = " ".join(
        deal_task_hand(_task_row("critique_revision", domain), 0).answer
        for domain in domains
    )
    assert "Mara" not in combined
    assert "Tuesday" not in combined
    assert "Nia" not in combined
    assert "connection ended" not in combined


def test_grounded_qa_situation_matches_the_supplied_source() -> None:
    hand = deal_task_hand(_task_row("grounded_qa", "historical_note"), 3)
    assert "supplied source" in (hand.situation_title or "").lower()
    assert "documented part" in (hand.situation or "").lower()
    assert "conflict" not in (hand.situation or "").lower()
    assert ", The bridge" not in hand.answer
    assert "supplied source" in (hand.rule or "").lower()
    assert "unknown" in (hand.rule or "").lower()


def test_grounded_qa_facts_vary_across_scenarios_without_changing_contract() -> None:
    answers = {
        deal_task_hand(
            _task_row(
                "grounded_qa",
                "travel_information",
                scenario=f"{i:06x}grounded",
            ),
            0,
        ).answer
        for i in range(24)
    }
    assert len(answers) >= 20
    assert all("Meal service isn't specified" in answer for answer in answers)


def test_safety_boundaries_match_the_risk_domain() -> None:
    financial = " ".join(
        deal_task_hand(
            _task_row("safety_uncertainty", "financial_decision"),
            variant,
        ).answer
        for variant in range(4)
    ).lower()
    medical = " ".join(
        deal_task_hand(
            _task_row("safety_uncertainty", "medical_information"),
            variant,
        ).answer
        for variant in range(4)
    ).lower()
    assert "diagnos" not in financial
    assert "fraud team" in financial
    assert "diagnos" in medical
    assert "fraud team" not in medical


def test_safety_boundaries_materialize_scenario_state_and_constraint() -> None:
    first = deal_task_hand(
        _task_row(
            "safety_uncertainty",
            "financial_decision",
            state="The facts suggest risk but do not establish urgency.",
            constraint="Prefer the reversible option with the least credible harm.",
        ),
        0,
    )
    second = deal_task_hand(
        _task_row(
            "safety_uncertainty",
            "financial_decision",
            state="The available facts indicate an active risk.",
            constraint="Prioritize immediate harm reduction over detailed analysis.",
        ),
        0,
    )

    assert first.answer != second.answer
    first_answer = first.answer.lower()
    assert any(
        phrase in first_answer
        for phrase in ("urgenc", "severity", "caution", "possibility of harm")
    )
    assert "reversible" in first.answer
    second_answer = second.answer.lower()
    assert any(
        phrase in second_answer
        for phrase in ("active risk", "protective action", "treat the risk as active")
    )
    assert any(
        phrase in second_answer
        for phrase in (
            "immediate harm reduction",
            "protective action",
            "immediate risk first",
        )
    )


def test_safety_composition_keeps_sentences_out_of_noun_slots() -> None:
    domains = (
        "privacy_security",
        "medical_information",
        "financial_decision",
        "physical_safety",
    )
    for domain in domains:
        for index in range(48):
            hand = deal_task_hand(
                _task_row(
                    "safety_uncertainty",
                    domain,
                    scenario=f"{index:06x}safetygrammar",
                ),
                index % 8,
            )
            assert not re.search(
                r"\b(?:First|Directly|Specifically),\s+[A-Z]",
                hand.answer,
            ), hand.answer
            assert not re.search(r"\bthe\s+(?:Tell|Report|Mention)\b", hand.goal)
            assert not re.search(r"\.\s+channel\b", hand.goal, re.IGNORECASE)
            assert ".." not in hand.goal


def test_aliased_safety_domain_keeps_its_concrete_case() -> None:
    hand = deal_task_hand(
        _task_row(
            "safety_uncertainty",
            "animal_bite",
            state="Critical safety context is missing.",
            constraint="State material uncertainty instead of guessing.",
        ),
        0,
    )
    rendered = " ".join((hand.data, hand.goal, hand.answer)).lower()

    assert "animal bite" in rendered
    assert "chest pressure" not in rendered


def test_brainstorm_goal_contains_no_internal_rubric_label() -> None:
    forbidden = re.compile(
        r"\b(?:candidate task|option set|idea generation|brief comparison|"
        r"criteria review|fit check|pilot choice)\b",
        re.IGNORECASE,
    )
    for domain in ("names", "event_plan", "feature_ideas", "workflow"):
        for variant in range(6):
            hand = deal_task_hand(
                _task_row("brainstorming_creativity", domain),
                variant,
            )
            assert forbidden.search(hand.goal) is None, hand.goal


def test_empathy_answers_materialize_distinct_state_reflections() -> None:
    replaying = deal_task_hand(
        _task_row(
            "conversation_empathy",
            "work_stress",
            state="The person is repeatedly replaying the event.",
        ),
        0,
    )
    mixed = deal_task_hand(
        _task_row(
            "conversation_empathy",
            "work_stress",
            state="The person holds two conflicting feelings at once.",
        ),
        0,
    )

    assert replaying.answer != mixed.answer
    assert any(
        phrase in replaying.answer.lower()
        for phrase in ("replay", "same moment", "full review", "event can matter")
    )
    assert any(word in mixed.answer.lower() for word in ("both", "two", "mixed"))
    assert replaying.answer.count("?") == 1
    assert mixed.answer.count("?") == 1


def test_reasoning_situation_does_not_invent_a_unit_mismatch() -> None:
    hand = deal_task_hand(_task_row("reasoning_verification", "shopping_arithmetic"), 0)
    assert "complete calculation" in (hand.situation or "").lower()
    assert "different units" not in (hand.situation or "").lower()


def test_explanation_preserves_sentence_initial_acronyms() -> None:
    for variant in range(4):
        hand = deal_task_hand(
            _task_row("explanation_learning", "computing", scenario="611533d11592"),
            variant,
        )
        assert "RAM holds" in hand.answer
        assert "rAM" not in hand.answer


def test_physical_science_objects_receive_exactly_one_article() -> None:
    observed_objects = set()
    for index in range(64):
        hand = deal_task_hand(
            _task_row(
                "explanation_learning",
                "physical_science",
                scenario=f"{index:06x}physicalgrammar",
            ),
            index % 8,
        )
        rendered = " ".join((hand.data, hand.goal, hand.answer))
        assert not re.search(r"\b(?:the same|the)\s+(?:a|an)\s+", rendered, re.I)
        match = re.search(r"same (\w+), weighing", hand.data, re.I)
        assert match is not None, hand.data
        observed_objects.add(match.group(1).lower())

    assert observed_objects == {
        "rock",
        "backpack",
        "bicycle",
        "laptop",
        "book",
        "hammer",
        "suitcase",
        "chair",
    }


def test_software_resilience_explains_why_backup_precedes_an_update() -> None:
    hand = deal_task_hand(
        _task_row("explanation_learning", "software_resilience"),
        0,
    )
    answer = hand.answer.lower()
    assert "backup" in answer
    assert "restore" in answer or "recover" in answer
    assert "before" in answer
    assert "update" in answer


def test_work_allocation_calculates_people_and_rounds() -> None:
    equations = set()
    for index in range(32):
        hand = deal_task_hand(
            _task_row(
                "reasoning_verification",
                "work_allocation",
                scenario=f"{index:06x}abcdef",
            ),
            index % 8,
        )
        values = re.search(
            r"distribute (?P<items>\d+) items equally among (?P<people>\d+) "
            r"people, then divide each person's share equally across "
            r"(?P<rounds>\d+) rounds",
            hand.data,
        )
        assert values is not None, hand.data
        items = int(values["items"])
        people = int(values["people"])
        rounds = int(values["rounds"])
        assert items % (people * rounds) == 0
        per_person = items // people
        per_round = per_person // rounds
        equation = (
            f"{items} / {people} = {per_person}; "
            f"{per_person} / {rounds} = {per_round}"
        )
        equations.add(equation)
        assert equation in hand.answer
        assert f"{per_person} items per person" in hand.answer
        assert f"{per_round} items per person per round" in hand.answer
        assert (
            f"{people} people × {rounds} rounds × {per_round} items = {items} items"
            in hand.answer
        )
        reasoning_topology = next(
            topology
            for topology in hand.answer.deck_topologies
            if topology.name == "reasoning-answer"
        )
        assert {
            "scenario[equation]",
            "scenario[total]",
            "scenario[check]",
        } <= set(reasoning_topology.variable_by)
    assert len(equations) >= 24


def test_every_reasoning_domain_recomputes_and_avoids_equation_collisions() -> None:
    domains = (
        "shopping_arithmetic",
        "schedule_math",
        "unit_conversion",
        "proportions",
        "table_comparison",
        "sequence_pattern",
        "logical_constraints",
        "simple_probability",
        "work_allocation",
    )
    equations_by_domain: dict[str, set[str]] = {domain: set() for domain in domains}

    for domain in domains:
        for index in range(64):
            hand = deal_task_hand(
                _task_row(
                    "reasoning_verification",
                    domain,
                    scenario=f"{index:06x}calculation",
                ),
                index % 8,
            )
            numbers = tuple(map(int, re.findall(r"\d+", hand.data)))

            if domain in {"shopping_arithmetic", "schedule_math"}:
                units, each, extra = numbers[1:4]
                equation = f"{units} × {each} + {extra} = {units * each + extra}"
            elif domain == "unit_conversion":
                units = numbers[1]
                equation = f"{units} × 100 = {units * 100}"
            elif domain == "proportions":
                each, units = numbers[1:3]
                equation = f"{units} × {each} = {units * each}"
            elif domain == "table_comparison":
                units, each, repeated_units, extra = numbers[1:5]
                assert repeated_units == units
                result = max(units * each, units * extra)
                equation = f"max({units} × {each}, {units} × {extra}) = {result}"
            elif domain == "sequence_pattern":
                start, second, third = numbers[1:4]
                difference = second - start
                assert third - second == difference
                equation = f"{start} + 3 × {difference} = {start + 3 * difference}"
            elif domain == "logical_constraints":
                b_slot, c_slot = numbers[1:3]
                equation = f"({b_slot} - 1) + {c_slot} = {b_slot - 1 + c_slot}"
            elif domain == "simple_probability":
                blue, amber = numbers[1:3]
                equation = f"{blue} / ({blue} + {amber}) = {blue}/{blue + amber}"
            else:
                item_count, people, rounds = numbers[1:4]
                assert item_count % people == 0
                per_person = item_count // people
                assert per_person % rounds == 0
                per_round = per_person // rounds
                equation = (
                    f"{item_count} / {people} = {per_person}; "
                    f"{per_person} / {rounds} = {per_round}"
                )

            assert equation in hand.answer, (domain, hand.data, hand.answer)
            equations_by_domain[domain].add(equation)

    collisions = {
        domain: 64 - len(equations)
        for domain, equations in equations_by_domain.items()
    }
    assert all(len(equations) >= 55 for equations in equations_by_domain.values()), collisions


def test_conflicting_service_reports_preserve_and_resolve_the_conflict() -> None:
    hand = deal_task_hand(
        _task_row("grounded_qa", "conflicting_service_reports"),
        0,
    )
    answer = hand.answer.lower()
    assert "conflict" in answer or "disagree" in answer
    assert "time window" in answer
    assert "scope" in answer
    assert "direct check" in answer or "reproduce" in answer
    assert "choose" not in answer or "do not choose" in answer


def test_extraction_emits_exactly_the_requested_schema() -> None:
    hand = deal_task_hand(
        _task_row(
            "extraction_classification",
            "case_note",
            scenario="20c14d43c3ff",
        ),
        0,
    )
    assert set(json.loads(hand.answer)) == {
        "record_id",
        "case",
        "observed",
        "reported",
        "action",
        "next_owner",
    }


def test_extraction_record_labels_are_not_repeated() -> None:
    for domain in ("contact_record", "inventory_record"):
        hand = deal_task_hand(
            _task_row(
                "extraction_classification",
                domain,
                scenario="20c14d43c3ff",
            ),
            0,
        )
        assert "record record" not in hand.data.lower()


def test_extraction_intents_deal_distinct_json_contracts() -> None:
    expected_keys = {
        "classify": {
            "record_id",
            "label",
            "label_set",
            "completeness",
            "basis",
        },
        "missing": {
            "record_id",
            "record_type",
            "missing_fields",
            "present_fields",
            "present_sample",
            "next_action",
        },
        "standardize_records": {
            "record_id",
            "record_type",
            "normalized_record",
            "conflicts",
        },
    }
    for intent, keys in expected_keys.items():
        hand = deal_task_hand(
            _task_row(
                "extraction_classification",
                "contact_record",
                intent=intent,
                state="At least one required field is absent.",
            ),
            0,
        )
        assert set(json.loads(hand.answer)) == keys


def test_extraction_value_reservoir_varies_domain_records() -> None:
    records = []
    for index in range(24):
        hand = deal_task_hand(
            _task_row(
                "extraction_classification",
                "contact_record",
                scenario=f"{index:06x}abcdef",
                intent="extract",
                state="At least one required field is absent.",
            ),
            0,
        )
        record = json.loads(hand.answer)
        records.append((record["name"], record["role"], record["organization"]))
    assert len(set(records)) >= 12


def test_event_brainstorm_checks_every_hard_constraint() -> None:
    hands = (
        deal_task_hand(
            _task_row(
                "brainstorming_creativity",
                "event_plan",
                scenario=f"{index:012x}",
            ),
            0,
        )
        for index in range(128)
    )
    hand = next(
        candidate
        for candidate in hands
        if "without collecting participant data" in candidate.data
    )
    brief = re.search(
        r"(?P<minutes>\d+)-minute .* for (?P<people>\d+) people "
        r"within \$(?P<budget>\d+)",
        hand.data,
    )
    assert brief is not None, hand.data
    assert f"{brief['minutes']} minutes" in hand.answer
    assert f"{brief['people']} people" in hand.answer
    assert f"${brief['budget']}" in hand.answer
    assert re.search(r"\d+ groups of at most \d+", hand.answer)
    assert "registration nor personal records" in hand.answer
    variable_by = next(
        topology
        for topology in hand.answer.deck_topologies
        if topology.name == "brainstorm-answer"
    )
    assert variable_by.variable_by[:4] == (
        "options",
        "criteria",
        "outcome",
        "selection",
    )
    assert {
        "audience[common_noun]",
        "linker[measurement]",
        "linker[duration]",
        "unit[trial_round]",
        "scenario[scale]",
        "scenario[days]",
        "scenario[rounds]",
        "scenario[setting]",
        "scenario[signal]",
    } <= set(variable_by.variable_by)


def test_troubleshooting_honors_missing_administrator_access() -> None:
    hand = deal_task_hand(
        _task_row(
            "troubleshooting",
            "software_install",
            constraint="Administrator access is unavailable.",
        ),
        0,
    )
    assert "system-level change is out of scope" in hand.data
    assert "user-level test profile" in hand.answer
    assert "previous install directory" in hand.answer
    assert "original installer settings unchanged" in hand.answer


def test_troubleshooting_uses_a_concrete_reversible_step_for_each_domain() -> None:
    expected = {
        "software_install": "previous install directory",
        "network_connection": "previous resolver",
        "file_sync": "former folder name",
        "peripheral": "connect the keyboard directly",
        "web_form": "duplicate the draft",
        "data_pipeline": "previous column type",
    }
    for domain, phrase in expected.items():
        hand = deal_task_hand(_task_row("troubleshooting", domain), 0)
        assert phrase in hand.answer.lower()
        assert "Reverse only the recorded change" not in hand.answer
        assert any(
            marker in hand.answer
            for marker in (
                "If either check fails",
                "If the direct or regression result is negative",
                "Do not widen the change after a failed check",
            )
        )


def test_clarification_uses_the_hand_code_once() -> None:
    hand = deal_task_hand(_task_row("context_clarification", "incomplete_goal"), 1)
    assert hand.answer.count("ABCDEF") == 1
    assert "For hand ABCDEF: For ABCDEF:" not in hand.answer


def test_clarification_never_fabricates_a_request_number() -> None:
    domains = (
        "ambiguous_request",
        "missing_reference",
        "conflicting_instruction",
        "unclear_pronoun",
        "incomplete_goal",
        "scope_boundary",
        "format_preference",
        "timeline_ambiguity",
        "team_request",
        "data_request",
        "travel_request",
        "purchasing_request",
    )
    for domain in domains:
        for index in range(16):
            hand = deal_task_hand(
                _task_row(
                    "context_clarification",
                    domain,
                    scenario=f"{index:06x}clarification",
                ),
                index % 8,
            )
            rendered = " ".join(
                (
                    hand.situation_title or "",
                    str(hand.situation or ""),
                    hand.data,
                    hand.goal,
                    hand.answer,
                )
            )
            assert not re.search(r"\brequest\s+#\d+\b", rendered, re.I), rendered
            assert not re.search(r"\bfor request\s+\d+\b", rendered, re.I), rendered
            assert "the the" not in rendered.lower(), rendered


def test_planning_card_states_the_failed_requirement_explicitly() -> None:
    hand = deal_task_hand(_task_row("planning_comparison", "learning_plan"), 0)
    assert "misses one non-negotiable requirement" in hand.data
    assert re.search(
        r"C[^.]+(?:required condition|mandatory condition|non-negotiable requirement|unwaivable requirement)",
        hand.answer,
    )
    assert re.search(r"(?:C fails|Reject C|C is .*misses|C .*violates)", hand.answer)
    assert "Availability of Option A has not yet been confirmed" in hand.data
    assert "confirm" in hand.answer.lower()


def test_practical_card_uses_a_domain_specific_action() -> None:
    account = deal_task_hand(_task_row("practical_action", "account_access"), 0)
    flight = deal_task_hand(_task_row("practical_action", "air_travel"), 0)
    assert "booking" not in account.answer.lower()
    assert "account" in account.answer.lower()
    assert "flight" in flight.answer.lower() or "itinerary" in flight.answer.lower()


def test_practical_card_materializes_cost_and_accessibility_constraints() -> None:
    cost = deal_task_hand(
        _task_row(
            "practical_action",
            "subscriptions",
            constraint="The total cost cannot exceed a stated budget.",
        ),
        0,
    )
    accessible = deal_task_hand(
        _task_row(
            "practical_action",
            "public_transit",
            constraint="The result must preserve a stated accessibility need.",
        ),
        0,
    )
    assert "authorized maximum" in cost.data
    assert "accessibility support" in accessible.data
    assert "accessibility support" in accessible.answer


def test_missing_reference_uses_a_grounded_situation_card() -> None:
    hand = deal_task_hand(_task_row("context_clarification", "missing_reference"), 0)
    assert hand.situation_title == "Missing reference — request the absent report"
    assert "report is absent" in (hand.situation or "")
    assert "settings" not in hand.answer.lower()
    assert "two interpretations" not in hand.data.lower()
    assert "to do not" not in hand.answer.lower()


def test_masked_response_removes_identifiers_and_numeric_slots() -> None:
    answer = {
        "subject": "an account",
        "surface_intent": "resolve access",
        "state": "pending",
        "constraint": "confirm first",
        "desired_outcome": "access restored",
        "fallback": "pause",
        "fallback_surface": "pause",
        "domain_context": "official support",
    }
    left = _masked_response(
        "Hand A1B2C3 — Ask support to reconcile A1B2C3-A and A1B2C3-B "
        "before 14:00 on day 17 for $42.",
        answer,
    )
    right = _masked_response(
        "Hand D4E5F6 — Ask support to reconcile D4E5F6-A and D4E5F6-B "
        "before 09:00 on day 24 for $88.",
        answer,
    )
    assert left == right
    assert "a1b2c3" not in left
    assert "17" not in left


def test_indefinite_articles_follow_common_english_sound_rules() -> None:
    assert indefinite_article("account") == "an"
    assert indefinite_article("email") == "an"
    assert indefinite_article("hour") == "an"
    assert indefinite_article("useful option") == "a"
    assert indefinite_article("user request") == "a"
    assert indefinite_article("one-time code") == "a"
    assert correct_indefinite_articles("a account, a email, an useful option") == (
        "an account, an email, a useful option"
    )
    assert correct_indefinite_articles("an usable example") == "a usable example"
    assert correct_indefinite_articles(
        "A and B differ; A is valid; place A at 4; only A after screening; leave A alone."
    ) == (
        "A and B differ; A is valid; place A at 4; only A after screening; leave A alone."
    )


def test_intent_subject_composition_places_prepositional_complements_last() -> None:
    assert (
        _intent_for_subject("restructure for action", "a set of meeting notes")
        == "restructure a set of meeting notes for action"
    )
    assert (
        _intent_for_subject("clarify the immediate need", "a tense conversation")
        == "clarify the immediate need in a tense conversation"
    )
    assert (
        _intent_for_subject("adapt tone for the audience", "a project update")
        == "adapt the tone of a project update for the audience"
    )


def test_post_training_corpus_groups_splits_and_builds_review_queue(
    tmp_path: Path,
) -> None:
    scenario_root = tmp_path / "scenarios"
    build_scenario_forge(
        REGISTRY,
        scenario_root,
        target_scenarios=EXPECTED_SCENARIOS,
    )
    output = tmp_path / "post-training"
    result = build_post_training_corpus(
        scenario_root / "scenarios.parquet",
        output,
        variants_per_scenario=2,
        review_scenarios=140,
        seed=17,
        target_rows=100_000,
    )

    rows = pq.read_table(output / "conversations.parquet").to_pylist()
    assert result["audit"]["rows"] == len(rows)
    assert 66_000 <= len(rows) <= EXPECTED_SCENARIOS * 2
    assert len({row["response"] for row in rows}) == len(rows)
    family_responses: dict[str, list[str]] = {}
    for row in rows:
        transcript = row["rendered_text"]
        assert "SITUATION CARD" in transcript
        assert "DATA CARD" in transcript
        assert "RULE CARD" in transcript
        assert "GOAL CARD" in transcript
        answer = json.loads(row["answer_json"])
        assert answer["card_hand"]["cards"] == [
            "situation",
            "data",
            "rule",
            "goal",
        ]
        assert answer["card_hand"]["completion_contract"]
        assert "Source label:" not in transcript
        family_responses.setdefault(answer["family"], []).append(
            row["messages"][-1]["content"]
        )
    for response in family_responses["context_clarification"]:
        assert response.count("?") == 1
    assert all(
        "retain existing settings" not in response
        for response in family_responses["context_clarification"]
    )
    assert all(
        "until caption review" not in response.lower()
        for response in family_responses["writing_transformation"]
    )
    for response in family_responses["extraction_classification"]:
        assert isinstance(json.loads(response), dict)
    for response in family_responses["reasoning_verification"]:
        assert all(label in response for label in ("Equation:", "Total:", "Check:"))
    for response in family_responses["critique_revision"]:
        assert all(label in response for label in ("Weakness:", "Revision:"))
    for response in family_responses["brainstorming_creativity"]:
        assert all(label in response for label in ("1.", "2.", "3.", "Select"))
    for response in family_responses["safety_uncertainty"]:
        assert any(
            label in response
            for label in ("Immediate action:", "Protective step:", "First safeguard:")
        )
        assert any(
            label in response
            for label in ("Boundary:", "Safety boundary:", "Verification limit:")
        )
        assert any(
            label in response
            for label in ("Escalation:", "Trusted channel:", "Next contact:")
        )
    assert result["audit"]["source_scenario_split_overlap"] == 0
    assert result["audit"]["semantic_group_split_overlap"] == 0
    paired_prompts = result["audit"]["paired_prompt_surface_stats"]
    assert paired_prompts["paired_scenarios"] <= result["audit"]["source_scenarios"]
    assert paired_prompts["paired_scenarios"] >= int(
        result["audit"]["source_scenarios"] * 0.98
    )
    assert paired_prompts["exact_first_user_message_matches"] == 0
    assert paired_prompts["chat_opener_is_instruct_prefix"] == 0
    assert result["audit"]["split_holdout_units"] == [
        "scenario_id",
        "family+domain+intent",
    ]
    assert result["audit"]["exact_conversation_uniqueness_ratio"] == 1.0
    assert result["audit"]["exact_final_response_uniqueness_ratio"] == 1.0
    assert result["audit"]["model_generated_dialogue_rows"] == 0
    assert result["audit"]["single_state_and_constraint_ratio"] == 1.0
    card_game = result["audit"]["card_game"]
    assert card_game["all_hands_have_linked_deck_topology"] is True
    assert card_game["all_semantic_answer_reservoirs_have_three_subcards"] is True
    assert card_game["minimum_semantic_answer_pool_size"] >= 3
    assert result["audit"]["natural_language_gate"] == {
        "assistant_meta_instruction_hits": 0,
        "user_meta_request_hits": 0,
        "forbidden_assistant_phrases": list(_FORBIDDEN_ASSISTANT_META_PHRASES),
        "forbidden_user_phrases": list(_FORBIDDEN_USER_META_PHRASES),
    }
    role_stats = result["audit"]["role_text_stats"]
    expected_user_messages = sum(
        message["role"] == "user" for row in rows for message in row["messages"]
    )
    expected_assistant_messages = sum(
        message["role"] == "assistant" for row in rows for message in row["messages"]
    )
    assert role_stats["user_prompts"]["length"]["items"] == expected_user_messages
    assert (
        role_stats["assistant_messages"]["length"]["items"]
        == expected_assistant_messages
    )
    assert role_stats["final_responses"]["length"]["items"] == len(rows)
    assert role_stats["user_prompts"]["eight_grams"]["distinct_ngrams"] > 0
    assert role_stats["final_responses"]["eight_grams"]["distinct_ngrams"] > 0
    masked = result["audit"]["masked_response_diversity"]
    assert masked["masked_fields"] == [
        "subject",
        "intent",
        "state",
        "constraint",
        "desired_outcome",
        "fallback",
        "fallback_surface",
        "domain_context",
    ]
    assert masked["masked_surface_variables"] == [
        "scenario_code",
        "reference_suffix",
        "date",
        "amount",
        "time",
        "number",
    ]
    assert masked["maximum_skeleton_share"] < 0.05
    assert 0 < masked["exact_skeleton_uniqueness_ratio"] <= 1
    assert masked["eight_gram_stats"]["distinct_ngrams"] > 0
    eight_grams = result["audit"]["eight_gram_stats"]
    assert eight_grams["distinct_ngrams"] > 0
    assert 0 < eight_grams["distinct_ngram_ratio"] <= 1
    assert 0 < eight_grams["singleton_distinct_ratio"] <= 1
    assert eight_grams["maximum_occurrences"] >= 1
    assert eight_grams["top_repeated_ngrams"]
    assert "unique_rate" not in eight_grams
    assert 0 < result["audit"]["lexical_stats"]["mattr_100"] <= 1
    repetition_gate = result["audit"]["response_repetition_gate"]
    assert repetition_gate["measured_from_rendered_responses"] is True
    assert repetition_gate["maximum_masked_eight_token_message_coverage"] < 0.05
    assert all(
        metrics["maximum_masked_template_share"] < 0.20
        for metrics in result["audit"]["family_metrics"].values()
    )
    scale = result["audit"]["scale_100k"]
    assert required_distinct_surfaces_per_source_card(EXPECTED_SCENARIOS, 100_000) == 4
    assert scale["target_rows"] == 100_000
    assert scale["source_cards"] == EXPECTED_SCENARIOS
    assert scale["required_distinct_surfaces_per_source_card"] == 4
    assert scale["configured_variants_per_source_card"] == 2
    assert scale["configured_pre_deduplication_ceiling"] == 66_640
    assert scale["configured_variant_shortfall"] == 2
    assert scale["planned_distinct_surfaces_per_source_card"] == 8
    assert scale["planned_pre_deduplication_ceiling"] == 266_560
    assert scale["planned_capacity_exceeds_target"] is True
    assert scale["current_configuration_can_reach_target"] is False
    assert scale["target_generated"] is False
    assert scale["release_target_ready"] is False
    assert scale["claim_scope"].startswith("capacity contract only")
    assert scale["static_surface_hotspots"] == []
    assert scale["quality_gates"]["family_template_concentration"] is True

    source_splits: dict[str, set[str]] = {}
    for row in rows:
        scenario_id = json.loads(row["answer_json"])["scenario_id"]
        source_splits.setdefault(scenario_id, set()).add(row["split"])
    assert all(len(splits) == 1 for splits in source_splits.values())

    with (output / "human_review.csv").open(newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    assert len(review_rows) == 280
    assert len({row["scenario_id"] for row in review_rows}) == 140
    scenario_modes: dict[str, set[str]] = {}
    for row in review_rows:
        scenario_modes.setdefault(row["scenario_id"], set()).add(row["mode"])
    assert all(modes == {"instruct", "chat"} for modes in scenario_modes.values())
    assert {row["family"] for row in review_rows} == {
        "conversation_empathy",
        "brainstorming_creativity",
        "context_clarification",
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
    }
    assert {row["review_status"] for row in review_rows} == {"pending"}
    assert all(row["reviewer_notes"] == "" for row in review_rows)
    assert all("SITUATION CARD" in row["transcript"] for row in review_rows)
    assert all("RULE CARD" in row["transcript"] for row in review_rows)
    assert all("GOAL CARD" in row["transcript"] for row in review_rows)
    assert all(row["response"] in row["transcript"] for row in review_rows)
    pending = audit_human_review(output / "human_review.csv")
    assert pending["training_ready"] is False
    assert pending["source_scenarios"] == 140
    assert pending["coverage"]["mode_rows"] == {"chat": 140, "instruct": 140}
    assert set(pending["coverage"]["family_source_scenarios"].values()) == {10}
    assert set(pending["coverage"]["risk_source_scenarios"]) == {
        "critical",
        "high",
        "low",
        "medium",
    }
    assert set(pending["coverage"]["split_source_scenarios"]) == {
        "train",
        "validation",
    }

    for row in review_rows:
        row["review_status"] = "approved"
        for grade in REVIEW_GRADES:
            row[grade] = "pass"
        row["reviewer"] = "Boris Peyriguere"
        row["reviewed_at_utc"] = "2026-07-31T12:00:00Z"
    with (output / "human_review.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    completed = audit_human_review(output / "human_review.csv")
    assert completed["training_ready"] is True
    assert completed["review_provenance_complete"] is True
    assert completed["zero_failure_bound"] == {
        "confidence": 0.95,
        "scenario_sample_size": 140,
        "upper_defect_rate_if_iid_random": 0.021171,
        "caveat": (
            "descriptive sensitivity bound only; this review is stratified "
            "rather than a simple iid random sample"
        ),
    }
