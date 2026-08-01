from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

import complexity_card_corpus.post_training as post_training
from complexity_card_corpus.english_morphology import (
    correct_indefinite_articles,
    indefinite_article,
)
from complexity_card_corpus.post_training import (
    REVIEW_GRADES,
    _apply_vocabulary_placements,
    audit_human_review,
    build_post_training_corpus,
)
from complexity_card_corpus.scenario_forge import build_scenario_forge
from complexity_card_corpus.task_cards import deal_task_hand


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/scenario-forge/scenario-forge-v1.json"


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
    assert sum(
        "statistical_alternative" in row.get("lexical_assignment_method", "")
        for row in enriched
    ) == 2


def _task_row(
    family: str,
    domain: str,
    *,
    scenario: str = "abcdef123456",
    constraint: str = "Keep the action bounded.",
) -> dict:
    return {
        "scenario_id": f"scenario:{scenario}",
        "family": family,
        "domain": domain,
        "intent": "verify",
        "constraint": constraint,
        "semantic_payload": json.dumps(
            {
                "subject": domain.replace("_", " "),
                "domain_context": f"Context for {domain}.",
            }
        ),
    }


def test_task_cards_do_not_invent_missing_trial_outcomes() -> None:
    hand = deal_task_hand(_task_row("critique_revision", "argument"), 0)
    assert "three of five" in hand.data.lower()
    assert "unrecorded" not in hand.answer.lower()
    assert "does not establish" in hand.answer.lower()


def test_email_critique_does_not_invent_revision_details() -> None:
    hand = deal_task_hand(_task_row("critique_revision", "email_draft"), 0)
    assert "Please send the files" in hand.answer
    assert "confirm the recipient, deadline, and file names" in hand.answer
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
    )
    for domain in domains:
        for variant in range(4):
            hand = deal_task_hand(_task_row("critique_revision", domain), variant)
            revision = hand.answer.split("Revision:", 1)[1].strip()
            sentences = re.findall(r"[^.!?]+[.!?](?:\s|$)", revision)
            assert len(sentences) == 2, (domain, variant, revision)
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
    assert all("Meal service is unknown" in answer for answer in answers)


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
    hand = deal_task_hand(
        _task_row("reasoning_verification", "work_allocation"),
        0,
    )
    assert "24" in hand.data
    assert "3 people" in hand.data
    assert "two rounds" in hand.data
    assert "8 items per person" in hand.answer
    assert "4 items per person per round" in hand.answer


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


def test_event_brainstorm_checks_every_hard_constraint() -> None:
    hand = deal_task_hand(
        _task_row(
            "brainstorming_creativity",
            "event_plan",
            scenario="000000000003",
        ),
        0,
    )
    assert "two hours" in hand.answer
    assert "three groups of eight" in hand.answer
    assert "registration nor personal records" in hand.answer


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
        assert "If either check fails" in hand.answer


def test_clarification_uses_the_hand_code_once() -> None:
    hand = deal_task_hand(_task_row("context_clarification", "incomplete_goal"), 1)
    assert hand.answer.count("ABCDEF") == 1
    assert "For hand ABCDEF: For ABCDEF:" not in hand.answer


def test_planning_card_states_the_failed_requirement_explicitly() -> None:
    hand = deal_task_hand(_task_row("planning_comparison", "learning_plan"), 0)
    assert "misses one non-negotiable requirement" in hand.data
    assert "non-negotiable requirement" in hand.answer
    assert "C fails" in hand.answer or "Reject C" in hand.answer
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
    left = post_training._masked_response(
        "Hand A1B2C3 — Ask support to reconcile A1B2C3-A and A1B2C3-B "
        "before 14:00 on day 17 for $42.",
        answer,
    )
    right = post_training._masked_response(
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


def test_intent_subject_composition_places_prepositional_complements_last() -> None:
    assert (
        post_training._intent_for_subject(
            "restructure for action", "a set of meeting notes"
        )
        == "restructure a set of meeting notes for action"
    )
    assert (
        post_training._intent_for_subject(
            "clarify the immediate need", "a tense conversation"
        )
        == "clarify the immediate need in a tense conversation"
    )
    assert (
        post_training._intent_for_subject(
            "adapt tone for the audience", "a project update"
        )
        == "adapt the tone of a project update for the audience"
    )


def test_post_training_corpus_groups_splits_and_builds_review_queue(
    tmp_path: Path,
) -> None:
    scenario_root = tmp_path / "scenarios"
    build_scenario_forge(REGISTRY, scenario_root)
    output = tmp_path / "post-training"
    result = build_post_training_corpus(
        scenario_root / "scenarios.parquet",
        output,
        variants_per_scenario=2,
        review_scenarios=140,
        seed=17,
    )

    rows = pq.read_table(output / "conversations.parquet").to_pylist()
    assert len(rows) == 30_000
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
        assert all(
            label in response
            for label in ("Immediate action:", "Boundary:", "Escalate")
        )
    assert result["audit"]["source_scenario_split_overlap"] == 0
    assert result["audit"]["semantic_group_split_overlap"] == 0
    paired_prompts = result["audit"]["paired_prompt_surface_stats"]
    assert paired_prompts["paired_scenarios"] == 15_000
    assert paired_prompts["exact_first_user_message_matches"] == 0
    assert paired_prompts["chat_opener_is_instruct_prefix"] == 0
    assert result["audit"]["split_holdout_units"] == [
        "scenario_id",
        "family+domain+intent",
    ]
    assert result["audit"]["exact_conversation_uniqueness_ratio"] == 1.0
    assert result["audit"]["exact_final_response_uniqueness_ratio"] >= 0.95
    assert result["audit"]["model_generated_dialogue_rows"] == 0
    assert result["audit"]["single_state_and_constraint_ratio"] == 1.0
    assert result["audit"]["natural_language_gate"] == {
        "assistant_meta_instruction_hits": 0,
        "user_meta_request_hits": 0,
        "forbidden_assistant_phrases": list(
            post_training._FORBIDDEN_ASSISTANT_META_PHRASES
        ),
        "forbidden_user_phrases": list(post_training._FORBIDDEN_USER_META_PHRASES),
    }
    role_stats = result["audit"]["role_text_stats"]
    assert role_stats["user_prompts"]["length"]["items"] == 45_000
    assert role_stats["assistant_messages"]["length"]["items"] == 45_000
    assert role_stats["final_responses"]["length"]["items"] == 30_000
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
