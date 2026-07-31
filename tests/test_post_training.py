from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow.parquet as pq

import complexity_card_corpus.post_training as post_training
from complexity_card_corpus.english_morphology import (
    correct_indefinite_articles,
    indefinite_article,
)
from complexity_card_corpus.post_training import (
    REVIEW_GRADES,
    audit_human_review,
    build_post_training_corpus,
)
from complexity_card_corpus.scenario_forge import build_scenario_forge


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/scenario-forge/scenario-forge-v1.json"


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
    assert post_training._intent_for_subject(
        "restructure for action", "a set of meeting notes"
    ) == "restructure a set of meeting notes for action"
    assert post_training._intent_for_subject(
        "clarify the immediate need", "a tense conversation"
    ) == "clarify the immediate need in a tense conversation"
    assert post_training._intent_for_subject(
        "adapt tone for the audience", "a project update"
    ) == "adapt the tone of a project update for the audience"


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
        review_scenarios=70,
        seed=17,
    )

    rows = pq.read_table(output / "conversations.parquet").to_pylist()
    assert len(rows) == 4_000
    assert result["audit"]["source_scenario_split_overlap"] == 0
    assert result["audit"]["semantic_group_split_overlap"] == 0
    paired_prompts = result["audit"]["paired_prompt_surface_stats"]
    assert paired_prompts["paired_scenarios"] == 2_000
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
        "forbidden_user_phrases": list(
            post_training._FORBIDDEN_USER_META_PHRASES
        ),
    }
    role_stats = result["audit"]["role_text_stats"]
    assert role_stats["user_prompts"]["length"]["items"] == 6_000
    assert role_stats["assistant_messages"]["length"]["items"] == 6_000
    assert role_stats["final_responses"]["length"]["items"] == 4_000
    assert role_stats["user_prompts"]["eight_grams"]["distinct_ngrams"] > 0
    assert role_stats["final_responses"]["eight_grams"]["distinct_ngrams"] > 0
    assert result["audit"]["fallback_surface_stats"][
        "maximum_formulation_share"
    ] < 0.05
    assert result["audit"]["conclusion_surface_stats"][
        "maximum_formulation_share"
    ] < 0.05
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
    assert result["audit"]["surface_pattern_stats"][
        "observed_opening_structure_pairs"
    ] > 144
    body = result["audit"]["body_surface_stats"]
    assert body["openings"]["maximum_formulation_share"] < 0.05
    assert body["actions"]["maximum_formulation_share"] < 0.05
    assert body["constraints"]["maximum_formulation_share"] < 0.05
    assert body["orders"]["patterns"] == 12
    assert body["masked_final_eight_token_phrase_ceiling"][
        "maximum_message_coverage"
    ] < 0.05

    source_splits: dict[str, set[str]] = {}
    for row in rows:
        scenario_id = json.loads(row["answer_json"])["scenario_id"]
        source_splits.setdefault(scenario_id, set()).add(row["split"])
    assert all(len(splits) == 1 for splits in source_splits.values())

    with (output / "human_review.csv").open(newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    assert len(review_rows) == 140
    assert len({row["scenario_id"] for row in review_rows}) == 70
    scenario_modes: dict[str, set[str]] = {}
    for row in review_rows:
        scenario_modes.setdefault(row["scenario_id"], set()).add(row["mode"])
    assert all(modes == {"instruct", "chat"} for modes in scenario_modes.values())
    assert {row["family"] for row in review_rows} == {
        "conversation_empathy",
        "explanation_learning",
        "planning_comparison",
        "practical_action",
        "safety_uncertainty",
        "troubleshooting",
        "writing_transformation",
    }
    assert {row["review_status"] for row in review_rows} == {"pending"}
    assert all(row["reviewer_notes"] == "" for row in review_rows)
    pending = audit_human_review(output / "human_review.csv")
    assert pending["training_ready"] is False
    assert pending["source_scenarios"] == 70
    assert pending["coverage"]["mode_rows"] == {"chat": 70, "instruct": 70}
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
        "scenario_sample_size": 70,
        "upper_defect_rate_if_iid_random": 0.041893,
        "caveat": (
            "descriptive sensitivity bound only; this review is stratified "
            "rather than a simple iid random sample"
        ),
    }
