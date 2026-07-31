from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow.parquet as pq

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
        review_rows=70,
        seed=17,
    )

    rows = pq.read_table(output / "conversations.parquet").to_pylist()
    assert len(rows) == 4_000
    assert result["audit"]["source_card_split_overlap"] == 0
    assert result["audit"]["split_holdout_unit"] == "scenario_id"
    assert result["audit"]["unique_rendered_ratio"] == 1.0
    assert result["audit"]["unique_final_response_ratio"] >= 0.95
    assert result["audit"]["model_generated_dialogue_rows"] == 0
    assert result["audit"]["single_state_and_constraint_ratio"] == 1.0
    assert result["audit"]["eight_gram_stats"]["unique_rate"] >= 0.14

    source_splits: dict[str, set[str]] = {}
    for row in rows:
        scenario_id = json.loads(row["answer_json"])["scenario_id"]
        source_splits.setdefault(scenario_id, set()).add(row["split"])
    assert all(len(splits) == 1 for splits in source_splits.values())

    with (output / "human_review.csv").open(newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    assert len(review_rows) == 70
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
    assert audit_human_review(output / "human_review.csv")["training_ready"] is False

    for row in review_rows:
        row["review_status"] = "approved"
        for grade in REVIEW_GRADES:
            row[grade] = "pass"
    with (output / "human_review.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    assert audit_human_review(output / "human_review.csv")["training_ready"] is True
