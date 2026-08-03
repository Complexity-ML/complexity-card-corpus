import csv
import json
from pathlib import Path

from complexity_card_corpus.definition_acceptance import (
    accept_definition_proposals,
    apply_definition_overlay_data,
    apply_definition_overlay_to_placement,
)


def test_definition_overlay_preserves_statistical_gloss() -> None:
    result = apply_definition_overlay_data(
        {
            "audit": {},
            "words": {
                "atom": {
                    "short_definition": "A statistical gloss.",
                    "definition_kind": "masked_context_statistical_gloss",
                }
            },
        },
        {"atom": "A basic unit of matter."},
        consensus_by_token={"atom": "supported_by_both"},
    )
    entry = result["words"]["atom"]
    assert entry["short_definition"] == "A basic unit of matter."
    assert entry["statistical_gloss"] == "A statistical gloss."
    assert entry["definition_review"]["decision"] == "accepted"
    assert result["audit"]["operator_accepted_definitions"] == 1


def test_accept_definition_proposals_requires_explicit_acceptance(tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.json"
    proposals = tmp_path / "proposals.json"
    review = tmp_path / "review.csv"
    dictionary.write_text(
        json.dumps({"audit": {}, "words": {"atom": {"short_definition": "old"}}})
    )
    proposals.write_text(json.dumps({"definitions": {"atom": "new"}}))
    with review.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["token", "consensus", "reviewer_decision", "reviewer_notes"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "token": "atom",
                "consensus": "mixed",
                "reviewer_decision": "pending",
                "reviewer_notes": "",
            }
        )
    output_dictionary = tmp_path / "accepted.json"
    output_review = tmp_path / "accepted.csv"
    result = accept_definition_proposals(
        dictionary,
        proposals,
        review,
        output_dictionary,
        output_review,
        accept_all=True,
    )
    assert result["accepted_definitions"] == 1
    assert json.loads(output_dictionary.read_text())["words"]["atom"][
        "short_definition"
    ] == "new"
    with output_review.open() as stream:
        row = next(csv.DictReader(stream))
    assert row["reviewer_decision"] == "accepted"


def test_definition_overlay_updates_placement_and_preserves_gloss(
    tmp_path: Path,
) -> None:
    placement = tmp_path / "placement.csv"
    output = tmp_path / "accepted-placement.csv"
    with placement.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["token", "short_definition", "family", "domain"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "token": "atom",
                "short_definition": "A statistical gloss.",
                "family": "grounded_qa",
                "domain": "science",
            }
        )
    result = apply_definition_overlay_to_placement(
        placement,
        output,
        {"atom": "A basic unit of matter."},
        consensus_by_token={"atom": "supported_by_both"},
    )
    with output.open() as stream:
        row = next(csv.DictReader(stream))
    assert result["placement_definitions_updated"] == 1
    assert row["short_definition"] == "A basic unit of matter."
    assert row["statistical_gloss"] == "A statistical gloss."
    assert row["definition_review_decision"] == "accepted"
    assert row["definition_embedding_consensus"] == "supported_by_both"
