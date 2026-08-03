from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REVIEW_FIELDS = (
    "token",
    "status",
    "current_definition",
    "family",
    "domain",
    "primary_model",
    "primary_cosine",
    "primary_family_rank",
    "secondary_model",
    "secondary_cosine",
    "secondary_family_rank",
    "proposed_definition",
    "reviewer_decision",
    "reviewer_notes",
)


def build_dictionary_review_data(
    primary_guidance: dict[str, Any],
    secondary_guidance: dict[str, Any],
    definition_proposals: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Cross two local embedding audits into a human review queue.

    The primary model defines the bounded queue. Agreement from the secondary
    model raises an item from uncertain to likely_wrong. No definition is
    generated or replaced automatically.
    """

    primary_rows = primary_guidance["dictionary_coherence_audit"]["review_queue"]
    secondary_rows = secondary_guidance["dictionary_coherence_audit"]["review_queue"]
    primary_by_token = {str(row["token"]): row for row in primary_rows}
    secondary_by_token = {str(row["token"]): row for row in secondary_rows}
    primary_model = str(primary_guidance["model"]["name"])
    secondary_model = str(secondary_guidance["model"]["name"])

    proposals = definition_proposals or {}
    rows = []
    status_counts = {"undefined": 0, "uncertain": 0, "likely_wrong": 0}
    candidate_tokens = sorted(set(primary_by_token) | set(secondary_by_token))
    for token in candidate_tokens:
        primary = primary_by_token.get(token)
        secondary = secondary_by_token.get(token)
        reference = primary or secondary
        assert reference is not None
        primary_reasons = set(primary.get("review_reasons", [])) if primary else set()
        secondary_reasons = (
            set(secondary.get("review_reasons", [])) if secondary else set()
        )
        semantic_reasons = {
            "token_definition_alignment_robust_outlier",
            "embedding_family_rank_above_5",
        }
        if (
            primary and primary.get("review_status") == "undefined"
        ) or (
            secondary and secondary.get("review_status") == "undefined"
        ):
            status = "undefined"
        elif (primary_reasons & semantic_reasons) and (
            secondary_reasons & semantic_reasons
        ):
            status = "likely_wrong"
        else:
            status = "uncertain"
        status_counts[status] += 1
        rows.append(
            {
                "token": token,
                "status": status,
                "current_definition": str(reference["definition"]),
                "family": str(reference["family"]),
                "domain": str(reference["domain"]),
                "primary_model": primary_model,
                "primary_cosine": (
                    float(primary["token_definition_cosine"])
                    if primary is not None
                    else None
                ),
                "primary_family_rank": (
                    int(primary["selected_family_rank"])
                    if primary is not None
                    else None
                ),
                "secondary_model": secondary_model,
                "secondary_cosine": (
                    float(secondary["token_definition_cosine"])
                    if secondary is not None
                    else None
                ),
                "secondary_family_rank": (
                    int(secondary["selected_family_rank"])
                    if secondary is not None
                    else None
                ),
                "proposed_definition": str(proposals.get(token, "")),
                "reviewer_decision": "pending",
                "reviewer_notes": "",
            }
        )

    rows.sort(
        key=lambda row: (
            {"undefined": 0, "likely_wrong": 1, "uncertain": 2}[row["status"]],
            row["primary_cosine"] if row["primary_cosine"] is not None else 1.0,
            row["token"],
        )
    )
    return {
        "format": "embedding-dictionary-review-v1",
        "policy": {
            "automatic_definition_replacement": False,
            "primary_model_defines_queue": True,
            "likely_wrong_requires_two_model_agreement": True,
            "human_acceptance_required": True,
        },
        "primary_model": primary_guidance["model"],
        "secondary_model": secondary_guidance["model"],
        "rows": len(rows),
        "status_counts": status_counts,
        "proposed_definitions": sum(
            bool(row["proposed_definition"]) for row in rows
        ),
        "review": rows,
    }


def write_dictionary_review(
    primary_guidance_path: Path,
    secondary_guidance_path: Path,
    output_json: Path,
    output_csv: Path,
    proposals_path: Path | None = None,
) -> dict[str, Any]:
    primary = json.loads(primary_guidance_path.read_text(encoding="utf-8"))
    secondary = json.loads(secondary_guidance_path.read_text(encoding="utf-8"))
    proposals: dict[str, str] = {}
    if proposals_path is not None:
        proposal_document = json.loads(proposals_path.read_text(encoding="utf-8"))
        raw_proposals = proposal_document.get("definitions")
        if not isinstance(raw_proposals, dict):
            raise ValueError("definition proposal file requires a definitions object")
        proposals = {
            str(token): str(definition).strip()
            for token, definition in raw_proposals.items()
            if str(definition).strip()
        }
    result = build_dictionary_review_data(primary, secondary, proposals)
    review_tokens = {row["token"] for row in result["review"]}
    unknown_proposals = sorted(set(proposals) - review_tokens)
    if unknown_proposals:
        raise ValueError(
            "definition proposals target tokens outside the review queue: "
            + ", ".join(unknown_proposals[:20])
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(result["review"])
    return result
