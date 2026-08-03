from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

def apply_definition_overlay_data(
    dictionary: dict[str, Any],
    definitions: dict[str, str],
    *,
    consensus_by_token: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Apply operator-accepted original definitions while preserving evidence."""

    result = copy.deepcopy(dictionary)
    words = result.get("words")
    if not isinstance(words, dict):
        raise ValueError("definition overlay requires a dictionary words object")
    cleaned = {
        str(token): str(definition).strip()
        for token, definition in definitions.items()
        if str(definition).strip()
    }
    unknown = sorted(set(cleaned) - set(words))
    if unknown:
        raise ValueError(
            "accepted definitions contain unknown tokens: " + ", ".join(unknown[:20])
        )
    consensus = consensus_by_token or {}
    for token, definition in cleaned.items():
        entry = words[token]
        previous = str(entry.get("short_definition", "")).strip()
        entry["statistical_gloss"] = str(
            entry.get("statistical_gloss", previous)
        )
        entry["short_definition"] = definition
        entry["definition_kind"] = "operator_accepted_original_definition"
        entry["definition_review"] = {
            "decision": "accepted",
            "embedding_consensus": consensus.get(token, "not_recorded"),
            "human_acceptance_required": True,
            "accepted_by": "operator",
        }
    result["definition_policy"] = (
        "Entries with definition_kind=operator_accepted_original_definition use "
        "original definitions explicitly accepted by the operator; their prior "
        "masked-context text is retained as statistical_gloss. All other short "
        "definitions remain statistical glosses and are not authoritative lexical "
        "senses."
    )
    audit = result.setdefault("audit", {})
    audit["operator_accepted_definitions"] = len(cleaned)
    audit["statistical_glosses_preserved_for_accepted_definitions"] = len(cleaned)
    return result


def load_definition_proposals(path: Path) -> dict[str, str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    raw = document.get("definitions")
    if not isinstance(raw, dict):
        raise ValueError("definition proposal file requires a definitions object")
    return {
        str(token): str(definition).strip()
        for token, definition in raw.items()
        if str(definition).strip()
    }


def apply_definition_overlay_to_placement(
    placement_path: Path,
    output_path: Path,
    definitions: dict[str, str],
    *,
    consensus_by_token: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Synchronize accepted definitions into vocabulary placement metadata."""

    with placement_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"token", "short_definition"}
    if not required.issubset(fieldnames):
        raise ValueError("vocabulary placement is missing definition fields")
    by_token = {str(row["token"]): row for row in rows}
    unknown = sorted(set(definitions) - set(by_token))
    if unknown:
        raise ValueError(
            "accepted definitions are missing from vocabulary placement: "
            + ", ".join(unknown[:20])
        )
    for extra in (
        "statistical_gloss",
        "definition_kind",
        "definition_review_decision",
        "definition_embedding_consensus",
    ):
        if extra not in fieldnames:
            fieldnames.append(extra)
    consensus = consensus_by_token or {}
    for token, definition in definitions.items():
        row = by_token[token]
        previous = str(row.get("short_definition", "")).strip()
        row["statistical_gloss"] = str(
            row.get("statistical_gloss", "") or previous
        )
        row["short_definition"] = str(definition).strip()
        row["definition_kind"] = "operator_accepted_original_definition"
        row["definition_review_decision"] = "accepted"
        row["definition_embedding_consensus"] = consensus.get(
            token, "not_recorded"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "placement": str(output_path.resolve()),
        "placement_rows": len(rows),
        "placement_definitions_updated": len(definitions),
    }


def accept_definition_proposals(
    dictionary_path: Path,
    proposals_path: Path,
    review_csv_path: Path,
    output_dictionary_path: Path,
    output_review_csv_path: Path,
    *,
    accept_all: bool,
    placement_path: Path | None = None,
    output_placement_path: Path | None = None,
) -> dict[str, Any]:
    from .vocabulary.dictionary import _write_masked_dictionary

    if not accept_all:
        raise ValueError("acceptance requires explicit accept_all=True")
    dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
    definitions = load_definition_proposals(proposals_path)
    with review_csv_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames or "token" not in fieldnames or "reviewer_decision" not in fieldnames:
        raise ValueError("definition review CSV is missing acceptance fields")
    by_token = {row["token"]: row for row in rows}
    missing = sorted(set(definitions) - set(by_token))
    if missing:
        raise ValueError(
            "definition proposals are missing from the review CSV: "
            + ", ".join(missing[:20])
        )
    for token in definitions:
        by_token[token]["reviewer_decision"] = "accepted"
        if not by_token[token].get("reviewer_notes", "").strip():
            by_token[token]["reviewer_notes"] = "Accepted by operator."
    consensus = {
        token: by_token[token].get("consensus", "not_recorded")
        for token in definitions
    }
    result = apply_definition_overlay_data(
        dictionary,
        definitions,
        consensus_by_token=consensus,
    )
    output_dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    output_review_csv_path.parent.mkdir(parents=True, exist_ok=True)
    _write_masked_dictionary(output_dictionary_path, result)
    with output_review_csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    placement_result: dict[str, Any] = {}
    if (placement_path is None) != (output_placement_path is None):
        raise ValueError(
            "placement_path and output_placement_path must be provided together"
        )
    if placement_path is not None and output_placement_path is not None:
        placement_result = apply_definition_overlay_to_placement(
            placement_path,
            output_placement_path,
            definitions,
            consensus_by_token=consensus,
        )
    return {
        "accepted_definitions": len(definitions),
        "dictionary": str(output_dictionary_path.resolve()),
        "review": str(output_review_csv_path.resolve()),
        **placement_result,
    }
