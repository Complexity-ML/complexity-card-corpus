from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..build import file_sha256
from ..definition_acceptance import apply_definition_overlay_data, load_definition_proposals
from .dictionary import _masked_dictionary, _role_counts, _write_masked_dictionary
from .placement_context import _context_counts
from .placement_schema import (
    ROLE_PRIORS,
    VOCABULARY_PLACEMENT_VERSION,
    _candidate_rows,
    _scenario_cells,
)


def _cell_scores(
    token: str,
    cells: list[tuple[str, str]],
    context: dict[str, dict[str, Counter[tuple[str, str]]]],
    roles: dict[str, Counter[str]],
    occurrences: dict[str, Counter[str]],
) -> tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str], int],
    dict[tuple[str, str], float],
]:
    context_scores: dict[tuple[str, str], float] = {}
    context_sources: dict[tuple[str, str], int] = {}
    role_scores: dict[tuple[str, str], float] = {}
    total_roles = sum(roles.get(token, {}).values()) or 1
    for cell in cells:
        family, _ = cell
        normalized = 0.0
        supported_sources = 0
        for source, source_occurrences in occurrences.get(token, {}).items():
            hits = context.get(token, {}).get(source, {}).get(cell, 0)
            if hits:
                supported_sources += 1
                normalized += hits / source_occurrences
        context_scores[cell] = normalized
        context_sources[cell] = supported_sources
        role_scores[cell] = sum(
            (count / total_roles) * ROLE_PRIORS.get(role, {}).get(family, 0.0)
            for role, count in roles.get(token, {}).items()
        )
    combined = {
        cell: (
            4.0 * context_sources[cell]
            + math.log1p(context_scores[cell] * 1_000)
            + role_scores[cell]
        )
        for cell in cells
    }
    return combined, context_sources, role_scores


def build_vocabulary_placement(
    vocabulary_review_path: Path,
    lexicon_path: Path,
    registry_path: Path,
    raw_root: Path,
    scenarios_path: Path,
    output_dir: Path,
    *,
    window_tokens: int = 16,
    accepted_definitions_path: Path | None = None,
) -> dict[str, Any]:
    """Place every mined vocabulary gap into a compatible scenario family.

    Placement uses only aggregate token, role, and local anchor co-occurrence
    counts. It does not retain or reproduce source sentences.
    """
    if window_tokens < 2:
        raise ValueError("window_tokens must be at least 2")
    candidate_rows = _candidate_rows(vocabulary_review_path)
    candidate_tokens = set(candidate_rows)
    scenario_counts, cell_anchors = _scenario_cells(scenarios_path)
    cells = sorted(scenario_counts)
    context, documents, masked_neighbors = _context_counts(
        registry_path,
        raw_root,
        candidate_tokens,
        cell_anchors,
        window_tokens=window_tokens,
    )
    roles, occurrences = _role_counts(lexicon_path, candidate_tokens)

    scored = {
        token: _cell_scores(token, cells, context, roles, occurrences)
        for token in candidate_tokens
    }

    def placement_priority(token: str) -> tuple[Any, ...]:
        combined, context_sources, _ = scored[token]
        ordered_scores = sorted(combined.values(), reverse=True)
        margin = ordered_scores[0] - ordered_scores[1]
        return (
            -max(context_sources.values()),
            -margin,
            -ordered_scores[0],
            hashlib.sha256(f"placement:{token}".encode()).digest(),
        )

    ordered = sorted(candidate_tokens, key=placement_priority)
    remaining = dict(scenario_counts)
    placements: list[dict[str, Any]] = []
    for token in ordered:
        combined, context_sources, role_scores = scored[token]
        available = [cell for cell, count in remaining.items() if count > 0]
        best_score = max(combined[cell] for cell in available)
        tied = [cell for cell in available if combined[cell] == best_score]
        cell = min(
            tied,
            key=lambda value: hashlib.sha256(
                f"placement-tie:{token}:{value}".encode()
            ).digest(),
        )
        remaining[cell] -= 1
        family, domain = cell
        source_support = context_sources[cell]
        role_score = role_scores[cell]
        method = (
            "cross_source_context"
            if source_support >= 2
            else "single_source_context"
            if source_support == 1
            else "role_prior"
            if role_score > 0
            else "balanced_hash_tiebreak"
        )
        source_row = candidate_rows[token]
        placements.append(
            {
                "token": token,
                "family": family,
                "domain": domain,
                "assignment_method": method,
                "contextual_sources": source_support,
                "context_score": round(combined[cell], 8),
                "role_prior_score": round(role_score, 8),
                "source_count": int(source_row["source_count"]),
                "total_occurrences": int(source_row["total_occurrences"]),
                "surface_policy": "grounded_quoted_term",
            }
        )

    placements.sort(key=lambda row: (row["family"], row["domain"], row["token"]))
    family_counts = Counter(row["family"] for row in placements)
    cell_counts = Counter(f"{row['family']}::{row['domain']}" for row in placements)
    method_counts = Counter(row["assignment_method"] for row in placements)
    dictionary = _masked_dictionary(
        placements,
        scored,
        masked_neighbors,
        roles,
        occurrences,
        cells,
    )
    if accepted_definitions_path is not None:
        dictionary = apply_definition_overlay_data(
            dictionary,
            load_definition_proposals(accepted_definitions_path),
        )
    enriched_placements: list[dict[str, Any]] = []
    for placement in placements:
        entry = dictionary["words"][str(placement["token"])]
        enriched_placements.append(
            {
                "token": placement["token"],
                "short_definition": entry["short_definition"],
                "statistical_gloss": entry.get("statistical_gloss", ""),
                "definition_kind": entry.get(
                    "definition_kind", "masked_context_statistical_gloss"
                ),
                "definition_review_decision": entry.get(
                    "definition_review", {}
                ).get("decision", "not_reviewed"),
                "definition_embedding_consensus": entry.get(
                    "definition_review", {}
                ).get("embedding_consensus", "not_recorded"),
                "classification_status": entry["selected"]["classification_status"],
                "selected_rank": entry["selected"]["rank"],
                "statistical_usages_json": json.dumps(
                    entry["statistical_usages"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                **{key: value for key, value in placement.items() if key != "token"},
            }
        )
    placements = enriched_placements
    audit = {
        "version": VOCABULARY_PLACEMENT_VERSION,
        "candidate_tokens": len(candidate_tokens),
        "placed_tokens": len(placements),
        "coverage_ratio": 1.0,
        "window_tokens": window_tokens,
        "cell_capacity": {
            f"{family}::{domain}": count
            for (family, domain), count in sorted(scenario_counts.items())
        },
        "unused_cell_capacity": sum(scenario_counts.values()) - len(placements),
        "family_placements": dict(sorted(family_counts.items())),
        "cell_placements": dict(sorted(cell_counts.items())),
        "assignment_methods": dict(sorted(method_counts.items())),
        "source_documents_scanned": dict(sorted(documents.items())),
        "source_text_retained": False,
        "source_phrases_retained": False,
        "semantic_claim_policy": (
            "tokens are presented only as source terms; the generated response "
            "must not invent a definition"
        ),
        "automatic_definition_generation": False,
        "masked_dictionary": dictionary["audit"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    placement_path = output_dir / "vocabulary_placement.csv"
    with placement_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(placements[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(placements)
    audit_path = output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    dictionary_path = output_dir / "vocabulary_dictionary.json"
    _write_masked_dictionary(dictionary_path, dictionary)
    manifest = {
        "format": VOCABULARY_PLACEMENT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "vocabulary_review": file_sha256(vocabulary_review_path),
            "lexicon": file_sha256(lexicon_path),
            "registry": file_sha256(registry_path),
            "scenarios": file_sha256(scenarios_path),
            **(
                {"accepted_definitions": file_sha256(accepted_definitions_path)}
                if accepted_definitions_path is not None
                else {}
            ),
        },
        "files": {
            "vocabulary_placement.csv": file_sha256(placement_path),
            "vocabulary_dictionary.json": file_sha256(dictionary_path),
            "audit.json": file_sha256(audit_path),
        },
        "audit": audit,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
