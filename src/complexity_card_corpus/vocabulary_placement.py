from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .build import file_sha256
from .lexical_mine import (
    _artifact_path,
    _source_role_documents,
    _words,
    load_lexical_registry,
)
from .vocabulary_gap import AUTHORING_STOPWORDS


VOCABULARY_PLACEMENT_VERSION = "statistical-vocabulary-placement-v1"
MASKED_DICTIONARY_VERSION = "masked-context-dictionary-v1"

# These anchors describe the original Complexity families. External text is
# used only to count local co-occurrence with them; no source phrase survives.
FAMILY_ANCHORS: dict[str, frozenset[str]] = {
    "practical_action": frozenset(
        "action build change choose create install make organize prepare use".split()
    ),
    "explanation_learning": frozenset(
        "concept definition example explain learn lesson mechanism principle teach".split()
    ),
    "troubleshooting": frozenset(
        "bug diagnose error fail failure fix issue problem recover test".split()
    ),
    "writing_transformation": frozenset(
        "draft edit paragraph rewrite sentence style summarize text tone write".split()
    ),
    "planning_comparison": frozenset(
        "budget compare cost criteria decision option plan priority schedule tradeoff".split()
    ),
    "conversation_empathy": frozenset(
        "emotion empathy feel feeling friend listen relationship support understand".split()
    ),
    "safety_uncertainty": frozenset(
        "danger emergency harm health legal medical privacy risk safe safety".split()
    ),
    "grounded_qa": frozenset(
        "answer citation document evidence fact question quote record source verify".split()
    ),
    "summarization_synthesis": frozenset(
        "decision key main notes overview points report summary synthesize theme".split()
    ),
    "extraction_classification": frozenset(
        "category classify column extract field label record schema structure value".split()
    ),
    "reasoning_verification": frozenset(
        "calculate check equation logic premise proof reason result solve verify".split()
    ),
    "critique_revision": frozenset(
        "argument critique evidence improve issue revise revision weak weakness writing".split()
    ),
    "brainstorming_creativity": frozenset(
        "brainstorm combine creative design generate idea imagine novel option".split()
    ),
    "context_clarification": frozenset(
        "ambiguous clarify clarification context intent mean missing request scope".split()
    ),
}

ROLE_PRIORS: dict[str, dict[str, float]] = {
    "intent_term": {
        "practical_action": 1.0,
        "troubleshooting": 0.9,
        "planning_comparison": 0.8,
        "reasoning_verification": 0.8,
        "critique_revision": 0.7,
        "brainstorming_creativity": 0.8,
    },
    "state_term": {
        "troubleshooting": 1.0,
        "conversation_empathy": 1.0,
        "safety_uncertainty": 0.9,
        "context_clarification": 0.9,
    },
    "constraint_term": {
        "safety_uncertainty": 1.0,
        "grounded_qa": 0.9,
        "extraction_classification": 0.8,
        "planning_comparison": 0.7,
        "context_clarification": 0.8,
    },
    "outcome_term": {
        "summarization_synthesis": 1.0,
        "writing_transformation": 0.9,
        "explanation_learning": 0.8,
        "reasoning_verification": 0.8,
    },
    "transition": {
        "summarization_synthesis": 1.0,
        "planning_comparison": 0.8,
        "reasoning_verification": 0.8,
        "writing_transformation": 0.7,
    },
}


def _candidate_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["token"]: row for row in csv.DictReader(stream)}
    if not rows:
        raise ValueError("vocabulary review contains no candidates")
    return rows


def _scenario_cells(
    scenarios_path: Path,
) -> tuple[Counter[tuple[str, str]], dict[tuple[str, str], set[str]]]:
    rows = pq.read_table(scenarios_path).to_pylist()
    counts: Counter[tuple[str, str]] = Counter()
    raw_anchors: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        cell = (str(row["family"]), str(row["domain"]))
        counts[cell] += 1
        payload = json.loads(row["semantic_payload"])
        values = [
            row["trigger"],
            row["state"],
            row["constraint"],
            row["desired_outcome"],
            *[value for value in payload.values() if isinstance(value, str)],
        ]
        for value in values:
            raw_anchors[cell].update(
                token
                for token, _ in _words(str(value))
                if len(token) >= 4 and token not in AUTHORING_STOPWORDS
            )
        raw_anchors[cell].update(FAMILY_ANCHORS[cell[0]])

    cell_frequency: Counter[str] = Counter()
    for anchors in raw_anchors.values():
        cell_frequency.update(anchors.keys())
    maximum_cells = max(5, math.floor(len(counts) * 0.20))
    anchors = {
        cell: {
            token
            for token, occurrences in values.items()
            if occurrences >= 2 and cell_frequency[token] <= maximum_cells
        }
        for cell, values in raw_anchors.items()
    }
    if any(not values for values in anchors.values()):
        raise ValueError("a scenario family/domain cell has no usable anchors")
    return counts, anchors


def _context_counts(
    registry_path: Path,
    raw_root: Path,
    candidates: set[str],
    cell_anchors: dict[tuple[str, str], set[str]],
    *,
    window_tokens: int,
) -> tuple[
    dict[str, dict[str, Counter[tuple[str, str]]]],
    Counter[str],
    dict[str, dict[str, Counter[str]]],
]:
    registry = load_lexical_registry(registry_path)
    anchor_cells: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for cell, anchors in cell_anchors.items():
        for anchor in anchors:
            anchor_cells[anchor].add(cell)

    counts: dict[str, dict[str, Counter[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    masked_neighbors: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    documents = Counter()
    for source in registry["sources"]:
        source_id = str(source["dataset_id"])
        for artifact in source["artifacts"]:
            path = _artifact_path(raw_root, source, artifact)
            if not path.exists():
                raise FileNotFoundError(path)
            for _, text in _source_role_documents(path, source):
                documents[source_id] += 1
                words = [token for token, _ in _words(text)]
                anchors_at = [anchor_cells.get(word, set()) for word in words]
                candidate_positions = [
                    (index, word)
                    for index, word in enumerate(words)
                    if word in candidates
                ]
                for index, token in candidate_positions:
                    local: Counter[str] = Counter()
                    start = max(0, index - window_tokens)
                    stop = min(len(words), index + window_tokens + 1)
                    for families in anchors_at[start:stop]:
                        local.update(families)
                    counts[token][source_id].update(local)
                    masked_neighbors[token][source_id].update(
                        neighbor
                        for neighbor in words[start:stop]
                        if neighbor in candidates and neighbor != token
                    )
    return counts, documents, masked_neighbors


def _masked_dictionary(
    placements: list[dict[str, Any]],
    scored: dict[
        str,
        tuple[
            dict[tuple[str, str], float],
            dict[tuple[str, str], int],
            dict[tuple[str, str], float],
        ],
    ],
    masked_neighbors: dict[str, dict[str, Counter[str]]],
    roles: dict[str, Counter[str]],
    occurrences: dict[str, Counter[str]],
    cells: list[tuple[str, str]],
) -> dict[str, Any]:
    """Build a home-grown dictionary from masked-token statistics.

    The short definition is a corpus gloss, not a claimed lexical definition.
    Only individual normalized words and aggregate counts are retained.
    """
    cell_index = {cell: index for index, cell in enumerate(cells)}
    entries: dict[str, Any] = {}
    status_counts: Counter[str] = Counter()
    neighbor_coverage = 0
    for placement in placements:
        token = str(placement["token"])
        selected = (str(placement["family"]), str(placement["domain"]))
        combined, context_sources, role_scores = scored[token]
        ranked_cells = sorted(
            cells,
            key=lambda cell: (-combined[cell], cell),
        )
        selected_rank = ranked_cells.index(selected) + 1
        selected_sources = context_sources[selected]
        classification_status = (
            "statistically_supported"
            if selected_rank <= 3 and selected_sources >= 2
            else "statistically_plausible"
            if selected_rank <= 10 and selected_sources >= 1
            else "review_required"
        )
        status_counts[classification_status] += 1

        token_occurrences = max(1, sum(occurrences.get(token, {}).values()))
        neighbor_rows: list[dict[str, Any]] = []
        neighbor_tokens = {
            neighbor
            for counter in masked_neighbors.get(token, {}).values()
            for neighbor in counter
        }
        for neighbor in neighbor_tokens:
            cooccurrences = sum(
                counter[neighbor]
                for counter in masked_neighbors[token].values()
            )
            neighbor_occurrences = max(
                1, sum(occurrences.get(neighbor, {}).values())
            )
            source_support = sum(
                counter[neighbor] > 0
                for counter in masked_neighbors[token].values()
            )
            association = cooccurrences / math.sqrt(
                token_occurrences * neighbor_occurrences
            )
            neighbor_rows.append(
                {
                    "token": neighbor,
                    "cooccurrences": cooccurrences,
                    "source_support": source_support,
                    "association": round(association, 8),
                }
            )
        neighbor_rows.sort(
            key=lambda row: (
                -int(row["source_support"]),
                -float(row["association"]),
                -int(row["cooccurrences"]),
                str(row["token"]),
            )
        )
        neighbor_rows = neighbor_rows[:12]
        neighbor_coverage += bool(neighbor_rows)
        gloss_terms = [row["token"] for row in neighbor_rows[:4]]
        domain_label = selected[1].replace("_", " ")

        # A word can serve several purposes. Preserve the selected generation
        # cell, then add the strongest distinct-domain alternatives. This keeps
        # capacity-driven placement separate from statistical polysemy.
        statistical_usages: list[dict[str, Any]] = []
        seen_domains: set[str] = set()
        usage_cells = [selected, *[cell for cell in ranked_cells if cell != selected]]
        for cell in usage_cells:
            family, domain = cell
            if domain in seen_domains:
                continue
            seen_domains.add(domain)
            selected_usage = cell == selected
            usage_label = (
                "selected generation"
                if selected_usage
                else "alternative statistical"
            )
            statistical_usages.append(
                {
                    "family": family,
                    "domain": domain,
                    "cell_index": cell_index[cell],
                    "rank": ranked_cells.index(cell) + 1,
                    "score": round(combined[cell], 8),
                    "usage_kind": (
                        "selected_generation"
                        if selected_usage
                        else "alternative_statistical_context"
                    ),
                    "usage_gloss": (
                        f"The {usage_label} "
                        f"use places this term in {domain.replace('_', ' ')} "
                        f"contexts for {family.replace('_', ' ')}."
                    ),
                }
            )
            if len(statistical_usages) == 5:
                break

        alternate_domains = [
            usage["domain"].replace("_", " ")
            for usage in statistical_usages[1:3]
        ]
        short_definition = (
            f"A term statistically associated with {', '.join(gloss_terms)}"
            if gloss_terms
            else "A term with insufficient masked neighbours"
        )
        short_definition += f". Selected corpus use: {domain_label}"
        if alternate_domains:
            short_definition += (
                f"; alternative contexts: "
                f"{', '.join(alternate_domains)}"
            )
        short_definition += "."

        entries[token] = {
            "short_definition": short_definition,
            "definition_kind": "masked_context_statistical_gloss",
            "selected": {
                "family": selected[0],
                "domain": selected[1],
                "cell_index": cell_index[selected],
                "rank": selected_rank,
                "assignment_method": placement["assignment_method"],
                "classification_status": classification_status,
            },
            "masked_context": {
                "neighbors": neighbor_rows,
                "window_source_support": selected_sources,
            },
            "role_profile": dict(sorted(roles.get(token, {}).items())),
            "source_occurrences": dict(
                sorted(occurrences.get(token, {}).items())
            ),
            "cell_score_vector": [
                round(combined[cell], 8) for cell in cells
            ],
            "top_cells": [
                {
                    "cell_index": cell_index[cell],
                    "family": cell[0],
                    "domain": cell[1],
                    "score": round(combined[cell], 8),
                    "contextual_sources": context_sources[cell],
                    "role_prior_score": round(role_scores[cell], 8),
                }
                for cell in ranked_cells[:8]
            ],
            "statistical_usages": statistical_usages,
        }

    return {
        "format": MASKED_DICTIONARY_VERSION,
        "definition_policy": (
            "Definitions are statistical glosses derived from masked-token "
            "co-occurrence. They are not copied dictionary definitions and must "
            "not be interpreted as authoritative lexical senses."
        ),
        "matrix": {
            "axis": [
                {"index": index, "family": family, "domain": domain}
                for index, (family, domain) in enumerate(cells)
            ],
            "shape": [len(entries), len(cells)],
            "value": "aggregate masked-context family/domain score",
        },
        "audit": {
            "words": len(entries),
            "words_with_neighbors": neighbor_coverage,
            "classification_status": dict(sorted(status_counts.items())),
            "source_text_retained": False,
            "source_phrases_retained": False,
        },
        "words": dict(sorted(entries.items())),
    }


def _write_masked_dictionary(path: Path, dictionary: dict[str, Any]) -> None:
    """Write valid JSON with readable metadata and one compact line per word."""
    with path.open("w", encoding="utf-8") as stream:
        stream.write("{\n")
        metadata = {
            key: value for key, value in dictionary.items() if key != "words"
        }
        for key, value in sorted(metadata.items()):
            serialized = json.dumps(value, indent=2, sort_keys=True)
            serialized = serialized.replace("\n", "\n  ")
            stream.write(f"  {json.dumps(key)}: {serialized},\n")
        stream.write('  "words": {\n')
        word_items = sorted(dictionary["words"].items())
        for index, (word, entry) in enumerate(word_items):
            suffix = "," if index + 1 < len(word_items) else ""
            serialized = json.dumps(
                entry, sort_keys=True, separators=(",", ":")
            )
            stream.write(f"    {json.dumps(word)}: {serialized}{suffix}\n")
        stream.write("  }\n}\n")


def _role_counts(
    lexicon_path: Path, candidates: set[str]
) -> tuple[dict[str, Counter[str]], dict[str, Counter[str]]]:
    roles: dict[str, Counter[str]] = defaultdict(Counter)
    occurrences: dict[str, Counter[str]] = defaultdict(Counter)
    for batch in pq.ParquetFile(lexicon_path).iter_batches(batch_size=8_192):
        for row in batch.to_pylist():
            token = str(row["token"])
            if token not in candidates:
                continue
            count = int(row["occurrences"])
            roles[token][str(row["role"])] += count
            occurrences[token][str(row["source_dataset"])] += count
    return roles, occurrences


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

    placements.sort(
        key=lambda row: (row["family"], row["domain"], row["token"])
    )
    family_counts = Counter(row["family"] for row in placements)
    cell_counts = Counter(
        f"{row['family']}::{row['domain']}" for row in placements
    )
    method_counts = Counter(row["assignment_method"] for row in placements)
    dictionary = _masked_dictionary(
        placements,
        scored,
        masked_neighbors,
        roles,
        occurrences,
        cells,
    )
    enriched_placements: list[dict[str, Any]] = []
    for placement in placements:
        entry = dictionary["words"][str(placement["token"])]
        enriched_placements.append(
            {
                "token": placement["token"],
                "short_definition": entry["short_definition"],
                "classification_status": entry["selected"][
                    "classification_status"
                ],
                "selected_rank": entry["selected"]["rank"],
                "statistical_usages_json": json.dumps(
                    entry["statistical_usages"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                **{
                    key: value
                    for key, value in placement.items()
                    if key != "token"
                },
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
