from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .placement_schema import MASKED_DICTIONARY_VERSION


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
                counter[neighbor] for counter in masked_neighbors[token].values()
            )
            neighbor_occurrences = max(1, sum(occurrences.get(neighbor, {}).values()))
            source_support = sum(
                counter[neighbor] > 0 for counter in masked_neighbors[token].values()
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
                "selected generation" if selected_usage else "alternative statistical"
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
            usage["domain"].replace("_", " ") for usage in statistical_usages[1:3]
        ]
        short_definition = (
            f"A term statistically associated with {', '.join(gloss_terms)}"
            if gloss_terms
            else "A term with insufficient masked neighbours"
        )
        short_definition += f". Selected corpus use: {domain_label}"
        if alternate_domains:
            short_definition += (
                f"; alternative contexts: {', '.join(alternate_domains)}"
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
            "source_occurrences": dict(sorted(occurrences.get(token, {}).items())),
            "cell_score_vector": [round(combined[cell], 8) for cell in cells],
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
        metadata = {key: value for key, value in dictionary.items() if key != "words"}
        for key, value in sorted(metadata.items()):
            serialized = json.dumps(value, indent=2, sort_keys=True)
            serialized = serialized.replace("\n", "\n  ")
            stream.write(f"  {json.dumps(key)}: {serialized},\n")
        stream.write('  "words": {\n')
        word_items = sorted(dictionary["words"].items())
        for index, (word, entry) in enumerate(word_items):
            suffix = "," if index + 1 < len(word_items) else ""
            serialized = json.dumps(entry, sort_keys=True, separators=(",", ":"))
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
