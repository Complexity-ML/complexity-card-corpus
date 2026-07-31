from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")


def _lookup_forms(token: str) -> list[str]:
    forms = [token]
    if token.endswith("'s"):
        forms.append(token[:-2])
    if token.endswith("ies") and len(token) > 4:
        forms.append(f"{token[:-3]}y")
    if token.endswith("es") and len(token) > 4:
        forms.extend((token[:-2], token[:-1]))
    elif token.endswith("s") and len(token) > 3:
        forms.append(token[:-1])
    if token.endswith("ied") and len(token) > 4:
        forms.append(f"{token[:-3]}y")
    if token.endswith("ed") and len(token) > 4:
        forms.extend((token[:-2], token[:-1]))
    if token.endswith("ing") and len(token) > 5:
        forms.extend((token[:-3], f"{token[:-3]}e"))
    if token.endswith("ly") and len(token) > 4:
        forms.append(token[:-2])
    return list(dict.fromkeys(forms))


def _synsets(wordnet: Any, token: str, *, maximum: int = 8) -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for form in _lookup_forms(token):
        for synset in wordnet.synsets(form):
            if synset.id in seen:
                continue
            seen.add(synset.id)
            result.append((form, synset))
            if len(result) == maximum:
                return result
    return result


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(len(vectors[0]))
    ]


def audit_vocabulary_with_wordnet(
    dictionary_path: Path,
    output_path: Path,
    *,
    lexicon: str = "oewn:2025+",
) -> dict[str, Any]:
    """Compare masked-context placement with an external WordNet proxy.

    Open English WordNet text is read locally for the comparison but no
    definition or source phrase is retained in the output.
    """
    try:
        import wn
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Install the wordnet-audit extra and download Open English WordNet"
        ) from exc

    dictionary = json.loads(dictionary_path.read_text())
    words: dict[str, dict[str, Any]] = dictionary["words"]
    axis = dictionary["matrix"]["axis"]
    wordnet = wn.Wordnet(lexicon)
    status_counts: Counter[str] = Counter()
    rank_counts: Counter[int] = Counter()
    family_rank_counts: Counter[int] = Counter()
    entries: dict[str, Any] = {}

    for token, entry in words.items():
        senses = _synsets(wordnet, token)
        if not senses:
            status_counts["wordnet_unresolved"] += 1
            entries[token] = {"status": "wordnet_unresolved"}
            continue
        sense_rows: list[dict[str, Any]] = []
        for lookup_form, synset in senses:
            definition_terms = {
                match.group(0).lower()
                for match in _WORD.finditer(synset.definition())
            }
            lemma_terms = {
                lemma.replace("_", " ").lower()
                for word in synset.words()
                for lemma in [word.lemma()]
                if " " not in lemma
            }
            related = sorted(
                ((definition_terms | lemma_terms) & set(words)) - {token}
            )
            vectors = [words[term]["cell_score_vector"] for term in related]
            if not vectors:
                sense_rows.append(
                    {
                        "lookup_form": lookup_form,
                        "synset_id": synset.id,
                        "related_dictionary_terms": related,
                        "status": "insufficient_vector_context",
                    }
                )
                continue
            semantic_vector = _mean_vector(vectors)
            ranked = sorted(
                range(len(semantic_vector)),
                key=lambda index: (-semantic_vector[index], index),
            )
            selected_index = int(entry["selected"]["cell_index"])
            rank = ranked.index(selected_index) + 1
            selected_family = str(entry["selected"]["family"])
            family_scores = {
                family: max(
                    semantic_vector[index]
                    for index, cell in enumerate(axis)
                    if cell["family"] == family
                )
                for family in {str(cell["family"]) for cell in axis}
            }
            ranked_families = sorted(
                family_scores,
                key=lambda family: (-family_scores[family], family),
            )
            family_rank = ranked_families.index(selected_family) + 1
            sense_rows.append(
                {
                    "lookup_form": lookup_form,
                    "synset_id": synset.id,
                    "related_dictionary_terms": related,
                    "status": "comparable",
                    "selected_rank": rank,
                    "selected_family_rank": family_rank,
                    "wordnet_proxy_top_cells": [
                        {
                            "cell_index": index,
                            "family": axis[index]["family"],
                            "domain": axis[index]["domain"],
                            "score": round(semantic_vector[index], 8),
                        }
                        for index in ranked[:10]
                    ],
                }
            )

        comparable_senses = [
            row for row in sense_rows if row["status"] == "comparable"
        ]
        if not comparable_senses:
            status_counts["insufficient_vector_context"] += 1
            entries[token] = {
                "status": "insufficient_vector_context",
                "senses": sense_rows,
            }
            continue

        rank = min(int(row["selected_rank"]) for row in comparable_senses)
        rank_counts[rank] += 1
        family_rank = min(
            int(row["selected_family_rank"]) for row in comparable_senses
        )
        family_rank_counts[family_rank] += 1
        status = (
            "top_1"
            if rank == 1
            else "top_3"
            if rank <= 3
            else "top_10"
            if rank <= 10
            else "outside_top_10"
        )
        status_counts[status] += 1
        entries[token] = {
            "status": status,
            "selected_rank": rank,
            "selected_family_rank": family_rank,
            "selected_cell_index": int(entry["selected"]["cell_index"]),
            "senses": sense_rows,
        }

    total = len(words)
    resolved = total - status_counts["wordnet_unresolved"]
    comparable = sum(rank_counts.values())

    def percentage(count: int, denominator: int) -> float | None:
        return round(100 * count / denominator, 4) if denominator else None

    top_1 = rank_counts[1]
    top_3 = sum(count for rank, count in rank_counts.items() if rank <= 3)
    top_10 = sum(count for rank, count in rank_counts.items() if rank <= 10)
    family_top_1 = family_rank_counts[1]
    family_top_3 = sum(
        count for rank, count in family_rank_counts.items() if rank <= 3
    )
    family_top_5 = sum(
        count for rank, count in family_rank_counts.items() if rank <= 5
    )
    result = {
        "format": "wordnet-placement-audit-v1",
        "lexicon": lexicon,
        "interpretation": (
            "This is an external semantic-proxy agreement test, not a ground-"
            "truth accuracy measurement. WordNet definitions are not retained."
        ),
        "summary": {
            "dictionary_words": total,
            "wordnet_resolved": resolved,
            "wordnet_coverage_percent": percentage(resolved, total),
            "vector_comparable": comparable,
            "vector_comparable_percent": percentage(comparable, total),
            "selected_cell_top_1_percent": percentage(top_1, comparable),
            "selected_cell_top_3_percent": percentage(top_3, comparable),
            "selected_cell_top_10_percent": percentage(top_10, comparable),
            "selected_family_top_1_percent": percentage(
                family_top_1, comparable
            ),
            "selected_family_top_3_percent": percentage(
                family_top_3, comparable
            ),
            "selected_family_top_5_percent": percentage(
                family_top_5, comparable
            ),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "entries": dict(sorted(entries.items())),
    }
    if not math.isclose(
        sum(result["summary"]["status_counts"].values()), total
    ):
        raise AssertionError("WordNet audit status coverage is incomplete")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
