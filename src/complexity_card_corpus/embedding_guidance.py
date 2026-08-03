from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from .semantic_audit import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    SentenceEmbedder,
    _load_embedder,
)


def _priority_actions(prompt_ratio: float, response_ratio: float) -> list[str]:
    actions = []
    if prompt_ratio >= 0.05:
        actions.append("expand_situation_goal_and_user_opening_decks")
    if response_ratio >= 0.05:
        actions.append("expand_answer_reasoning_and_conclusion_decks")
    if max(prompt_ratio, response_ratio) >= 0.10:
        actions.append("overgenerate_then_semantically_select")
    return actions or ["maintain_current_surface_decks"]


def build_embedding_guidance_data(
    dictionary: dict[str, Any],
    semantic_audit: dict[str, Any],
    *,
    embedder: SentenceEmbedder,
    model_name: str,
    model_revision: str,
    batch_size: int = 128,
    workers: int = 8,
    random_state: int = 42,
    alternatives_per_token: int = 5,
) -> dict[str, Any]:
    words = dictionary.get("words")
    if not isinstance(words, dict) or not words:
        raise ValueError("embedding guidance requires a non-empty dictionary")
    if alternatives_per_token < 1:
        raise ValueError("alternatives_per_token must be positive")
    prompt_rates = semantic_audit["views"]["prompts"][
        "family_semantic_duplicate_ratio"
    ]
    response_rates = semantic_audit["views"]["responses"][
        "family_semantic_duplicate_ratio"
    ]

    records = []
    for token, payload in sorted(words.items()):
        selected = payload.get("selected") or {}
        family = str(selected.get("family", ""))
        domain = str(selected.get("domain", ""))
        definition = str(payload.get("short_definition", "")).strip()
        if not family or not definition:
            continue
        records.append(
            {
                "token": token,
                "family": family,
                "domain": domain,
                "definition": definition,
                "classification_status": str(
                    selected.get("classification_status", "review_required")
                ),
                "statistical_rank": int(selected.get("rank", 0) or 0),
                "surface": f"{token}. {definition}",
                "masked_neighbors": [
                    str(item.get("token", ""))
                    for item in (payload.get("masked_context") or {}).get(
                        "neighbors", []
                    )[:5]
                    if str(item.get("token", "")).strip()
                ],
            }
        )
    embeddings = np.asarray(
        embedder.encode(
            [record["surface"] for record in records],
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    if embeddings.ndim != 2 or len(embeddings) != len(records):
        raise ValueError("embedder returned an invalid vocabulary matrix")
    token_embeddings = np.asarray(
        embedder.encode(
            [record["token"] for record in records],
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    definition_embeddings = np.asarray(
        embedder.encode(
            [record["definition"] for record in records],
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    token_definition_alignment = np.sum(
        token_embeddings * definition_embeddings, axis=1
    )
    alignment_median = float(np.median(token_definition_alignment))
    alignment_mad = float(
        np.median(np.abs(token_definition_alignment - alignment_median))
    )
    alignment_scale = max(1e-8, 1.4826 * alignment_mad)
    alignment_outlier_threshold = alignment_median - 3.0 * alignment_scale

    indices_by_family: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        indices_by_family[record["family"]].append(index)

    alternatives: dict[str, list[dict[str, Any]]] = {}
    semantic_decks: dict[str, list[dict[str, Any]]] = {}
    for family, indices in sorted(indices_by_family.items()):
        family_embeddings = embeddings[indices]
        neighbor_count = min(len(indices), alternatives_per_token + 1)
        if neighbor_count > 1:
            neighbors = NearestNeighbors(
                n_neighbors=neighbor_count, metric="cosine", n_jobs=workers
            ).fit(family_embeddings)
            distances, neighbor_indices = neighbors.kneighbors(family_embeddings)
            for local_index, global_index in enumerate(indices):
                candidates = []
                for distance, candidate_local_index in zip(
                    distances[local_index], neighbor_indices[local_index]
                ):
                    candidate_global_index = indices[int(candidate_local_index)]
                    if candidate_global_index == global_index:
                        continue
                    candidate = records[candidate_global_index]
                    candidates.append(
                        {
                            "token": candidate["token"],
                            "domain": candidate["domain"],
                            "definition": candidate["definition"],
                            "cosine_similarity": float(1.0 - distance),
                        }
                    )
                    if len(candidates) == alternatives_per_token:
                        break
                alternatives[records[global_index]["token"]] = candidates
        else:
            alternatives[records[indices[0]]["token"]] = []

        cluster_count = min(
            len(indices), max(1, min(16, math.ceil(math.sqrt(len(indices)) / 2)))
        )
        if cluster_count == 1:
            labels = np.zeros(len(indices), dtype=np.int64)
        else:
            labels = MiniBatchKMeans(
                n_clusters=cluster_count,
                batch_size=min(1024, max(64, len(indices))),
                n_init="auto",
                random_state=random_state,
            ).fit_predict(family_embeddings)
        family_decks = []
        for cluster in range(cluster_count):
            cluster_tokens = sorted(
                records[indices[local_index]]["token"]
                for local_index, label in enumerate(labels)
                if int(label) == cluster
            )
            family_decks.append(
                {
                    "deck_id": f"{family}:semantic:{cluster:02d}",
                    "tokens": cluster_tokens,
                    "size": len(cluster_tokens),
                }
            )
        semantic_decks[family] = family_decks

    family_centroids = {}
    for family, indices in indices_by_family.items():
        centroid = embeddings[indices].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        family_centroids[family] = centroid / norm if norm > 0 else centroid
    ordered_families = sorted(family_centroids)
    centroid_matrix = np.asarray([family_centroids[family] for family in ordered_families])
    family_ranks = []
    selected_family_similarity = []
    neighbor_overlap = []
    review_rows = []
    for index, record in enumerate(records):
        similarities = embeddings[index] @ centroid_matrix.T
        selected_index = ordered_families.index(record["family"])
        rank = int(np.count_nonzero(similarities > similarities[selected_index]) + 1)
        family_ranks.append(rank)
        selected_similarity = float(similarities[selected_index])
        selected_family_similarity.append(selected_similarity)
        internal_neighbors = set(record["masked_neighbors"])
        semantic_neighbors = {
            item["token"] for item in alternatives.get(record["token"], [])
        }
        overlap = (
            len(internal_neighbors & semantic_neighbors) / len(internal_neighbors)
            if internal_neighbors
            else None
        )
        if overlap is not None:
            neighbor_overlap.append(overlap)
        review_rows.append(
            {
                "token": record["token"],
                "family": record["family"],
                "domain": record["domain"],
                "definition": record["definition"],
                "classification_status": record["classification_status"],
                "statistical_rank": record["statistical_rank"],
                "token_definition_cosine": float(token_definition_alignment[index]),
                "token_definition_robust_z": float(
                    (token_definition_alignment[index] - alignment_median)
                    / alignment_scale
                ),
                "token_definition_outlier": bool(
                    token_definition_alignment[index]
                    < alignment_outlier_threshold
                ),
                "selected_family_cosine": selected_similarity,
                "selected_family_rank": rank,
                "masked_to_semantic_neighbor_overlap": overlap,
            }
        )
    alignment_p05 = float(np.quantile(token_definition_alignment, 0.05))
    review_candidates = []
    for row in review_rows:
        undefined = (
            not row["definition"].strip()
            or "insufficient masked neighbours" in row["definition"].lower()
        )
        reasons = []
        if undefined:
            reasons.append("definition_missing_or_unsupported")
        if row["classification_status"] == "review_required":
            reasons.append("statistical_assignment_review_required")
        if row["token_definition_outlier"]:
            reasons.append("token_definition_alignment_robust_outlier")
        if row["selected_family_rank"] > 5:
            reasons.append("embedding_family_rank_above_5")
        if not reasons:
            continue
        review_candidates.append(
            {
                **row,
                "review_status": "undefined" if undefined else "uncertain",
                "review_reasons": reasons,
            }
        )
    review_queue = sorted(
        review_candidates,
        key=lambda item: (
            item["review_status"] != "undefined",
            -len(item["review_reasons"]),
            item["token_definition_cosine"],
            -item["selected_family_rank"],
            item["selected_family_cosine"],
            item["token"],
        ),
    )

    priorities = {}
    for family in sorted(set(prompt_rates) | set(response_rates)):
        prompt_ratio = float(prompt_rates.get(family, 0.0))
        response_ratio = float(response_rates.get(family, 0.0))
        priorities[family] = {
            "prompt_semantic_duplicate_ratio": prompt_ratio,
            "response_semantic_duplicate_ratio": response_ratio,
            "priority_score": max(prompt_ratio, response_ratio),
            "recommended_actions": _priority_actions(prompt_ratio, response_ratio),
            "semantic_decks": len(semantic_decks.get(family, [])),
            "vocabulary_cards": len(indices_by_family.get(family, [])),
        }

    return {
        "format": "semantic-surface-guidance-v1",
        "model": {
            "name": model_name,
            "revision": model_revision,
            "license": "Apache-2.0",
            "embedding_dimensions": int(embeddings.shape[1]),
        },
        "dictionary": {
            "format": dictionary.get("format"),
            "cards": len(records),
            "definition_policy": dictionary.get("definition_policy"),
        },
        "policy": {
            "automatic_synonym_replacement": False,
            "semantic_alternatives_require_grammar_and_compatibility_checks": True,
            "source_text_retained": False,
            "purpose": (
                "rank original vocabulary cards and prioritize surface-deck "
                "expansion without generating training prose"
            ),
        },
        "dictionary_coherence_audit": {
            "interpretation": (
                "Relative embedding-coherence diagnostics for human review; "
                "not proof that a statistical gloss is a true definition."
            ),
            "token_definition_alignment": {
                "mean_cosine": float(np.mean(token_definition_alignment)),
                "median_cosine": float(np.median(token_definition_alignment)),
                "p05_cosine": alignment_p05,
                "below_p05_count": int(
                    np.count_nonzero(token_definition_alignment < alignment_p05)
                ),
                "mad": alignment_mad,
                "robust_outlier_threshold": alignment_outlier_threshold,
                "robust_outlier_count": int(
                    np.count_nonzero(
                        token_definition_alignment < alignment_outlier_threshold
                    )
                ),
            },
            "selected_family_alignment": {
                "mean_cosine": float(np.mean(selected_family_similarity)),
                "top_1_rate": float(np.mean(np.asarray(family_ranks) <= 1)),
                "top_3_rate": float(np.mean(np.asarray(family_ranks) <= 3)),
                "top_5_rate": float(np.mean(np.asarray(family_ranks) <= 5)),
            },
            "masked_to_semantic_neighbor_overlap": {
                "comparable_cards": len(neighbor_overlap),
                "mean_recall_at_5": (
                    float(np.mean(neighbor_overlap)) if neighbor_overlap else None
                ),
                "warning": (
                    "Masked co-occurrence and sentence-embedding proximity "
                    "measure different relations; low overlap is a review signal."
                ),
            },
            "review_queue": review_queue,
        },
        "family_priorities": priorities,
        "semantic_decks": semantic_decks,
        "token_alternatives": alternatives,
    }


def build_embedding_guidance(
    dictionary_path: Path,
    semantic_audit_path: Path,
    output_path: Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    model_revision: str = DEFAULT_EMBEDDING_REVISION,
    device: str | None = None,
    batch_size: int = 128,
    workers: int = 8,
    alternatives_per_token: int = 5,
) -> dict[str, Any]:
    dictionary = json.loads(dictionary_path.read_text())
    semantic_audit = json.loads(semantic_audit_path.read_text())
    embedder = _load_embedder(model_name, model_revision, device)
    guidance = build_embedding_guidance_data(
        dictionary,
        semantic_audit,
        embedder=embedder,
        model_name=model_name,
        model_revision=model_revision,
        batch_size=batch_size,
        workers=workers,
        alternatives_per_token=alternatives_per_token,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(guidance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return guidance
