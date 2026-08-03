from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from .quality_audit import _normalize, _stable_sample, resolve_quality_audit_policy


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_REVISION = "46605decb5369335a3847c9f41bb0b896c07dd1a"


class SentenceEmbedder(Protocol):
    max_seq_length: int

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray: ...


def _load_embedder(model_name: str, revision: str, device: str | None) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Embedding audit requires the optional semantic-audit dependencies. "
            "Install them with `pip install -e '.[semantic-audit]'`."
        ) from error
    return SentenceTransformer(model_name, revision=revision, device=device)


def _nearest_other(
    embeddings: np.ndarray, workers: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(embeddings) < 2:
        empty = np.zeros(len(embeddings), dtype=np.float32)
        return empty, np.zeros(len(embeddings), dtype=np.int64)
    neighbor_count = min(len(embeddings), 8)
    neighbors = NearestNeighbors(
        n_neighbors=neighbor_count, metric="cosine", n_jobs=workers
    )
    neighbors.fit(embeddings)
    distances, indices = neighbors.kneighbors(embeddings)
    similarities = np.empty(len(embeddings), dtype=np.float32)
    neighbor_indices = np.empty(len(embeddings), dtype=np.int64)
    for row_index, (row_distances, row_indices) in enumerate(
        zip(distances, indices)
    ):
        position = next(
            (
                candidate
                for candidate, neighbor_index in enumerate(row_indices)
                if int(neighbor_index) != row_index
            ),
            None,
        )
        if position is None:
            similarities[row_index] = 0.0
            neighbor_indices[row_index] = row_index
        else:
            similarities[row_index] = 1.0 - row_distances[position]
            neighbor_indices[row_index] = row_indices[position]
    return similarities, neighbor_indices


def _cross_split_similarity(
    embeddings: np.ndarray, splits: list[str], workers: int
) -> np.ndarray:
    train = np.asarray([index for index, split in enumerate(splits) if split == "train"])
    evaluation = np.asarray(
        [index for index, split in enumerate(splits) if split != "train"]
    )
    if len(train) == 0 or len(evaluation) == 0:
        return np.zeros(len(evaluation), dtype=np.float32)
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=workers)
    neighbors.fit(embeddings[train])
    distances, _ = neighbors.kneighbors(embeddings[evaluation])
    return 1.0 - distances[:, 0]


def _effective_rank(embeddings: np.ndarray) -> dict[str, float]:
    if len(embeddings) < 2:
        return {"entropy_rank": 1.0, "participation_ratio": 1.0}
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variance = np.square(singular_values, dtype=np.float64)
    total = float(variance.sum())
    if total <= 0:
        return {"entropy_rank": 1.0, "participation_ratio": 1.0}
    probabilities = variance / total
    nonzero = probabilities[probabilities > 0]
    return {
        "entropy_rank": float(np.exp(-(nonzero * np.log(nonzero)).sum())),
        "participation_ratio": float(1.0 / np.square(probabilities).sum()),
    }


def _mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    count = len(embeddings)
    if count < 2:
        return 1.0
    vector_sum = embeddings.sum(axis=0, dtype=np.float64)
    similarity_sum = float(np.dot(vector_sum, vector_sum) - count)
    return similarity_sum / (count * (count - 1))


def _family_dispersion(
    embeddings: np.ndarray, labels: list[str]
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    labels_array = np.asarray(labels)
    for label in sorted(set(labels)):
        family_embeddings = embeddings[labels_array == label]
        centroid = family_embeddings.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        centroid_similarity = (
            family_embeddings @ (centroid / norm) if norm > 0 else np.zeros(len(family_embeddings))
        )
        result[label] = {
            "rows": int(len(family_embeddings)),
            "mean_cosine_to_centroid": float(np.mean(centroid_similarity)),
            "semantic_dispersion": float(1.0 - np.mean(centroid_similarity)),
        }
    return result


def _embedding_view(
    embeddings: np.ndarray,
    *,
    labels: list[str],
    splits: list[str],
    example_ids: list[str],
    cluster_count: int,
    semantic_duplicate_threshold: float,
    random_state: int,
    workers: int,
) -> dict[str, Any]:
    nearest, nearest_indices = _nearest_other(embeddings, workers)
    cross_split = _cross_split_similarity(embeddings, splits, workers)
    clusterer = MiniBatchKMeans(
        n_clusters=cluster_count,
        batch_size=min(4096, max(256, len(embeddings))),
        n_init="auto",
        random_state=random_state,
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    cluster_sizes = Counter(int(label) for label in cluster_labels)
    duplicate_count = int(
        np.count_nonzero(nearest >= semantic_duplicate_threshold)
    )
    family_duplicate_counts = Counter(
        label
        for label, similarity in zip(labels, nearest)
        if similarity >= semantic_duplicate_threshold
    )
    family_counts = Counter(labels)
    top_indices = np.argsort(nearest)[::-1][:50]
    mean_pairwise = _mean_pairwise_cosine(embeddings)
    return {
        "semantic_neighbors": {
            "duplicate_threshold": semantic_duplicate_threshold,
            "duplicate_rows": duplicate_count,
            "duplicate_ratio": duplicate_count / len(embeddings),
            "mean_similarity": float(np.mean(nearest)),
            "p50_similarity": float(np.quantile(nearest, 0.50)),
            "p95_similarity": float(np.quantile(nearest, 0.95)),
            "maximum_similarity": float(np.max(nearest)),
        },
        "cross_split_neighbors": {
            "evaluation_rows": int(len(cross_split)),
            "mean_similarity": float(np.mean(cross_split)) if len(cross_split) else None,
            "p95_similarity": (
                float(np.quantile(cross_split, 0.95)) if len(cross_split) else None
            ),
            "maximum_similarity": float(np.max(cross_split)) if len(cross_split) else None,
        },
        "global_geometry": {
            "mean_pairwise_cosine": mean_pairwise,
            "semantic_dispersion": float(1.0 - mean_pairwise),
            **_effective_rank(embeddings),
        },
        "clusters": {
            "count": cluster_count,
            "occupied": len(cluster_sizes),
            "minimum_size": min(cluster_sizes.values()),
            "maximum_size": max(cluster_sizes.values()),
            "maximum_share": max(cluster_sizes.values()) / len(embeddings),
        },
        "family_dispersion": _family_dispersion(embeddings, labels),
        "family_semantic_duplicate_ratio": {
            label: family_duplicate_counts[label] / count
            for label, count in sorted(family_counts.items())
        },
        "nearest_pair_preview": [
            {
                "example_id": example_ids[int(index)],
                "neighbor_example_id": example_ids[int(nearest_indices[index])],
                "task": labels[int(index)],
                "neighbor_task": labels[int(nearest_indices[index])],
                "similarity": float(nearest[index]),
            }
            for index in top_indices
        ],
    }


def _sft_readiness_proxy(
    embeddings_by_view: dict[str, np.ndarray],
    views: dict[str, dict[str, Any]],
    *,
    cluster_count: int,
    random_state: int,
    corpus_rows: int,
) -> dict[str, Any]:
    prompt_clusters = MiniBatchKMeans(
        n_clusters=cluster_count,
        batch_size=min(4096, max(256, len(embeddings_by_view["prompts"]))),
        n_init="auto",
        random_state=random_state,
    ).fit_predict(embeddings_by_view["prompts"])
    response_embeddings = embeddings_by_view["responses"]
    conditional_similarity = []
    conditional_weights = []
    for cluster in sorted(set(int(value) for value in prompt_clusters)):
        selected = response_embeddings[prompt_clusters == cluster]
        if len(selected) < 2:
            continue
        conditional_similarity.append(_mean_pairwise_cosine(selected))
        conditional_weights.append(len(selected))
    weighted_conditional_similarity = float(
        np.average(conditional_similarity, weights=conditional_weights)
    )
    prompt_duplicate_ratio = views["prompts"]["semantic_neighbors"][
        "duplicate_ratio"
    ]
    response_duplicate_ratio = views["responses"]["semantic_neighbors"][
        "duplicate_ratio"
    ]
    severe = prompt_duplicate_ratio >= 0.10 or response_duplicate_ratio >= 0.10
    elevated = prompt_duplicate_ratio >= 0.05 or response_duplicate_ratio >= 0.05
    status = "high_repetition_risk" if severe else (
        "review_recommended" if elevated else "ready_for_training_trial"
    )
    return {
        "status": status,
        "interpretation": (
            "Pre-training diagnostic for repetition, memorization and response "
            "collapse risk; it does not predict SFT loss or downstream quality."
        ),
        "does_not_estimate": [
            "training loss",
            "downstream accuracy",
            "instruction-following quality",
            "model-size-specific learning dynamics",
        ],
        "prompt_semantic_duplicate_ratio": prompt_duplicate_ratio,
        "response_semantic_duplicate_ratio": response_duplicate_ratio,
        "prompt_conditioned_response_mean_cosine": (
            weighted_conditional_similarity
        ),
        "sample_nonduplicate_response_rows": int(
            round(len(response_embeddings) * (1.0 - response_duplicate_ratio))
        ),
        "projected_nonduplicate_response_rows": int(
            round(corpus_rows * (1.0 - response_duplicate_ratio))
        ),
        "projection_warning": (
            "The projected count extrapolates a sample ratio; it is not an "
            "exact corpus-wide semantic deduplication count."
        ),
    }


def audit_rows_semantic_diversity(
    rows: list[dict[str, Any]],
    *,
    input_label: str,
    embedder: SentenceEmbedder,
    model_name: str,
    model_revision: str,
    sample_size: int | None = None,
    cluster_count: int | None = None,
    semantic_duplicate_threshold: float = 0.98,
    batch_size: int = 128,
    random_state: int = 42,
    workers: int = 8,
    prompt_key: str = "prompt",
    response_key: str = "response",
) -> dict[str, Any]:
    if not rows:
        raise ValueError("semantic audit requires at least one row")
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch_size must be positive")
    policy = resolve_quality_audit_policy(
        len(rows), sample_size=sample_size, cluster_count=cluster_count
    )
    sampled = _stable_sample(rows, policy["sample_size"])
    prompts = [_normalize(str(row.get(prompt_key, ""))) for row in sampled]
    responses = [_normalize(str(row.get(response_key, ""))) for row in sampled]
    texts = [f"{prompt}\n{response}" for prompt, response in zip(prompts, responses)]
    labels = [str(row.get("task", "unknown")) for row in sampled]
    splits = [str(row.get("split", "train")) for row in sampled]
    example_ids = [str(row.get("example_id", "")) for row in sampled]
    embeddings_by_view = {}
    for view, surfaces in (
        ("combined", texts),
        ("prompts", prompts),
        ("responses", responses),
    ):
        embeddings = np.asarray(
            embedder.encode(
                surfaces,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(sampled):
            raise ValueError(f"embedder returned an invalid {view} matrix shape")
        embeddings_by_view[view] = embeddings

    dimensions = int(embeddings_by_view["combined"].shape[1])
    views = {
        view: _embedding_view(
            embeddings,
            labels=labels,
            splits=splits,
            example_ids=example_ids,
            cluster_count=policy["cluster_count"],
            semantic_duplicate_threshold=semantic_duplicate_threshold,
            random_state=random_state,
            workers=workers,
        )
        for view, embeddings in embeddings_by_view.items()
    }
    audit = {
        "input": input_label,
        "rows": len(rows),
        "sample": {
            "rows": len(sampled),
            "coverage_ratio": len(sampled) / len(rows),
            "selection": "stable SHA-256 rank of example_id",
        },
        "model": {
            "name": model_name,
            "revision": model_revision,
            "license": "Apache-2.0",
            "embedding_dimensions": dimensions,
            "maximum_sequence_length": int(getattr(embedder, "max_seq_length", 0)),
        },
        "method": {
            "kind": "normalized sentence embeddings",
            "scope": "deterministic statistical sample",
            "warning": (
                "Embedding proximity is a semantic diagnostic, not proof of "
                "dataset correctness or independence."
            ),
        },
        "views": views,
        "sft_readiness_proxy": _sft_readiness_proxy(
            embeddings_by_view,
            views,
            cluster_count=policy["cluster_count"],
            random_state=random_state,
            corpus_rows=len(rows),
        ),
    }
    audit["checks"] = {
        f"{view}_all_clusters_occupied": (
            report["clusters"]["occupied"] == policy["cluster_count"]
        )
        for view, report in views.items()
    }
    audit["checks"].update(
        {
            f"{view}_semantic_duplicate_ratio_below_five_percent": (
                report["semantic_neighbors"]["duplicate_ratio"] < 0.05
            )
            for view, report in views.items()
        }
    )
    audit["checks"].update(
        {
            f"{view}_largest_cluster_below_ten_percent": (
                report["clusters"]["maximum_share"] < 0.10
            )
            for view, report in views.items()
        }
    )
    audit["passed"] = all(audit["checks"].values())
    return audit


def audit_dataset_semantic_diversity(
    conversations_path: Path,
    output_path: Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    model_revision: str = DEFAULT_EMBEDDING_REVISION,
    device: str | None = None,
    sample_size: int | None = None,
    cluster_count: int | None = None,
    semantic_duplicate_threshold: float = 0.98,
    batch_size: int = 128,
    random_state: int = 42,
    workers: int = 8,
) -> dict[str, Any]:
    columns = ["example_id", "task", "split", "prompt", "response"]
    rows = pq.read_table(conversations_path, columns=columns).to_pylist()
    embedder = _load_embedder(model_name, model_revision, device)
    audit = audit_rows_semantic_diversity(
        rows,
        input_label=str(conversations_path.resolve()),
        embedder=embedder,
        model_name=model_name,
        model_revision=model_revision,
        sample_size=sample_size,
        cluster_count=cluster_count,
        semantic_duplicate_threshold=semantic_duplicate_threshold,
        batch_size=batch_size,
        random_state=random_state,
        workers=workers,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit
