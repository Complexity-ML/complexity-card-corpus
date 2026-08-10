from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC


_SPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", text.casefold()).strip()


def _is_valid_structured_response(text: str) -> bool:
    """Return whether a compact response is a non-empty JSON object or array."""

    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return False
    return isinstance(value, (dict, list)) and bool(value)


def _stable_sample(rows: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            str(row.get("example_id", "")).encode("utf-8")
        ).digest(),
    )
    return ranked[:limit]


def _source_groups(row: dict[str, Any]) -> set[str]:
    keys = row.get("source_keys") or []
    return {str(key) for key in keys if str(key).strip()}


def _nearest_other_similarity(matrix: Any, *, workers: int) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    neighbors = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=workers)
    neighbors.fit(matrix)
    distances, _ = neighbors.kneighbors(matrix)
    return 1.0 - distances[:, 1]


def _cross_split_similarity(
    train_matrix: Any, evaluation_matrix: Any, *, workers: int
) -> np.ndarray:
    if train_matrix.shape[0] == 0 or evaluation_matrix.shape[0] == 0:
        return np.zeros(evaluation_matrix.shape[0], dtype=np.float32)
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=workers)
    neighbors.fit(train_matrix)
    distances, _ = neighbors.kneighbors(evaluation_matrix)
    return 1.0 - distances[:, 0]


def resolve_quality_audit_policy(
    row_count: int,
    *,
    sample_size: int | None = None,
    max_features: int | None = None,
    cluster_count: int | None = None,
) -> dict[str, int]:
    """Resolve adaptive model sizes while retaining all-row scoring."""

    if row_count < 1:
        raise ValueError("row_count must be positive")
    resolved_sample_size = min(
        row_count,
        sample_size
        if sample_size is not None and sample_size > 0
        else max(5_000, math.ceil(math.cbrt(row_count) * 120)),
    )
    resolved_max_features = (
        max_features
        if max_features is not None and max_features > 0
        else max(20_000, resolved_sample_size * 4)
    )
    resolved_cluster_count = min(
        resolved_sample_size,
        cluster_count
        if cluster_count is not None and cluster_count > 0
        else min(256, max(2, math.ceil(math.sqrt(resolved_sample_size) / 2))),
    )
    return {
        "sample_size": resolved_sample_size,
        "max_features": resolved_max_features,
        "cluster_count": resolved_cluster_count,
        "score_batch_size": min(
            10_000, max(1_000, math.ceil(math.sqrt(row_count) * 10))
        ),
    }


def audit_rows_quality(
    rows: list[dict[str, Any]],
    *,
    input_label: str,
    sample_size: int | None = None,
    near_duplicate_threshold: float = 0.95,
    max_features: int | None = None,
    cluster_count: int | None = None,
    random_state: int = 42,
    workers: int = 8,
    prompt_key: str = "prompt",
    response_key: str = "response",
) -> dict[str, Any]:
    """Audit lexical diversity, split leakage, clusters, and family ambiguity.

    The audit is intentionally a statistical complement to the exact schema and
    source-group checks. It does not generate or rewrite dataset content.
    """

    if workers < 1:
        raise ValueError("workers must be positive")
    if not rows:
        raise ValueError("quality audit requires at least one row")
    row_count = len(rows)
    policy = resolve_quality_audit_policy(
        row_count,
        sample_size=sample_size,
        max_features=max_features,
        cluster_count=cluster_count,
    )
    resolved_sample_size = policy["sample_size"]
    resolved_max_features = policy["max_features"]
    resolved_cluster_count = policy["cluster_count"]
    sampled = _stable_sample(rows, resolved_sample_size)
    texts = [
        _normalize(f"{row.get(prompt_key, '')}\n{row.get(response_key, '')}")
        for row in sampled
    ]
    response_texts = [
        _normalize(str(row.get(response_key, ""))) for row in sampled
    ]
    labels = [str(row.get("task", "unknown")) for row in sampled]
    splits = [str(row.get("split", "train")) for row in sampled]

    all_normalized_texts = (
        _normalize(f"{row.get(prompt_key, '')}\n{row.get(response_key, '')}")
        for row in rows
    )
    normalized_counts = Counter(all_normalized_texts)
    exact_duplicate_rows = sum(count - 1 for count in normalized_counts.values())
    normalized_response_counts = Counter(
        _normalize(str(row.get(response_key, ""))) for row in rows
    )
    exact_duplicate_responses = sum(
        count - 1 for count in normalized_response_counts.values()
    )

    train_groups: set[str] = set()
    evaluation_groups: set[str] = set()
    for row in rows:
        target = train_groups if row.get("split") == "train" else evaluation_groups
        target.update(_source_groups(row))
    leaking_groups = sorted(train_groups & evaluation_groups)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=resolved_max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    nearest = _nearest_other_similarity(matrix, workers=workers)
    near_duplicate_rows = int(np.count_nonzero(nearest >= near_duplicate_threshold))

    # Prompt diversity can conceal a collapsed answer surface when both roles
    # are embedded together.  Score final responses independently so lexical
    # substitutions around one learned discourse pattern cannot pass merely
    # because their prompts differ.
    response_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=resolved_max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    response_matrix = response_vectorizer.fit_transform(response_texts)
    response_nearest = _nearest_other_similarity(
        response_matrix,
        workers=workers,
    )
    near_duplicate_responses = int(
        np.count_nonzero(response_nearest >= near_duplicate_threshold)
    )

    train_indices = np.asarray([i for i, split in enumerate(splits) if split == "train"])
    eval_indices = np.asarray([i for i, split in enumerate(splits) if split != "train"])
    cross_similarity = _cross_split_similarity(
        matrix[train_indices], matrix[eval_indices], workers=workers
    )

    clusterer = MiniBatchKMeans(
        n_clusters=resolved_cluster_count,
        batch_size=min(4096, max(256, len(sampled))),
        n_init="auto",
        random_state=random_state,
    )
    cluster_labels = clusterer.fit_predict(matrix)
    cluster_sizes = Counter(int(label) for label in cluster_labels)

    family_f1: float | None = None
    classifier: LinearSVC | None = None
    family_counts = Counter(labels)
    eligible = [i for i, label in enumerate(labels) if family_counts[label] >= 4]
    eligible_labels = [labels[i] for i in eligible]
    if len(set(eligible_labels)) >= 2 and len(eligible) >= 40:
        train_idx, test_idx = train_test_split(
            np.asarray(eligible),
            test_size=0.2,
            random_state=random_state,
            stratify=eligible_labels,
        )
        classifier = LinearSVC(class_weight="balanced", random_state=random_state)
        classifier.fit(matrix[train_idx], np.asarray(labels)[train_idx])
        predictions = classifier.predict(matrix[test_idx])
        family_f1 = float(
            f1_score(
                np.asarray(labels)[test_idx],
                predictions,
                average="macro",
                zero_division=0,
            )
        )

    outlier_limit = min(
        len(sampled), max(1_000, math.ceil(len(sampled) / 5))
    )
    outlier_indices = np.arange(outlier_limit)
    component_count = min(32, max(2, matrix.shape[1] - 1), outlier_limit - 1)
    reducer = TruncatedSVD(
        n_components=component_count,
        n_iter=5,
        random_state=random_state,
    )
    reduced = reducer.fit_transform(matrix[outlier_indices])
    outlier_model = IsolationForest(
        n_estimators=100,
        max_samples=min(2048, outlier_limit),
        contamination="auto",
        n_jobs=workers,
        random_state=random_state,
    )
    outlier_model.fit(reduced)
    outlier_scores = outlier_model.score_samples(reduced)

    all_cluster_sizes: Counter[int] = Counter()
    family_mismatches = 0
    worst_surfaces: list[tuple[float, str, str, str]] = []
    malformed: list[dict[str, str]] = []
    malformed_count = 0
    batch_size = policy["score_batch_size"]
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        batch_texts = [
            _normalize(f"{row.get(prompt_key, '')}\n{row.get(response_key, '')}")
            for row in batch
        ]
        for row in batch:
            prompt = str(row.get(prompt_key, "")).strip()
            response = str(row.get(response_key, "")).strip()
            reasons = []
            if not prompt:
                reasons.append("empty_prompt")
            if not response:
                reasons.append("empty_response")
            if prompt and response and _normalize(prompt) == _normalize(response):
                reasons.append("prompt_equals_response")
            if len(response.split()) < 2 and not _is_valid_structured_response(response):
                reasons.append("response_below_two_words")
            if reasons:
                malformed_count += 1
                if len(malformed) < 100:
                    malformed.append(
                        {
                            "example_id": str(row.get("example_id", "")),
                            "task": str(row.get("task", "unknown")),
                            "reason": ",".join(reasons),
                        }
                    )
        batch_matrix = vectorizer.transform(batch_texts)
        all_cluster_sizes.update(
            int(label) for label in clusterer.predict(batch_matrix)
        )
        batch_reduced = reducer.transform(batch_matrix)
        batch_scores = outlier_model.score_samples(batch_reduced)
        predicted = (
            classifier.predict(batch_matrix)
            if classifier is not None
            else np.asarray(["unknown"] * len(batch))
        )
        for row, score, predicted_task in zip(
            batch, batch_scores, predicted, strict=True
        ):
            task = str(row.get("task", "unknown"))
            if classifier is not None and str(predicted_task) != task:
                family_mismatches += 1
            item = (
                -float(score),
                str(row.get("example_id", "")),
                task,
                str(predicted_task),
            )
            if len(worst_surfaces) < 100:
                heapq.heappush(worst_surfaces, item)
            elif item[0] > worst_surfaces[0][0]:
                heapq.heapreplace(worst_surfaces, item)

    outlier_review = [
        {
            "example_id": example_id,
            "task": task,
            "predicted_task": predicted_task,
            "score": -negative_score,
        }
        for negative_score, example_id, task, predicted_task in sorted(
            worst_surfaces, reverse=True
        )
    ]

    audit = {
        "input": input_label,
        "rows": len(rows),
        "sample_rows": len(sampled),
        "sample_method": "sha256(example_id), lowest hashes",
        "adaptive_policy": {
            "sample_size": resolved_sample_size,
            "sample_formula": "min(N, max(5000, ceil(120*cuberoot(N))))",
            "max_features": resolved_max_features,
            "feature_formula": "max(20000, 4*sample_size)",
            "clusters": resolved_cluster_count,
            "cluster_formula": "min(256, ceil(sqrt(sample_size)/2))",
            "score_batch_size": batch_size,
            "manual_overrides_supported": True,
        },
        "vectorization": {
            "kind": "tfidf_char_wb",
            "ngram_range": [3, 5],
            "features": int(matrix.shape[1]),
            "workers": workers,
        },
        "exact_duplicates": {
            "rows": exact_duplicate_rows,
            "ratio": exact_duplicate_rows / max(1, len(rows)),
            "coverage": "all_rows",
        },
        "near_duplicates": {
            "threshold": near_duplicate_threshold,
            "rows": near_duplicate_rows,
            "ratio": near_duplicate_rows / max(1, len(sampled)),
            "maximum_similarity": float(nearest.max(initial=0.0)),
            "p95_similarity": float(np.quantile(nearest, 0.95)) if len(nearest) else 0.0,
        },
        "response_only_repetition": {
            "method": "tfidf_char_wb_nearest_response",
            "threshold": near_duplicate_threshold,
            "sample_rows": len(sampled),
            "exact_duplicate_rows": exact_duplicate_responses,
            "near_duplicate_rows": near_duplicate_responses,
            "near_duplicate_ratio": near_duplicate_responses
            / max(1, len(sampled)),
            "maximum_similarity": float(response_nearest.max(initial=0.0)),
            "p95_similarity": float(np.quantile(response_nearest, 0.95))
            if len(response_nearest)
            else 0.0,
            "interpretation": (
                "Responses are measured without prompts so prompt diversity "
                "cannot conceal repeated answer behaviour."
            ),
        },
        "split_leakage": {
            "source_group_overlap_count": len(leaking_groups),
            "source_group_overlap_preview": leaking_groups[:20],
            "evaluation_rows_sampled": int(len(eval_indices)),
            "near_duplicate_evaluation_rows": int(
                np.count_nonzero(cross_similarity >= near_duplicate_threshold)
            ),
            "maximum_cross_split_similarity": float(
                cross_similarity.max(initial=0.0)
            ),
        },
        "clusters": {
            "count": resolved_cluster_count,
            "occupied": len(all_cluster_sizes),
            "minimum_size": min(all_cluster_sizes.values(), default=0),
            "maximum_size": max(all_cluster_sizes.values(), default=0),
            "maximum_share": max(all_cluster_sizes.values(), default=0)
            / max(1, len(rows)),
            "coverage": "all_rows_scored_after_sample_fit",
        },
        "family_predictability": {
            "macro_f1": family_f1,
            "all_rows_mismatch_count": family_mismatches,
            "all_rows_mismatch_ratio": family_mismatches / max(1, len(rows)),
            "interpretation": (
                "Higher values indicate clearer lexical separation between declared families; "
                "this is a diagnostic, not a quality target."
            ),
        },
        "surface_outliers": {
            "method": "tfidf_char_ngrams+truncated_svd+isolation_forest",
            "audited_rows": outlier_limit,
            "components": component_count,
            "score_p01": float(np.quantile(outlier_scores, 0.01)),
            "score_median": float(np.median(outlier_scores)),
            "scored_rows": len(rows),
            "lowest_score_examples": outlier_review,
            "interpretation": (
                "Low scores identify unusual lexical surfaces for human review; "
                "they do not establish grammatical incorrectness."
            ),
        },
        "format_checks": {
            "coverage": "all_rows",
            "malformed_count": malformed_count,
            "malformed_preview": malformed,
        },
        "checks": {
            "no_exact_duplicates": exact_duplicate_rows == 0,
            "no_source_group_leakage": not leaking_groups,
            "near_duplicate_ratio_below_five_percent": (
                near_duplicate_rows / max(1, len(sampled)) < 0.05
            ),
            "response_near_duplicate_ratio_below_five_percent": (
                near_duplicate_responses / max(1, len(sampled)) < 0.05
            ),
            "all_clusters_occupied": (
                len(all_cluster_sizes) == resolved_cluster_count
            ),
            "no_basic_format_failures": malformed_count == 0,
        },
    }
    audit["passed"] = all(audit["checks"].values())
    return audit


def audit_dataset_quality(
    conversations_path: Path,
    output_path: Path,
    *,
    sample_size: int | None = None,
    near_duplicate_threshold: float = 0.95,
    max_features: int | None = None,
    cluster_count: int | None = None,
    random_state: int = 42,
    workers: int = 8,
) -> dict[str, Any]:
    columns = [
        "example_id",
        "task",
        "split",
        "prompt",
        "response",
        "source_keys",
    ]
    rows = pq.read_table(conversations_path, columns=columns).to_pylist()
    audit = audit_rows_quality(
        rows,
        input_label=str(conversations_path.resolve()),
        sample_size=sample_size,
        near_duplicate_threshold=near_duplicate_threshold,
        max_features=max_features,
        cluster_count=cluster_count,
        random_state=random_state,
        workers=workers,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit
