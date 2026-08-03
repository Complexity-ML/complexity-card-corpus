from __future__ import annotations

import hashlib

import numpy as np

from complexity_card_corpus.semantic_audit import audit_rows_semantic_diversity


class _FakeEmbedder:
    max_seq_length = 256

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        vectors = []
        for sentence in sentences:
            digest = hashlib.sha256(sentence.encode("utf-8")).digest()
            vector = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            vector -= vector.mean()
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.asarray(vectors)


def test_semantic_audit_reports_vector_geometry_and_family_dispersion() -> None:
    rows = [
        {
            "example_id": f"semantic:{index:03d}",
            "task": "grounded_qa" if index % 2 else "planning_comparison",
            "split": "validation" if index % 10 == 0 else "train",
            "prompt": f"Distinct request {index} about a documented decision.",
            "response": f"Distinct answer {index} preserves the supported result.",
        }
        for index in range(80)
    ]

    audit = audit_rows_semantic_diversity(
        rows,
        input_label="semantic regression",
        embedder=_FakeEmbedder(),
        model_name="fake/model",
        model_revision="test-revision",
        sample_size=80,
        cluster_count=8,
        workers=1,
    )

    assert audit["rows"] == 80
    assert audit["sample"]["rows"] == 80
    assert audit["model"]["embedding_dimensions"] == 32
    assert set(audit["views"]) == {"combined", "prompts", "responses"}
    assert audit["views"]["combined"]["clusters"]["occupied"] == 8
    assert set(audit["views"]["responses"]["family_dispersion"]) == {
        "grounded_qa",
        "planning_comparison",
    }
    assert audit["views"]["combined"]["global_geometry"]["entropy_rank"] > 1
    assert len(audit["views"]["responses"]["nearest_pair_preview"]) == 50
    assert set(audit["views"]["responses"]["family_semantic_duplicate_ratio"]) == {
        "grounded_qa",
        "planning_comparison",
    }
    assert audit["checks"]["combined_all_clusters_occupied"]
    assert audit["sft_readiness_proxy"]["status"] in {
        "ready_for_training_trial",
        "review_recommended",
        "high_repetition_risk",
    }
    assert "training loss" in audit["sft_readiness_proxy"]["does_not_estimate"]
