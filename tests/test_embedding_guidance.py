from __future__ import annotations

import hashlib

import numpy as np

from complexity_card_corpus.embedding_guidance import build_embedding_guidance_data


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
            digest = hashlib.sha256(sentence.encode()).digest()
            vector = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            vector -= vector.mean()
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.asarray(vectors)


def test_embedding_guidance_builds_family_decks_without_claiming_synonyms() -> None:
    words = {}
    for index in range(24):
        family = "grounded_qa" if index < 12 else "planning_comparison"
        words[f"term{index}"] = {
            "short_definition": f"Internal statistical gloss number {index}.",
            "selected": {"family": family, "domain": f"domain{index % 3}"},
            "masked_context": {
                "neighbors": [
                    {"token": f"term{(index + offset) % 24}"}
                    for offset in range(1, 4)
                ]
            },
        }
    dictionary = {
        "format": "masked-context-dictionary-v1",
        "definition_policy": "Internal statistical glosses.",
        "words": words,
    }
    audit = {
        "views": {
            "prompts": {
                "family_semantic_duplicate_ratio": {
                    "grounded_qa": 0.20,
                    "planning_comparison": 0.01,
                }
            },
            "responses": {
                "family_semantic_duplicate_ratio": {
                    "grounded_qa": 0.08,
                    "planning_comparison": 0.02,
                }
            },
        }
    }

    guidance = build_embedding_guidance_data(
        dictionary,
        audit,
        embedder=_FakeEmbedder(),
        model_name="fake/model",
        model_revision="test-revision",
        workers=1,
        alternatives_per_token=3,
    )

    assert guidance["dictionary"]["cards"] == 24
    assert guidance["policy"]["automatic_synonym_replacement"] is False
    assert len(guidance["token_alternatives"]["term0"]) == 3
    assert (
        "overgenerate_then_semantically_select"
        in guidance["family_priorities"]["grounded_qa"]["recommended_actions"]
    )
    assert set(guidance["semantic_decks"]) == {
        "grounded_qa",
        "planning_comparison",
    }
    coherence = guidance["dictionary_coherence_audit"]
    assert coherence["token_definition_alignment"]["below_p05_count"] > 0
    assert 0 <= coherence["selected_family_alignment"]["top_3_rate"] <= 1
    assert coherence["review_queue"]
    assert all(row["review_reasons"] for row in coherence["review_queue"])
    assert (
        coherence["token_definition_alignment"]["robust_outlier_count"]
        <= coherence["token_definition_alignment"]["below_p05_count"]
    )
