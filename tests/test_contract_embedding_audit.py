from __future__ import annotations

import json

import numpy as np

from complexity_card_corpus.contract_embedding_audit import (
    audit_contract_rows_with_embeddings,
)


class _KeywordEmbedder:
    max_seq_length = 128

    def encode(self, texts, **_kwargs):
        vocabulary = ("answer", "evidence", "unknown", "classify", "label", "basis")
        matrix = []
        for text in texts:
            lowered = text.lower()
            vector = np.asarray(
                [float(word in lowered) for word in vocabulary], dtype=np.float32
            )
            norm = np.linalg.norm(vector)
            matrix.append(vector / norm if norm else np.ones(6, dtype=np.float32) / 6)
        return np.asarray(matrix, dtype=np.float32)


def _row(family: str, intent: str, surface: str, contract: list[str]) -> dict:
    return {
        "answer_json": json.dumps(
            {
                "family": family,
                "intent": intent,
                "surface_intent": surface,
                "card_hand": {"completion_contract": contract},
            }
        )
    }


def test_contract_embedding_audit_compares_intents_with_family_contracts() -> None:
    rows = [
        _row(
            "grounded_qa",
            "answer",
            "answer from evidence and mark unknown details",
            ["direct_answer", "evidence", "unknown"],
        ),
        _row(
            "extraction_classification",
            "classify",
            "classify with a label and evidence basis",
            ["json", "defined_label", "evidence_basis"],
        ),
    ]
    report = audit_contract_rows_with_embeddings(
        rows,
        input_label="memory",
        embedder=_KeywordEmbedder(),
        model_name="fake",
        model_revision="test",
    )

    assert report["family_intent_pairs"] == 2
    assert report["passed_structural_checks"] is False
    assert report["summary"]["top1_contract_match_ratio"] == 1.0


def test_contract_embedding_audit_rejects_multiple_contracts_for_one_intent() -> None:
    rows = [
        _row("grounded_qa", "answer", "answer directly", ["direct_answer"]),
        _row("grounded_qa", "answer", "answer directly", ["other_shape"]),
    ]
    try:
        audit_contract_rows_with_embeddings(
            rows,
            input_label="memory",
            embedder=_KeywordEmbedder(),
            model_name="fake",
            model_revision="test",
        )
    except ValueError as error:
        assert "multiple contracts" in str(error)
    else:
        raise AssertionError("ambiguous family-intent contract was accepted")
