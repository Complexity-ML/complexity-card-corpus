import numpy as np

from complexity_card_corpus.proposal_embedding_review import (
    audit_definition_proposals_data,
    merge_definition_proposal_audits_data,
)


class FakeEmbedder:
    max_seq_length = 256

    vectors = {
        "A weak science gloss.": [0.0, 1.0, 0.0],
        "A strong science definition.": [1.0, 0.0, 0.0],
        "A second science definition.": [1.0, 0.0, 0.0],
        "A planning definition.": [0.0, 1.0, 0.0],
        "atom": [1.0, 0.0, 0.0],
        'What is the meaning of the word "atom"?': [1.0, 0.0, 0.0],
    }

    def encode(self, sentences, **_kwargs):
        values = np.asarray([self.vectors[text] for text in sentences], dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norms, 1e-8)


def _dictionary() -> dict:
    return {
        "words": {
            "atom": {
                "short_definition": "A weak science gloss.",
                "selected": {"family": "grounded_qa", "domain": "science"},
            },
            "plan": {
                "short_definition": "A planning definition.",
                "selected": {"family": "planning", "domain": "work"},
            },
            "molecule": {
                "short_definition": "A second science definition.",
                "selected": {"family": "grounded_qa", "domain": "science"},
            },
        }
    }


def test_proposal_embedding_audit_keeps_embedding_signal_separate_from_acceptance() -> None:
    result = audit_definition_proposals_data(
        _dictionary(),
        {"definitions": {"atom": "A strong science definition."}},
        embedder=FakeEmbedder(),
        model_name="fake",
        model_revision="test",
    )

    row = result["review"][0]
    assert row["model_signal"] == "improved"
    assert row["proposed_bare_token_cosine"] > row["current_bare_token_cosine"]
    assert row["proposed_lexical_query_cosine"] > row["current_lexical_query_cosine"]
    assert result["policy"]["automatic_definition_replacement"] is False


def test_cross_model_merge_requires_two_improved_signals_for_full_support() -> None:
    base = {
        "model": {"name": "primary", "revision": "test"},
        "review": [
            {
                "token": "atom",
                "current_definition": "old",
                "proposed_definition": "new",
                "family": "grounded_qa",
                "domain": "science",
                "model_signal": "improved",
                "token_delta": 0.2,
                "bare_token_delta": 0.2,
                "lexical_query_delta": 0.2,
                "family_delta": 0.1,
                "current_family_rank": 4,
                "proposed_family_rank": 1,
            }
        ],
    }
    secondary = {
        **base,
        "model": {"name": "secondary", "revision": "test"},
        "review": [{**base["review"][0], "model_signal": "inconclusive"}],
    }
    result = merge_definition_proposal_audits_data(base, secondary)
    assert result["review"][0]["consensus"] == "partially_supported"
    assert result["review"][0]["reviewer_decision"] == "pending"


def test_current_family_drift_cannot_veto_better_token_definition_alignment() -> None:
    result = audit_definition_proposals_data(
        _dictionary(),
        {"definitions": {"atom": "A strong science definition."}},
        embedder=FakeEmbedder(),
        model_name="fake",
        model_revision="test",
    )
    row = result["review"][0]
    assert row["token_delta"] > 0
    assert row["model_signal"] == "improved"
