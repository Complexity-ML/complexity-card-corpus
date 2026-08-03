from complexity_card_corpus.dictionary_review import build_dictionary_review_data


def _guidance(model: str, tokens: tuple[str, ...]) -> dict:
    return {
        "model": {"name": model, "revision": "test", "license": "Apache-2.0"},
        "dictionary_coherence_audit": {
            "review_queue": [
                {
                    "token": token,
                    "definition": f"Definition for {token}.",
                    "family": "grounded_qa",
                    "domain": "science_passage",
                    "token_definition_cosine": 0.1 + index / 100,
                    "selected_family_rank": index + 1,
                    "review_status": "undefined" if token == "missing" else "uncertain",
                    "review_reasons": (
                        ["definition_missing_or_unsupported"]
                        if token == "missing"
                        else ["token_definition_alignment_robust_outlier"]
                    ),
                }
                for index, token in enumerate(tokens)
            ]
        },
    }


def test_dictionary_review_requires_cross_model_agreement_for_likely_wrong() -> None:
    result = build_dictionary_review_data(
        _guidance("primary", ("missing", "shared", "primary-only")),
        _guidance("secondary", ("shared", "secondary-only")),
        {"shared": "A reviewed definition."},
    )

    by_token = {row["token"]: row for row in result["review"]}
    assert result["rows"] == 4
    assert by_token["missing"]["status"] == "undefined"
    assert by_token["shared"]["status"] == "likely_wrong"
    assert by_token["primary-only"]["status"] == "uncertain"
    assert by_token["secondary-only"]["status"] == "uncertain"
    assert by_token["shared"]["proposed_definition"] == "A reviewed definition."
    assert result["proposed_definitions"] == 1
    assert result["policy"]["automatic_definition_replacement"] is False
