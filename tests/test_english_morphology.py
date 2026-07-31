from __future__ import annotations

import pytest

from complexity_card_corpus.english_morphology import (
    VerbFeatures,
    VerbPhrase,
    realize_clause,
    realize_verb_phrase,
    verb_forms,
)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        (
            "verify a proposed fix",
            {
                "base": "verify a proposed fix",
                "third_person_singular": "verifies a proposed fix",
                "past": "verified a proposed fix",
                "past_participle": "verified a proposed fix",
                "present_participle": "verifying a proposed fix",
            },
        ),
        (
            "choose a gentle next step",
            {
                "base": "choose a gentle next step",
                "third_person_singular": "chooses a gentle next step",
                "past": "chose a gentle next step",
                "past_participle": "chosen a gentle next step",
                "present_participle": "choosing a gentle next step",
            },
        ),
        (
            "set a clear boundary",
            {
                "base": "set a clear boundary",
                "third_person_singular": "sets a clear boundary",
                "past": "set a clear boundary",
                "past_participle": "set a clear boundary",
                "present_participle": "setting a clear boundary",
            },
        ),
    ],
)
def test_verb_forms_cover_regular_and_irregular_phrases(
    phrase: str, expected: dict[str, str]
) -> None:
    assert verb_forms(phrase) == expected


def test_clause_realization_handles_agreement_aspect_and_negation() -> None:
    phrase = "verify a proposed fix"

    assert realize_clause("the response", phrase) == (
        "the response verifies a proposed fix"
    )
    assert realize_clause(
        "the responses", phrase, VerbFeatures(number="plural")
    ) == "the responses verify a proposed fix"
    assert realize_clause(
        "the response", phrase, VerbFeatures(tense="past")
    ) == "the response verified a proposed fix"
    assert realize_clause(
        "the response", phrase, VerbFeatures(aspect="progressive")
    ) == "the response is verifying a proposed fix"
    assert realize_clause(
        "the response", phrase, VerbFeatures(aspect="perfect")
    ) == "the response has verified a proposed fix"
    assert realize_clause(
        "the response", phrase, VerbFeatures(tense="future")
    ) == "the response will verify a proposed fix"
    assert realize_clause(
        "the response", phrase, VerbFeatures(negated=True)
    ) == "the response does not verify a proposed fix"


def test_clause_realization_handles_modals_questions_and_be() -> None:
    assert realize_clause(
        "the response",
        "verify a proposed fix",
        VerbFeatures(modal="should", interrogative=True),
    ) == "should the response verify a proposed fix"
    assert realize_clause(
        "the response",
        "verify a proposed fix",
        VerbFeatures(tense="past", interrogative=True),
    ) == "did the response verify a proposed fix"
    assert realize_clause(
        "the response",
        "verify a proposed fix",
        VerbFeatures(interrogative=True, negated=True),
    ) == "does the response not verify a proposed fix"
    assert realize_verb_phrase("be ready") == "is ready"
    assert realize_verb_phrase(
        "be ready", VerbFeatures(negated=True)
    ) == "is not ready"
    assert realize_clause(
        "the operator", "be ready", VerbFeatures(interrogative=True)
    ) == "is the operator ready"


def test_verb_phrase_rejects_non_lexical_input() -> None:
    with pytest.raises(ValueError, match="alphabetic lemma"):
        VerbPhrase.parse("42 checks")
