from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from complexity_card_corpus.vocabulary.dictionary import (
    _masked_dictionary,
    _write_masked_dictionary,
)
from complexity_card_corpus.vocabulary_wordnet_audit import _lookup_forms


def test_masked_dictionary_keeps_matrix_and_multiple_statistical_usages() -> None:
    practical = ("practical_action", "home_repair")
    explanation = ("explanation_learning", "computing")
    cells = [practical, explanation]
    placements = [
        {
            "token": "wrench",
            "family": practical[0],
            "domain": practical[1],
            "assignment_method": "cross_source_context",
        }
    ]
    scored = {
        "wrench": (
            {practical: 9.0, explanation: 4.0},
            {practical: 2, explanation: 1},
            {practical: 0.5, explanation: 0.0},
        )
    }
    neighbors = defaultdict(lambda: defaultdict(Counter))
    neighbors["wrench"]["source-a"].update({"tool": 4})
    neighbors["wrench"]["source-b"].update({"tool": 3})
    roles = {"wrench": Counter({"intent_term": 7})}
    occurrences = {
        "wrench": Counter({"source-a": 5, "source-b": 4}),
        "tool": Counter({"source-a": 10, "source-b": 8}),
    }

    result = _masked_dictionary(
        placements,
        scored,
        neighbors,
        roles,
        occurrences,
        cells,
    )

    assert result["matrix"]["shape"] == [1, 2]
    entry = result["words"]["wrench"]
    assert entry["selected"]["classification_status"] == ("statistically_supported")
    assert entry["cell_score_vector"] == [9.0, 4.0]
    assert [usage["family"] for usage in entry["statistical_usages"]] == [
        "practical_action",
        "explanation_learning",
    ]
    assert entry["statistical_usages"][0]["usage_gloss"] == (
        "The selected generation use places this term in home repair "
        "contexts for practical action."
    )
    assert entry["statistical_usages"][0]["usage_kind"] == ("selected_generation")
    assert entry["masked_context"]["neighbors"][0]["token"] == "tool"
    assert "tool" in entry["short_definition"]
    assert result["audit"]["source_text_retained"] is False


def test_wordnet_lookup_forms_cover_common_inflections() -> None:
    assert "test" in _lookup_forms("tests")
    assert "study" in _lookup_forms("studies")
    assert "murder" in _lookup_forms("murdered")
    assert "capture" in _lookup_forms("capturing")


def test_masked_dictionary_writer_keeps_valid_readable_json(
    tmp_path: Path,
) -> None:
    output = tmp_path / "dictionary.json"
    dictionary = {
        "format": "fixture",
        "matrix": {"axis": [], "shape": [1, 0]},
        "words": {"term": {"short_definition": "A small definition."}},
    }

    _write_masked_dictionary(output, dictionary)

    assert json.loads(output.read_text()) == dictionary
    assert len(output.read_text().splitlines()) > 4
