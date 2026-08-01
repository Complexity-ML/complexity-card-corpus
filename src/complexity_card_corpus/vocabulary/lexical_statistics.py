from __future__ import annotations

import hashlib
import re
import statistics
from collections import Counter
from typing import Any

from ..surface_reference import SurfaceStructureAccumulator
from .lexical_audit import _normalized_tokens
from .lexical_schema import _ApproxDistinct


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return float(ordered[index])


def _source_stats(
    lengths: list[int],
    vocabulary: Counter[str],
    questions: int,
    retained: set[str],
    unique_documents_estimate: int,
    sentence_count: int,
    unique_sentences_estimate: int,
    surface_structure: dict[str, Any],
) -> dict[str, Any]:
    occurrences = sum(vocabulary.values())
    retained_occurrences = sum(vocabulary[token] for token in retained)
    return {
        "documents": len(lengths),
        "word_occurrences": occurrences,
        "observed_vocabulary": len(vocabulary),
        "mean_words": round(statistics.fmean(lengths), 3) if lengths else 0.0,
        "median_words": round(statistics.median(lengths), 3) if lengths else 0.0,
        "p95_words": _percentile(lengths, 0.95),
        "question_rate": round(questions / len(lengths), 6) if lengths else 0.0,
        "unique_document_rate_estimate": round(
            unique_documents_estimate / len(lengths), 6
        )
        if lengths
        else 0.0,
        "unique_sentence_rate_estimate": round(
            unique_sentences_estimate / sentence_count, 6
        )
        if sentence_count
        else 0.0,
        "distinct_counter": "linear_counting_2^24_bits",
        "type_token_ratio": round(len(vocabulary) / occurrences, 6)
        if occurrences
        else 0.0,
        "retained_vocabulary": len(retained),
        "retained_occurrence_coverage": round(retained_occurrences / occurrences, 6)
        if occurrences
        else 0.0,
        "surface_structure": surface_structure,
    }


_REPETITION_LEVELS = (
    ("unique", 1, 1),
    ("2-4", 2, 4),
    ("5-9", 5, 9),
    ("10-24", 10, 24),
    ("25+", 25, None),
)


def _unit_digest(value: str) -> bytes:
    return hashlib.blake2b(value.encode(), digest_size=16).digest()


def _repetition_profile(frequencies: Counter[bytes]) -> dict[str, Any]:
    total_occurrences = sum(frequencies.values())
    levels: dict[str, dict[str, int | float]] = {}
    for label, minimum, maximum in _REPETITION_LEVELS:
        values = [
            count
            for count in frequencies.values()
            if count >= minimum and (maximum is None or count <= maximum)
        ]
        occurrences = sum(values)
        levels[label] = {
            "units": len(values),
            "occurrences": occurrences,
            "occurrence_share": round(occurrences / total_occurrences, 6)
            if total_occurrences
            else 0.0,
        }
    repeated_occurrences = sum(max(0, count - 1) for count in frequencies.values())
    return {
        "counting_unit": "blake2b_128_digest_in_memory",
        "hashes_retained": False,
        "units": len(frequencies),
        "occurrences": total_occurrences,
        "maximum_occurrences": max(frequencies.values(), default=0),
        "repeated_occurrences": repeated_occurrences,
        "repeated_occurrence_share": round(repeated_occurrences / total_occurrences, 6)
        if total_occurrences
        else 0.0,
        "levels": levels,
    }


def _new_stats_accumulator() -> dict[str, Any]:
    return {
        "lengths": [],
        "vocabulary": Counter(),
        "questions": 0,
        "document_counter": _ApproxDistinct(),
        "sentence_counter": _ApproxDistinct(),
        "sentence_count": 0,
        "surface_structure": SurfaceStructureAccumulator(window_tokens=8),
        "document_frequencies": Counter(),
        "sentence_frequencies": Counter(),
    }


def _accumulate_stats(
    accumulator: dict[str, Any], text: str, tokens: list[str]
) -> None:
    accumulator["surface_structure"].add(text)
    accumulator["lengths"].append(len(tokens))
    accumulator["questions"] += int(text.rstrip().endswith("?"))
    normalized_document = " ".join(tokens)
    accumulator["document_counter"].add(normalized_document)
    accumulator["document_frequencies"][_unit_digest(normalized_document)] += 1
    accumulator["vocabulary"].update(tokens)
    for sentence in re.split(r"[.!?]+", text):
        if not sentence.strip():
            continue
        sentence_tokens = _normalized_tokens(sentence)
        if sentence_tokens:
            accumulator["sentence_count"] += 1
            normalized_sentence = " ".join(sentence_tokens)
            accumulator["sentence_counter"].add(normalized_sentence)
            accumulator["sentence_frequencies"][_unit_digest(normalized_sentence)] += 1


def _finalize_stats(accumulator: dict[str, Any], retained: set[str]) -> dict[str, Any]:
    return {
        **_source_stats(
            accumulator["lengths"],
            accumulator["vocabulary"],
            accumulator["questions"],
            retained,
            accumulator["document_counter"].estimate(),
            accumulator["sentence_count"],
            accumulator["sentence_counter"].estimate(),
            accumulator["surface_structure"].summary(),
        ),
        "document_repetition": _repetition_profile(accumulator["document_frequencies"]),
        "sentence_repetition": _repetition_profile(accumulator["sentence_frequencies"]),
    }
