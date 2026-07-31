from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import Any, Iterable


SURFACE_TOKEN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|[0-9]+")
SENTENCE_BOUNDARY = re.compile(r"[.!?]+")

TRANSITIONS = frozenset(
    {
        "after",
        "although",
        "before",
        "because",
        "finally",
        "first",
        "following",
        "however",
        "instead",
        "meanwhile",
        "next",
        "once",
        "otherwise",
        "then",
        "therefore",
        "unless",
        "until",
        "when",
        "while",
    }
)
DETERMINERS = frozenset(
    {"a", "an", "another", "any", "each", "either", "every", "no", "some", "the", "this", "that", "these", "those"}
)
PRONOUNS = frozenset(
    {
        "he",
        "her",
        "hers",
        "him",
        "his",
        "i",
        "it",
        "its",
        "me",
        "mine",
        "our",
        "ours",
        "she",
        "their",
        "theirs",
        "them",
        "they",
        "we",
        "you",
        "your",
        "yours",
    }
)
AUXILIARIES = frozenset(
    {
        "am",
        "are",
        "be",
        "been",
        "being",
        "can",
        "could",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "is",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "was",
        "were",
        "will",
        "would",
    }
)
CONJUNCTIONS = frozenset({"and", "but", "for", "nor", "or", "so", "yet"})
PREPOSITIONS = frozenset(
    {
        "about",
        "above",
        "across",
        "against",
        "among",
        "around",
        "at",
        "by",
        "during",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "over",
        "through",
        "to",
        "under",
        "with",
        "within",
        "without",
    }
)
NEGATIONS = frozenset({"can't", "cannot", "never", "no", "not", "won't"})
_REPETITION_LEVELS = (
    ("unique", 1, 1),
    ("2-4", 2, 4),
    ("5-9", 5, 9),
    ("10-24", 10, 24),
    ("25+", 25, None),
)


def _tokens(text: str) -> list[str]:
    return [
        match.group(0).replace("’", "'").lower()
        for match in SURFACE_TOKEN.finditer(text)
    ]


def _token_class(token: str) -> str:
    if token.isdigit():
        return "NUMBER"
    if token in NEGATIONS:
        return "NEGATION"
    if token in TRANSITIONS:
        return "TRANSITION"
    if token in DETERMINERS:
        return "DETERMINER"
    if token in PRONOUNS:
        return "PRONOUN"
    if token in AUXILIARIES:
        return "AUXILIARY"
    if token in CONJUNCTIONS:
        return "CONJUNCTION"
    if token in PREPOSITIONS:
        return "PREPOSITION"
    if token.endswith("ly"):
        return "ADVERB_SHAPE"
    return "CONTENT"


def _entropy(counter: Counter[tuple[str, ...]]) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total) for count in counter.values()
    )


def _repetition_levels(
    counter: Counter[tuple[str, ...]], shape_tokens: int
) -> dict[str, Any]:
    occurrences = sum(counter.values())
    levels: dict[str, dict[str, int | float]] = {}
    for label, minimum, maximum in _REPETITION_LEVELS:
        values = [
            count
            for count in counter.values()
            if count >= minimum and (maximum is None or count <= maximum)
        ]
        level_occurrences = sum(values)
        levels[label] = {
            "shapes": len(values),
            "occurrences": level_occurrences,
            "occurrence_share": round(level_occurrences / occurrences, 6)
            if occurrences
            else 0.0,
        }
    return {
        "mask": "coarse_token_classes",
        "shape_tokens": shape_tokens,
        "shapes": len(counter),
        "occurrences": occurrences,
        "maximum_occurrences": max(counter.values(), default=0),
        "levels": levels,
        "shape_values_retained": False,
        "lexical_tokens_retained": False,
    }


def _js_divergence(
    left: Counter[tuple[str, ...] | str],
    right: Counter[tuple[str, ...] | str],
) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not left_total or not right_total:
        return 0.0
    divergence = 0.0
    for key in left.keys() | right.keys():
        p = left[key] / left_total
        q = right[key] / right_total
        midpoint = (p + q) / 2
        if p:
            divergence += 0.5 * p * math.log2(p / midpoint)
        if q:
            divergence += 0.5 * q * math.log2(q / midpoint)
    return divergence


class SurfaceStructureAccumulator:
    """Aggregate non-lexical sentence and eight-token structure statistics."""

    def __init__(self, window_tokens: int = 8) -> None:
        if window_tokens < 4:
            raise ValueError("surface structure windows must contain at least four tokens")
        self.window_tokens = window_tokens
        self.window_shapes: Counter[tuple[str, ...]] = Counter()
        self.opening_shapes: Counter[tuple[str, ...]] = Counter()
        self.closing_shapes: Counter[tuple[str, ...]] = Counter()
        self.transition_positions: Counter[str] = Counter()
        self.sentence_lengths: list[int] = []
        self.adjacent_repetitions = 0
        self.token_pairs = 0

    def add(self, text: str) -> None:
        for sentence in SENTENCE_BOUNDARY.split(text):
            tokens = _tokens(sentence)
            if not tokens:
                continue
            classes = [_token_class(token) for token in tokens]
            self.sentence_lengths.append(len(tokens))
            self.opening_shapes[tuple(classes[:3])] += 1
            self.closing_shapes[tuple(classes[-3:])] += 1
            for index in range(max(0, len(classes) - self.window_tokens + 1)):
                shape = tuple(classes[index : index + self.window_tokens])
                self.window_shapes[shape] += 1
            for index, token in enumerate(tokens):
                if token in TRANSITIONS:
                    if index == 0:
                        bucket = "initial"
                    elif index <= 2:
                        bucket = "early"
                    else:
                        bucket = "interior"
                    self.transition_positions[bucket] += 1
            self.token_pairs += max(0, len(tokens) - 1)
            self.adjacent_repetitions += sum(
                left == right for left, right in zip(tokens, tokens[1:])
            )

    def extend(self, texts: Iterable[str]) -> None:
        for text in texts:
            self.add(text)

    def summary(self) -> dict[str, Any]:
        sentences = len(self.sentence_lengths)
        transitions = sum(self.transition_positions.values())
        windows = sum(self.window_shapes.values())
        return {
            "window_tokens": self.window_tokens,
            "sentences": sentences,
            "eight_token_windows": windows,
            "unique_abstract_window_shapes": len(self.window_shapes),
            "window_shape_entropy_bits": round(_entropy(self.window_shapes), 6),
            "masked_window_repetition": _repetition_levels(
                self.window_shapes, self.window_tokens
            ),
            "mean_sentence_words": round(
                statistics.fmean(self.sentence_lengths), 6
            )
            if sentences
            else 0.0,
            "transition_rate_per_sentence": round(transitions / sentences, 6)
            if sentences
            else 0.0,
            "sentence_initial_transition_rate": round(
                self.transition_positions["initial"] / sentences, 6
            )
            if sentences
            else 0.0,
            "adjacent_repetition_rate": round(
                self.adjacent_repetitions / self.token_pairs, 8
            )
            if self.token_pairs
            else 0.0,
            "retained_lexical_ngrams": False,
            "retained_source_text": False,
        }


def compare_surface_structures(
    reference: SurfaceStructureAccumulator,
    candidate: SurfaceStructureAccumulator,
) -> dict[str, Any]:
    if reference.window_tokens != candidate.window_tokens:
        raise ValueError("surface structure windows must use the same size")
    reference_summary = reference.summary()
    candidate_summary = candidate.summary()
    reference_repetition = reference_summary["masked_window_repetition"]["levels"]
    candidate_repetition = candidate_summary["masked_window_repetition"]["levels"]
    repetition_deltas = {
        label: round(
            candidate_repetition[label]["occurrence_share"]
            - reference_repetition[label]["occurrence_share"],
            6,
        )
        for label, _, _ in _REPETITION_LEVELS
    }
    return {
        "window_tokens": reference.window_tokens,
        "reference": reference_summary,
        "candidate": candidate_summary,
        "eight_token_shape_js_divergence_bits": round(
            _js_divergence(reference.window_shapes, candidate.window_shapes), 6
        ),
        "sentence_opening_js_divergence_bits": round(
            _js_divergence(reference.opening_shapes, candidate.opening_shapes), 6
        ),
        "sentence_closing_js_divergence_bits": round(
            _js_divergence(reference.closing_shapes, candidate.closing_shapes), 6
        ),
        "transition_position_js_divergence_bits": round(
            _js_divergence(
                reference.transition_positions, candidate.transition_positions
            ),
            6,
        ),
        "masked_repetition_level_deltas": repetition_deltas,
        "masked_repetition_total_variation": round(
            sum(abs(value) for value in repetition_deltas.values()) / 2,
            6,
        ),
        "mean_sentence_words_delta": round(
            candidate_summary["mean_sentence_words"]
            - reference_summary["mean_sentence_words"],
            6,
        ),
        "transition_rate_delta": round(
            candidate_summary["transition_rate_per_sentence"]
            - reference_summary["transition_rate_per_sentence"],
            6,
        ),
        "scope": (
            "aggregate coarse-class comparison; lower divergence means a closer "
            "structural distribution, not proof of grammatical correctness; "
            "masked repetition compares frequency levels without source wording"
        ),
        "source_text_retained": False,
        "source_ngrams_retained": False,
    }
