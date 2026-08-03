from __future__ import annotations

import hashlib
import re
import statistics
from collections import Counter
from typing import Any

from .english_morphology import correct_indefinite_articles
from .scenario_language import QUESTION_FRAME_IDS, uses_question_surface


SURFACE_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
TRANSITIONS = frozenset(
    {
        "after", "although", "before", "finally", "first", "following",
        "however", "instead", "meanwhile", "next", "once", "otherwise",
        "then", "therefore", "unless", "until", "when", "while",
    }
)


def _tokens(text: str) -> list[str]:
    return [
        match.group(0).replace("’", "'").lower()
        for match in SURFACE_WORD.finditer(text)
    ]


def _normalized(text: str) -> str:
    return " ".join(_tokens(text))


def _mattr(tokens: list[str], window: int = 100) -> float:
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    counts = Counter(tokens[:window])
    total = len(counts) / window
    windows = 1
    for index in range(window, len(tokens)):
        outgoing = tokens[index - window]
        counts[outgoing] -= 1
        if counts[outgoing] == 0:
            del counts[outgoing]
        counts[tokens[index]] += 1
        total += len(counts) / window
        windows += 1
    return total / windows


def scenario_surface_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [str(row["situation"]) for row in rows]
    tokenized = [_tokens(text) for text in texts]
    lengths = [len(tokens) for tokens in tokenized]
    vocabulary = Counter(token for tokens in tokenized for token in tokens)
    sentence_hashes: set[bytes] = set()
    sentence_count = 0
    sentence_lengths: list[int] = []
    transition_count = 0
    for text in texts:
        for sentence in re.split(r"[.!?]+", text):
            tokens = _tokens(sentence)
            if tokens:
                sentence_count += 1
                sentence_lengths.append(len(tokens))
                transition_count += sum(token in TRANSITIONS for token in tokens)
                sentence_hashes.add(
                    hashlib.sha256(" ".join(tokens).encode()).digest()
                )
    occurrences = sum(lengths)
    all_tokens = [token for tokens in tokenized for token in tokens]
    ordered = sorted(lengths)
    p95_index = min(len(ordered) - 1, round((len(ordered) - 1) * 0.95))
    return {
        "documents": len(texts),
        "word_occurrences": occurrences,
        "observed_vocabulary": len(vocabulary),
        "mean_words": round(statistics.fmean(lengths), 3),
        "median_words": round(statistics.median(lengths), 3),
        "p95_words": float(ordered[p95_index]),
        "question_rate": round(
            sum(text.rstrip().endswith("?") for text in texts) / len(texts), 6
        ),
        "raw_type_token_ratio": round(len(vocabulary) / occurrences, 6),
        "mattr_100": round(_mattr(all_tokens, 100), 6),
        "unique_document_rate": round(len(set(texts)) / len(texts), 6),
        "unique_sentence_rate": round(len(sentence_hashes) / sentence_count, 6),
        "mean_sentence_words": round(statistics.fmean(sentence_lengths), 6),
        "transitions_per_sentence": round(transition_count / sentence_count, 6),
    }


def audit_scenario_surface(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Lint every composed surface without claiming full linguistic proof."""
    issues: list[dict[str, str]] = []
    anchors_checked = 0
    anchors_matched = 0
    frame_family_cells: set[tuple[str, str]] = set()
    malformed_spacing = re.compile(r" {2,}|\s+[,:;.!?]|[,:;]{2,}|[.!?]{2,}")
    repeated_word = re.compile(r"\b([a-z]+)\s+\1\b", re.IGNORECASE)
    suspect_sequences = (
        "it makes acknowledge",
        "it makes arrange",
        "it makes clarify",
        "it makes compare",
        "it makes diagnose",
        "it makes explain",
        "it makes preserve",
        "it makes reflect",
        "it makes revise",
        "it makes summarize",
    )

    for row in rows:
        scenario_id = str(row["scenario_id"])
        text = str(row["situation"]).strip()
        normalized = _normalized(text)
        frame_family_cells.add((str(row["family"]), str(row["narrative_frame"])))

        if not text.endswith((".", "?")):
            issues.append({"scenario_id": scenario_id, "kind": "terminal_punctuation"})
        sentence_parts = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
        expected_sentences = 6
        if len(sentence_parts) != expected_sentences:
            issues.append({"scenario_id": scenario_id, "kind": "sentence_count"})
        if any(part[:1] and not part[:1].isupper() for part in sentence_parts):
            issues.append({"scenario_id": scenario_id, "kind": "sentence_capitalization"})
        if malformed_spacing.search(text):
            issues.append({"scenario_id": scenario_id, "kind": "punctuation_spacing"})
        # Inspect the rendered surface rather than the punctuation-stripped
        # normalization. Otherwise a sentence ending in ``result`` followed
        # by a sentence beginning with ``Result`` is reported as the false
        # adjacent pair ``result result``.
        if repeated_word.search(text):
            issues.append({"scenario_id": scenario_id, "kind": "adjacent_word_repeat"})
        if any(sequence in normalized for sequence in suspect_sequences):
            issues.append({"scenario_id": scenario_id, "kind": "suspect_verb_chain"})
        if correct_indefinite_articles(text) != text:
            issues.append({"scenario_id": scenario_id, "kind": "indefinite_article"})

        should_question = uses_question_surface(row["narrative_frame"])
        if should_question != text.endswith("?") or text.count("?") != int(should_question):
            issues.append({"scenario_id": scenario_id, "kind": "question_contract"})

        for field in ("state", "constraint", "desired_outcome"):
            anchors_checked += 1
            if _normalized(str(row[field])) in normalized:
                anchors_matched += 1
            else:
                issues.append(
                    {"scenario_id": scenario_id, "kind": f"missing_{field}_anchor"}
                )
        for field in ("state", "constraint"):
            anchor = _normalized(str(row[field]))
            if normalized.count(anchor) != 1:
                issues.append(
                    {"scenario_id": scenario_id, "kind": f"repeated_{field}_anchor"}
                )

    return {
        "checked_rows": len(rows),
        "issues": issues,
        "issue_count": len(issues),
        "semantic_anchors_checked": anchors_checked,
        "semantic_anchor_match_rate": round(anchors_matched / anchors_checked, 6),
        "question_frames": sorted(QUESTION_FRAME_IDS),
        "frame_family_cells": len(frame_family_cells),
        "stats": scenario_surface_stats(rows),
        "scope": (
            "deterministic composition lint; punctuation, template contracts and "
            "semantic anchors, not a general grammar-model score"
        ),
    }
