from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from ..vocabulary_gap import AUTHORING_STOPWORDS
from .lexical_schema import _words


VOCABULARY_PLACEMENT_VERSION = "statistical-vocabulary-placement-v1"


MASKED_DICTIONARY_VERSION = "masked-context-dictionary-v1"


FAMILY_ANCHORS: dict[str, frozenset[str]] = {
    "practical_action": frozenset(
        "action build change choose create install make organize prepare use".split()
    ),
    "explanation_learning": frozenset(
        "concept definition example explain learn lesson mechanism principle teach".split()
    ),
    "troubleshooting": frozenset(
        "bug diagnose error fail failure fix issue problem recover test".split()
    ),
    "writing_transformation": frozenset(
        "draft edit paragraph rewrite sentence style summarize text tone write".split()
    ),
    "planning_comparison": frozenset(
        "budget compare cost criteria decision option plan priority schedule tradeoff".split()
    ),
    "conversation_empathy": frozenset(
        "emotion empathy feel feeling friend listen relationship support understand".split()
    ),
    "safety_uncertainty": frozenset(
        "danger emergency harm health legal medical privacy risk safe safety".split()
    ),
    "grounded_qa": frozenset(
        "answer citation document evidence fact question quote record source verify".split()
    ),
    "summarization_synthesis": frozenset(
        "decision key main notes overview points report summary synthesize theme".split()
    ),
    "extraction_classification": frozenset(
        "category classify column extract field label record schema structure value".split()
    ),
    "reasoning_verification": frozenset(
        "calculate check equation logic premise proof reason result solve verify".split()
    ),
    "critique_revision": frozenset(
        "argument critique evidence improve issue revise revision weak weakness writing".split()
    ),
    "brainstorming_creativity": frozenset(
        "brainstorm combine creative design generate idea imagine novel option".split()
    ),
    "context_clarification": frozenset(
        "ambiguous clarify clarification context intent mean missing request scope".split()
    ),
}


ROLE_PRIORS: dict[str, dict[str, float]] = {
    "intent_term": {
        "practical_action": 1.0,
        "troubleshooting": 0.9,
        "planning_comparison": 0.8,
        "reasoning_verification": 0.8,
        "critique_revision": 0.7,
        "brainstorming_creativity": 0.8,
    },
    "state_term": {
        "troubleshooting": 1.0,
        "conversation_empathy": 1.0,
        "safety_uncertainty": 0.9,
        "context_clarification": 0.9,
    },
    "constraint_term": {
        "safety_uncertainty": 1.0,
        "grounded_qa": 0.9,
        "extraction_classification": 0.8,
        "planning_comparison": 0.7,
        "context_clarification": 0.8,
    },
    "outcome_term": {
        "summarization_synthesis": 1.0,
        "writing_transformation": 0.9,
        "explanation_learning": 0.8,
        "reasoning_verification": 0.8,
    },
    "transition": {
        "summarization_synthesis": 1.0,
        "planning_comparison": 0.8,
        "reasoning_verification": 0.8,
        "writing_transformation": 0.7,
    },
}


def _candidate_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["token"]: row for row in csv.DictReader(stream)}
    if not rows:
        raise ValueError("vocabulary review contains no candidates")
    return rows


def _scenario_cells(
    scenarios_path: Path,
) -> tuple[Counter[tuple[str, str]], dict[tuple[str, str], set[str]]]:
    rows = pq.read_table(scenarios_path).to_pylist()
    counts: Counter[tuple[str, str]] = Counter()
    raw_anchors: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        cell = (str(row["family"]), str(row["domain"]))
        counts[cell] += 1
        payload = json.loads(row["semantic_payload"])
        values = [
            row["trigger"],
            row["state"],
            row["constraint"],
            row["desired_outcome"],
            *[value for value in payload.values() if isinstance(value, str)],
        ]
        for value in values:
            raw_anchors[cell].update(
                token
                for token, _ in _words(str(value))
                if len(token) >= 4 and token not in AUTHORING_STOPWORDS
            )
        raw_anchors[cell].update(FAMILY_ANCHORS[cell[0]])

    cell_frequency: Counter[str] = Counter()
    for anchors in raw_anchors.values():
        cell_frequency.update(anchors.keys())
    maximum_cells = max(5, math.floor(len(counts) * 0.20))
    anchors = {
        cell: {
            token
            for token, occurrences in values.items()
            if occurrences >= 2 and cell_frequency[token] <= maximum_cells
        }
        for cell, values in raw_anchors.items()
    }
    if any(not values for values in anchors.values()):
        raise ValueError("a scenario family/domain cell has no usable anchors")
    return counts, anchors
