from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .build import file_sha256


VOCABULARY_GAP_VERSION = "cross-source-vocabulary-gap-v1"
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")

# Function words help surface statistics but not a vocabulary-authoring queue.
AUTHORING_STOPWORDS = frozenset(
    """
    about after again against all also am an and any are as at be because been
    before being both but by can could did do does doing down during each few
    for from further had has have having he her here hers herself him himself
    his how i if in into is it its itself just me more most my myself no nor
    not now of off on once only or other our ours ourselves out over own same
    she should so some such than that the their theirs them themselves then
    there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours
    yourself yourselves
    """.split()
)


def _normalized_words(value: str) -> list[str]:
    return [
        match.group(0).replace("’", "'").lower()
        for match in WORD_PATTERN.finditer(value)
    ]


def _conversation_vocabulary(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=2_048, columns=["messages"]
    ):
        for row in batch.to_pylist():
            for message in row.get("messages") or []:
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, str):
                    counts.update(_normalized_words(content))
    return counts


def build_vocabulary_gap(
    lexicon_path: Path,
    conversations_path: Path,
    output_dir: Path,
    *,
    min_sources: int = 2,
    min_occurrences_per_source: int = 20,
    max_candidates: int = 4_000,
) -> dict[str, Any]:
    """Build a human-review queue of mined words absent from the corpus.

    The mine contains normalized single tokens only. This function cannot see
    source sentences and never inserts a candidate into a generated card.
    """
    if min_sources < 2:
        raise ValueError("min_sources must be at least 2")
    if min_occurrences_per_source < 1:
        raise ValueError("min_occurrences_per_source must be positive")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")

    current = _conversation_vocabulary(conversations_path)
    source_counts: dict[str, dict[str, int]] = defaultdict(dict)
    roles: dict[str, Counter[str]] = defaultdict(Counter)

    for batch in pq.ParquetFile(lexicon_path).iter_batches(batch_size=8_192):
        for row in batch.to_pylist():
            if row.get("mined_unit") != "single_normalized_token":
                raise ValueError("vocabulary gap input must contain single tokens only")
            if row.get("source_text_retained") is not False:
                raise ValueError("vocabulary gap input must not retain source text")
            token = str(row["token"])
            source = str(row["source_dataset"])
            occurrences = int(row["occurrences"])
            source_counts[token][source] = (
                source_counts[token].get(source, 0) + occurrences
            )
            roles[token][str(row["role"])] += occurrences

    candidates: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for token, sources in source_counts.items():
        if token in current:
            rejection_counts["already_observed"] += 1
            continue
        if token in AUTHORING_STOPWORDS:
            rejection_counts["function_word"] += 1
            continue
        if len(token) < 3:
            rejection_counts["too_short"] += 1
            continue
        supported = {
            source: count
            for source, count in sources.items()
            if count >= min_occurrences_per_source
        }
        if len(supported) < min_sources:
            rejection_counts["insufficient_cross_source_support"] += 1
            continue
        role_counts = roles[token]
        candidates.append(
            {
                "token": token,
                "roles": ";".join(
                    role
                    for role, _ in sorted(
                        role_counts.items(), key=lambda item: (-item[1], item[0])
                    )
                ),
                "source_count": len(supported),
                "sources": ";".join(sorted(supported)),
                "total_occurrences": sum(supported.values()),
                "minimum_source_occurrences": min(supported.values()),
                "current_occurrences": 0,
                "review_status": "pending",
                "target_family": "",
                "target_domain": "",
                "authoring_notes": "",
            }
        )

    candidates.sort(
        key=lambda row: (
            -int(row["source_count"]),
            -int(row["minimum_source_occurrences"]),
            -int(row["total_occurrences"]),
            str(row["token"]),
        )
    )
    eligible_count = len(candidates)
    candidates = candidates[:max_candidates]

    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "vocabulary_review.csv"
    fieldnames = list(candidates[0]) if candidates else [
        "token",
        "roles",
        "source_count",
        "sources",
        "total_occurrences",
        "minimum_source_occurrences",
        "current_occurrences",
        "review_status",
        "target_family",
        "target_domain",
        "authoring_notes",
    ]
    with review_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    role_candidates: Counter[str] = Counter()
    for row in candidates:
        for role in str(row["roles"]).split(";"):
            if role:
                role_candidates[role] += 1
    audit = {
        "version": VOCABULARY_GAP_VERSION,
        "current_word_occurrences": sum(current.values()),
        "current_observed_vocabulary": len(current),
        "min_sources": min_sources,
        "min_occurrences_per_source": min_occurrences_per_source,
        "eligible_cross_source_gaps": eligible_count,
        "review_rows": len(candidates),
        "review_status": "pending",
        "role_candidate_counts": dict(sorted(role_candidates.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "max_retained_ngram_tokens": 1,
        "source_text_retained": False,
        "automatic_surface_insertion": False,
        "release_ready": False,
        "scope": (
            "cross-source single-token authoring queue; human semantic review "
            "and original sentence composition required"
        ),
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "format": VOCABULARY_GAP_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "lexicon": {
                "path": str(lexicon_path),
                "sha256": file_sha256(lexicon_path),
            },
            "conversations": {
                "path": str(conversations_path),
                "sha256": file_sha256(conversations_path),
            },
        },
        "files": {
            "review": {
                "path": review_path.name,
                "sha256": file_sha256(review_path),
            },
            "audit": {
                "path": "audit.json",
                "sha256": file_sha256(output_dir / "audit.json"),
            },
        },
        "audit": audit,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
