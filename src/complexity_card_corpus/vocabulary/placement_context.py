from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .lexical_sources import (
    _artifact_path,
    _source_role_documents,
    load_lexical_registry,
)
from .lexical_schema import _words


def _context_counts(
    registry_path: Path,
    raw_root: Path,
    candidates: set[str],
    cell_anchors: dict[tuple[str, str], set[str]],
    *,
    window_tokens: int,
) -> tuple[
    dict[str, dict[str, Counter[tuple[str, str]]]],
    Counter[str],
    dict[str, dict[str, Counter[str]]],
]:
    registry = load_lexical_registry(registry_path)
    anchor_cells: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for cell, anchors in cell_anchors.items():
        for anchor in anchors:
            anchor_cells[anchor].add(cell)

    counts: dict[str, dict[str, Counter[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    masked_neighbors: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    documents = Counter()
    for source in registry["sources"]:
        source_id = str(source["dataset_id"])
        for artifact in source["artifacts"]:
            path = _artifact_path(raw_root, source, artifact)
            if not path.exists():
                raise FileNotFoundError(path)
            for _, text in _source_role_documents(path, source):
                documents[source_id] += 1
                words = [token for token, _ in _words(text)]
                anchors_at = [anchor_cells.get(word, set()) for word in words]
                candidate_positions = [
                    (index, word)
                    for index, word in enumerate(words)
                    if word in candidates
                ]
                for index, token in candidate_positions:
                    local: Counter[str] = Counter()
                    start = max(0, index - window_tokens)
                    stop = min(len(words), index + window_tokens + 1)
                    for families in anchors_at[start:stop]:
                        local.update(families)
                    counts[token][source_id].update(local)
                    masked_neighbors[token][source_id].update(
                        neighbor
                        for neighbor in words[start:stop]
                        if neighbor in candidates and neighbor != token
                    )
    return counts, documents, masked_neighbors
