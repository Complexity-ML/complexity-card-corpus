from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .models import Card, CardDataset, Relation


@dataclass(frozen=True)
class RenderedDocument:
    document_id: str
    dataset_id: str
    domain: str
    language: str
    split: str
    template: str
    source_keys: list[str]
    text: str
    source: str
    source_urls: list[str]
    license: str
    version: str


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _relation_label(relation: Relation) -> str:
    return relation.type.replace("_", " ").replace("-", " ")


def _entity_document(card: Card) -> str:
    sections = [card.name, _sentence(card.summary)]
    description = card.description or card.summary
    if description.strip() != card.summary.strip():
        sections.append(_sentence(description))
    if card.facts:
        sections.append("Known facts: " + " ".join(_sentence(fact) for fact in card.facts))
    if card.attributes:
        pairs = [
            f"{key.replace('_', ' ')} is {json.dumps(value, ensure_ascii=False)}"
            for key, value in sorted(card.attributes.items())
        ]
        sections.append("Recorded attributes: " + "; ".join(pairs) + ".")
    if card.aliases:
        sections.append("Also known as: " + ", ".join(card.aliases) + ".")
    return "\n\n".join(sections)


def _neighborhood_document(
    card: Card,
    dataset_id: str,
    cards: dict[tuple[str, str], Card],
) -> str:
    statements = []
    for relation in card.relations:
        target_dataset_id = relation.target_dataset_id or dataset_id
        target = cards[(target_dataset_id, relation.target_key)]
        statement = f"{card.name} {_relation_label(relation)} {target.name}"
        if relation.detail:
            statement += f": {relation.detail.strip()}"
        statements.append(_sentence(statement))
    return (
        f"Connections around {card.name}\n\n"
        + _sentence(card.summary)
        + "\n\n"
        + " ".join(statements)
    )


def _paths_from(
    root_identity: tuple[str, str],
    adjacency: dict[tuple[str, str], list[tuple[Relation, tuple[str, str]]]],
    *,
    max_depth: int,
    limit: int,
) -> Iterable[list[tuple[tuple[str, str], Relation, tuple[str, str]]]]:
    paths: list[list[tuple[tuple[str, str], Relation, tuple[str, str]]]] = []

    def visit(
        current: tuple[str, str],
        path: list[tuple[tuple[str, str], Relation, tuple[str, str]]],
        seen: set[tuple[str, str]],
    ) -> None:
        if len(paths) >= limit:
            return
        if len(path) >= 2:
            paths.append(path.copy())
        if len(path) >= max_depth:
            return
        for relation, target in adjacency.get(current, []):
            if target in seen:
                continue
            path.append((current, relation, target))
            visit(target, path, seen | {target})
            path.pop()

    visit(root_identity, [], {root_identity})
    return paths


def render_documents(
    datasets: list[CardDataset],
    *,
    max_path_depth: int = 3,
    max_paths_per_card: int = 4,
) -> list[RenderedDocument]:
    cards = {
        (dataset.metadata.dataset_id, card.key): card
        for dataset in datasets
        for card in dataset.cards
    }
    adjacency: dict[
        tuple[str, str], list[tuple[Relation, tuple[str, str]]]
    ] = defaultdict(list)
    for dataset in datasets:
        dataset_id = dataset.metadata.dataset_id
        for card in dataset.cards:
            source_identity = (dataset_id, card.key)
            for relation in card.relations:
                target_dataset_id = relation.target_dataset_id or dataset_id
                adjacency[source_identity].append(
                    (relation, (target_dataset_id, relation.target_key))
                )

    documents: list[RenderedDocument] = []
    for dataset in datasets:
        metadata = dataset.metadata

        def add(template: str, suffix: str, source_keys: list[str], text: str) -> None:
            documents.append(
                RenderedDocument(
                    document_id=f"{metadata.dataset_id}:{template}:{suffix}",
                    dataset_id=metadata.dataset_id,
                    domain=metadata.domain,
                    language=metadata.language,
                    split=metadata.split,
                    template=template,
                    source_keys=source_keys,
                    text=text,
                    source=metadata.source,
                    source_urls=metadata.source_urls,
                    license=metadata.license,
                    version=metadata.version,
                )
            )

        for card in sorted(dataset.cards, key=lambda item: item.key):
            add("entity", card.key, [card.key], _entity_document(card))
            if card.relations:
                add(
                    "neighborhood",
                    card.key,
                    [card.key, *[relation.target_key for relation in card.relations]],
                    _neighborhood_document(card, metadata.dataset_id, cards),
                )

            root_identity = (metadata.dataset_id, card.key)
            for path_index, path in enumerate(
                _paths_from(
                    root_identity,
                    adjacency,
                    max_depth=max_path_depth,
                    limit=max_paths_per_card,
                )
            ):
                statements = []
                source_keys = [path[0][0][1]]
                for source_identity, relation, target_identity in path:
                    source_card = cards[source_identity]
                    target_card = cards[target_identity]
                    statements.append(
                        _sentence(
                            f"{source_card.name} {_relation_label(relation)} {target_card.name}"
                        )
                    )
                    source_keys.append(target_identity[1])
                text = (
                    f"Relationship path from {cards[root_identity].name}\n\n"
                    + " ".join(statements)
                )
                add("path", f"{card.key}:{path_index:02d}", source_keys, text)

    return sorted(documents, key=lambda item: item.document_id)

