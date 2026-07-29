from __future__ import annotations

import json
from pathlib import Path

from .models import Card, CardDataset, DatasetMetadata


def load_dataset(directory: Path) -> CardDataset:
    metadata_path = directory / "dataset.json"
    cards_path = directory / "cards.json"
    if not metadata_path.exists() or not cards_path.exists():
        raise FileNotFoundError(f"{directory} must contain dataset.json and cards.json")

    metadata = DatasetMetadata.model_validate_json(metadata_path.read_text())
    raw_cards = json.loads(cards_path.read_text())
    if not isinstance(raw_cards, list) or not raw_cards:
        raise ValueError(f"{cards_path} must contain a non-empty JSON list")
    cards = [Card.model_validate(card) for card in raw_cards]
    return CardDataset(metadata=metadata, cards=cards)


def discover_datasets(source_root: Path) -> list[CardDataset]:
    directories = sorted(
        path.parent for path in source_root.rglob("dataset.json") if path.parent.is_dir()
    )
    if not directories:
        raise FileNotFoundError(f"No dataset.json found below {source_root}")
    datasets = [load_dataset(directory) for directory in directories]
    validate_graph(datasets)
    return datasets


def validate_graph(datasets: list[CardDataset]) -> None:
    cards_by_identity: dict[tuple[str, str], Card] = {}
    for dataset in datasets:
        dataset_id = dataset.metadata.dataset_id
        for card in dataset.cards:
            identity = (dataset_id, card.key)
            if identity in cards_by_identity:
                raise ValueError(f"Duplicate card identity: {dataset_id}/{card.key}")
            cards_by_identity[identity] = card

    for dataset in datasets:
        dataset_id = dataset.metadata.dataset_id
        for card in dataset.cards:
            for relation in card.relations:
                target_dataset_id = relation.target_dataset_id or dataset_id
                target = (target_dataset_id, relation.target_key)
                if target not in cards_by_identity:
                    raise ValueError(
                        f"Missing relation target: {dataset_id}/{card.key} -> "
                        f"{target_dataset_id}/{relation.target_key}"
                    )

