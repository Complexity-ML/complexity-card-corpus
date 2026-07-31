from __future__ import annotations

import pytest

from complexity_card_corpus.forge import ForgeBlueprint, forge_dataset


def _scale_blueprint(*, creature_count: int = 10_000) -> ForgeBlueprint:
    return ForgeBlueprint.model_validate(
        {
            "format": "complexity-atlas-forge-v1",
            "seed": 777,
            "metadata": {
                "datasetId": "forge-scale-test",
                "title": "Forge Scale Test",
                "domain": "test",
                "themes": ["scale", "creatures"],
                "language": "en",
                "version": "1.0.0",
                "split": "train",
                "source": "Test fixture",
                "sourceUrls": [],
                "license": "CC BY-NC 4.0",
                "description": "A synthetic fixture for exercising Atlas Forge.",
            },
            "archetypes": [
                {
                    "id": "creature",
                    "kind": "creature",
                    "keyPrefix": "creature",
                    "count": creature_count,
                    "slots": {
                        "prefix": [f"Prism{i:02d}" for i in range(100)],
                        "suffix": [f"ling{i:02d}" for i in range(100)],
                        "affinity": ["ember", "tide", "moss", "gale"],
                    },
                    "keySlots": ["prefix", "suffix"],
                    "nameTemplate": "{prefix}{suffix}",
                    "summaryTemplate": "A {affinity}-affinity scale-test creature.",
                    "descriptionTemplate": "Generated deterministically for testing.",
                    "factTemplates": ["Affinity: {affinity}."],
                    "tags": ["scale-test"],
                    "attributeTemplates": {"affinity": "{affinity}"},
                },
                {
                    "id": "habitat",
                    "kind": "habitat",
                    "keyPrefix": "habitat",
                    "count": 100,
                    "slots": {
                        "name": [f"Habitat {index:03d}" for index in range(100)]
                    },
                    "keySlots": ["name"],
                    "nameTemplate": "{name}",
                    "summaryTemplate": "A scale-test habitat.",
                    "descriptionTemplate": "Generated deterministically for testing.",
                },
            ],
            "relationRules": [
                {
                    "sourceArchetype": "creature",
                    "relation": "inhabits",
                    "targetArchetype": "habitat",
                    "stride": 17,
                    "detailTemplate": "{source} inhabits {target}.",
                }
            ],
        }
    )


def test_forge_scales_deterministically() -> None:
    blueprint = _scale_blueprint()
    first = forge_dataset(blueprint)
    second = forge_dataset(blueprint)

    assert len(first.cards) == 10_100
    assert first.model_dump(mode="json") == second.model_dump(mode="json")

    keys = [card.key for card in first.cards]
    assert len(keys) == len(set(keys))
    known_keys = set(keys)
    relation_count = 0
    for card in first.cards:
        for relation in card.relations:
            relation_count += 1
            assert relation.target_key in known_keys
    assert relation_count == 10_000


def test_forge_rejects_insufficient_key_capacity() -> None:
    with pytest.raises(ValueError, match="only 10000 unique combinations"):
        _scale_blueprint(creature_count=10_001)
