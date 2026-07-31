from __future__ import annotations

import json
import hashlib
import math
import re
import string
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import Card, CardDataset, DatasetMetadata, Relation

SLUG_RE = re.compile(r"[^a-z0-9]+")
FORMATTER = string.Formatter()


def _template_fields(template: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, _, _ in FORMATTER.parse(template):
        if not field_name:
            continue
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        fields.add(root)
    return fields


class ArchetypeBlueprint(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    archetype_id: str = Field(alias="id")
    kind: str
    key_prefix: str = Field(alias="keyPrefix")
    count: int = Field(gt=0)
    slots: dict[str, list[Any]]
    key_slots: list[str] = Field(default_factory=list, alias="keySlots")
    name_template: str = Field(alias="nameTemplate")
    alias_templates: list[str] = Field(default_factory=list, alias="aliasTemplates")
    summary_template: str = Field(alias="summaryTemplate")
    description_template: str = Field(alias="descriptionTemplate")
    fact_templates: list[str] = Field(default_factory=list, alias="factTemplates")
    tags: list[str] = Field(default_factory=list)
    attribute_templates: dict[str, Any] = Field(
        default_factory=dict,
        alias="attributeTemplates",
    )

    @field_validator("archetype_id", "kind", "key_prefix")
    @classmethod
    def required_identifier(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("forge identifiers cannot be empty")
        return value

    @field_validator("slots")
    @classmethod
    def populated_slots(cls, slots: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if not slots:
            raise ValueError("each archetype needs at least one slot")
        empty = [name for name, values in slots.items() if not values]
        if empty:
            raise ValueError(f"forge slots cannot be empty: {', '.join(empty)}")
        return slots

    @model_validator(mode="after")
    def scalable_keys(self) -> "ArchetypeBlueprint":
        if len(self.key_slots) != len(set(self.key_slots)):
            raise ValueError("keySlots cannot contain duplicates")
        unknown = [name for name in self.key_slots if name not in self.slots]
        if unknown:
            raise ValueError(
                f"keySlots reference unknown slots: {', '.join(unknown)}"
            )
        name_fields = _template_fields(self.name_template)
        unused = [name for name in self.key_slots if name not in name_fields]
        if unused:
            raise ValueError(
                "every keySlot must appear in nameTemplate: "
                + ", ".join(unused)
            )
        if self.key_slots and "index" not in name_fields:
            capacity = math.prod(len(self.slots[name]) for name in self.key_slots)
            if self.count > capacity:
                raise ValueError(
                    f"archetype {self.archetype_id!r} requests {self.count} cards "
                    f"but keySlots provide only {capacity} unique combinations"
                )
        return self


class RelationRule(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    source_archetype: str = Field(alias="sourceArchetype")
    relation: str
    target_archetype: str = Field(alias="targetArchetype")
    offset: int = 0
    stride: int = Field(default=1, gt=0)
    detail_template: str | None = Field(default=None, alias="detailTemplate")


class ForgeBlueprint(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    format: str
    seed: int = 0
    metadata: DatasetMetadata
    archetypes: list[ArchetypeBlueprint]
    relation_rules: list[RelationRule] = Field(
        default_factory=list,
        alias="relationRules",
    )

    @model_validator(mode="after")
    def valid_graph_recipe(self) -> "ForgeBlueprint":
        if self.format != "complexity-atlas-forge-v1":
            raise ValueError("unsupported Atlas Forge blueprint format")
        archetype_ids = [item.archetype_id for item in self.archetypes]
        if len(archetype_ids) != len(set(archetype_ids)):
            raise ValueError("archetype ids must be unique")
        known = set(archetype_ids)
        for rule in self.relation_rules:
            if rule.source_archetype not in known:
                raise ValueError(
                    f"unknown source archetype: {rule.source_archetype}"
                )
            if rule.target_archetype not in known:
                raise ValueError(
                    f"unknown target archetype: {rule.target_archetype}"
                )
        return self


def _slug(value: str) -> str:
    slug = SLUG_RE.sub("_", value.lower()).strip("_")
    if not slug:
        raise ValueError(f"cannot derive a key from name {value!r}")
    return slug


def _format(value: Any, slots: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(slots)
    if isinstance(value, list):
        return [_format(item, slots) for item in value]
    if isinstance(value, dict):
        return {key: _format(item, slots) for key, item in value.items()}
    return value


def _slots_for(
    archetype: ArchetypeBlueprint,
    index: int,
    seed: int,
) -> dict[str, Any]:
    slots: dict[str, Any] = {"index": index + 1}
    if archetype.key_slots:
        capacity = math.prod(
            len(archetype.slots[name]) for name in archetype.key_slots
        )
        stride = _coprime_stride(capacity, seed)
        position = (seed + index * stride) % capacity
        for name in archetype.key_slots:
            values = archetype.slots[name]
            slots[name] = values[position % len(values)]
            position //= len(values)

    for slot_index, (name, values) in enumerate(sorted(archetype.slots.items())):
        if name in slots:
            continue
        if archetype.key_slots:
            position = _stable_slot_position(
                seed=seed,
                archetype_id=archetype.archetype_id,
                index=index,
                slot_name=name,
                size=len(values),
            )
        else:
            position = (
                seed
                + index * (2 * slot_index + 1)
                + slot_index * slot_index
            ) % len(values)
        slots[name] = values[position]
    return slots


def _stable_slot_position(
    *,
    seed: int,
    archetype_id: str,
    index: int,
    slot_name: str,
    size: int,
) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{archetype_id}:{index}:{slot_name}".encode(),
        digest_size=8,
        person=b"atlas-v1",
    ).digest()
    return int.from_bytes(digest, "little") % size


def _coprime_stride(capacity: int, seed: int) -> int:
    if capacity <= 1:
        return 1
    candidate = (abs(seed) * 2 + 1) % capacity
    if candidate == 0:
        candidate = 1
    while math.gcd(candidate, capacity) != 1:
        candidate = (candidate + 1) % capacity
        if candidate == 0:
            candidate = 1
    return candidate


def forge_dataset(blueprint: ForgeBlueprint) -> CardDataset:
    generated: dict[str, list[Card]] = {}
    for archetype in blueprint.archetypes:
        cards: list[Card] = []
        for index in range(archetype.count):
            slots = _slots_for(archetype, index, blueprint.seed)
            name = _format(archetype.name_template, slots)
            card = Card(
                key=f"{archetype.key_prefix}:{_slug(name)}",
                kind=archetype.kind,
                name=name,
                aliases=[
                    _format(template, slots)
                    for template in archetype.alias_templates
                ],
                summary=_format(archetype.summary_template, slots),
                description=_format(archetype.description_template, slots),
                facts=[
                    _format(template, slots)
                    for template in archetype.fact_templates
                ],
                tags=list(archetype.tags),
                attributes=_format(archetype.attribute_templates, slots),
                relations=[],
            )
            cards.append(card)
        keys = [card.key for card in cards]
        if len(keys) != len(set(keys)):
            raise ValueError(
                f"archetype {archetype.archetype_id!r} generated duplicate keys; "
                "expand its name slots or reduce count"
            )
        generated[archetype.archetype_id] = cards

    for rule in blueprint.relation_rules:
        sources = generated[rule.source_archetype]
        targets = generated[rule.target_archetype]
        for index, source in enumerate(sources):
            target = targets[(rule.offset + index * rule.stride) % len(targets)]
            source.relations.append(
                Relation(
                    type=rule.relation,
                    targetKey=target.key,
                    detail=(
                        rule.detail_template.format(
                            source=source.name,
                            target=target.name,
                            index=index + 1,
                        )
                        if rule.detail_template
                        else None
                    ),
                )
            )

    cards = [
        card
        for archetype in blueprint.archetypes
        for card in generated[archetype.archetype_id]
    ]
    keys = [card.key for card in cards]
    if len(keys) != len(set(keys)):
        raise ValueError("the thematic deck generated duplicate card keys")
    return CardDataset(metadata=blueprint.metadata, cards=cards)


def load_blueprint(path: Path) -> ForgeBlueprint:
    return ForgeBlueprint.model_validate(json.loads(path.read_text()))


def write_forged_dataset(
    blueprint_path: Path,
    output_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    blueprint = load_blueprint(blueprint_path)
    dataset = forge_dataset(blueprint)
    dataset_path = output_root / "dataset.json"
    cards_path = output_root / "cards.json"
    if not force and (dataset_path.exists() or cards_path.exists()):
        raise FileExistsError(
            f"{output_root} already contains a forged dataset; pass --force to replace it"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps(
            dataset.metadata.model_dump(mode="json", by_alias=True),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    cards_path.write_text(
        json.dumps(
            [card.model_dump(mode="json", by_alias=True) for card in dataset.cards],
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return {
        "dataset_id": dataset.metadata.dataset_id,
        "domain": dataset.metadata.domain,
        "themes": dataset.metadata.themes,
        "cards": len(dataset.cards),
        "relations": sum(len(card.relations) for card in dataset.cards),
        "output": str(output_root.resolve()),
    }
