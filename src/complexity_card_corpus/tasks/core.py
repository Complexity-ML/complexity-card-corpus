from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskHand:
    """A concrete, solvable hand of cards for one training scenario."""

    data: str
    goal: str
    answer: str
    contract: tuple[str, ...]
    situation_title: str | None = None
    situation: str | None = None
    rule: str | None = None


@dataclass(frozen=True)
class SubcardPool:
    """Named reservoir of interchangeable subcards with the same semantic role."""

    name: str
    cards: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a subcard pool requires a name")
        if not self.cards or not any(card.strip() for card in self.cards):
            raise ValueError("a subcard pool requires at least one visible card")
        if len(set(self.cards)) != len(self.cards):
            raise ValueError("a subcard pool cannot contain duplicate cards")


@dataclass(frozen=True)
class DeckTopology:
    """Compact, serializable shape of one linked subcard deck."""

    name: str
    pool_names: tuple[str, ...]
    pool_sizes: tuple[int, ...]
    link_counts: tuple[int, ...]


class DealtCard(str):
    """Rendered card carrying the linked-deck topology that produced it."""

    deck_name: str
    deck_lineage: tuple[str, ...]
    deck_topologies: tuple[DeckTopology, ...]
    pool_names: tuple[str, ...]
    compatibility_links: tuple[tuple[tuple[int, int], ...], ...]

    def __new__(
        cls,
        value: str,
        *,
        deck_name: str,
        deck_topologies: tuple[DeckTopology, ...],
        pool_names: tuple[str, ...],
        compatibility_links: tuple[tuple[tuple[int, int], ...], ...],
    ) -> DealtCard:
        card = super().__new__(cls, value)
        card.deck_name = deck_name
        card.deck_lineage = tuple(topology.name for topology in deck_topologies)
        card.deck_topologies = deck_topologies
        card.pool_names = pool_names
        card.compatibility_links = compatibility_links
        return card


@dataclass(frozen=True)
class LinkedSubcardDeck:
    """Ordered subcard reservoirs joined by explicit compatibility edges."""

    pools: tuple[SubcardPool, ...]
    links: tuple[tuple[tuple[int, int], ...], ...] = ()
    cycle_first_pool: bool = False

    def __post_init__(self) -> None:
        if not self.pools:
            raise ValueError("a linked subcard deck requires at least one pool")
        names = tuple(pool.name for pool in self.pools)
        if len(set(names)) != len(names):
            raise ValueError("subcard pool names must be unique inside a deck")
        if self.links and len(self.links) != len(self.pools) - 1:
            raise ValueError("subcard links must connect every adjacent pool pair")
        for slot_index, edges in enumerate(self.compatibility_links()):
            left_size = len(self.pools[slot_index].cards)
            right_size = len(self.pools[slot_index + 1].cards)
            if any(
                left < 0
                or left >= left_size
                or right < 0
                or right >= right_size
                for left, right in edges
            ):
                raise ValueError("subcard link references an unknown node")
            if {left for left, _right in edges} != set(range(left_size)):
                raise ValueError("a subcard node has no outgoing compatible link")
            if {_right for _left, _right in edges} != set(range(right_size)):
                raise ValueError("a subcard node has no incoming compatible link")

    def compatibility_links(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        if self.links:
            return self.links
        return tuple(
            tuple(
                (left, right)
                for left in range(len(self.pools[index].cards))
                for right in range(len(self.pools[index + 1].cards))
            )
            for index in range(len(self.pools) - 1)
        )

    @property
    def pool_names(self) -> tuple[str, ...]:
        return tuple(pool.name for pool in self.pools)

    def deal(self, row: dict[str, Any], variant: int, deck: str) -> tuple[str, ...]:
        current = (
            variant % len(self.pools[0].cards)
            if self.cycle_first_pool
            else _number(
                f"{deck}:{self.pools[0].name}:{row['scenario_id']}:{variant}",
                0,
                len(self.pools[0].cards) - 1,
            )
        )
        path = [self.pools[0].cards[current]]
        for slot_index, edges in enumerate(self.compatibility_links(), start=1):
            candidates = tuple(right for left, right in edges if left == current)
            current = candidates[
                _number(
                    f"{deck}:{self.pools[slot_index].name}-link:"
                    f"{row['scenario_id']}:{variant}",
                    0,
                    len(candidates) - 1,
                )
            ]
            path.append(self.pools[slot_index].cards[current])
        return tuple(path)


def _number(key: str, low: int, high: int) -> int:
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return low + value % (high - low + 1)


def _pick(key: str, values: tuple[str, ...]) -> str:
    return values[_number(key, 0, len(values) - 1)]


def _deal_task_frames(
    row: dict[str, Any],
    variant: int,
    deck: str,
    data_cards: tuple[str, ...],
    goal_cards: tuple[str, ...],
) -> tuple[str, str]:
    """Deal independent input and objective cards for one task family."""
    return (
        _compose_subcards(
            row,
            variant,
            f"{deck}-data",
            (data_cards,),
            pool_names=("data",),
        ),
        _compose_subcards(
            row,
            variant,
            f"{deck}-goal",
            (goal_cards,),
            pool_names=("goal",),
        ),
    )


def _compose_subcards(
    row: dict[str, Any],
    variant: int,
    deck: str,
    slots: tuple[tuple[str, ...], ...],
    *,
    pool_names: tuple[str, ...],
    links: tuple[tuple[tuple[int, int], ...], ...] = (),
    cycle_first_pool: bool = False,
    separator: str = " ",
) -> DealtCard:
    """Compose one card by walking compatible links between named reservoirs."""
    if len(pool_names) != len(slots):
        raise ValueError("every subcard reservoir requires exactly one semantic name")
    pools = tuple(
        SubcardPool(name=name, cards=cards)
        for name, cards in zip(pool_names, slots, strict=True)
    )
    linked_deck = LinkedSubcardDeck(
        pools,
        links,
        cycle_first_pool=cycle_first_pool,
    )
    nested_topologies = tuple(
        topology
        for pool in pools
        for card in pool.cards
        if isinstance(card, DealtCard)
        for topology in card.deck_topologies
    )
    own_topology = DeckTopology(
        name=deck,
        pool_names=linked_deck.pool_names,
        pool_sizes=tuple(len(pool.cards) for pool in pools),
        link_counts=tuple(
            len(edges) for edges in linked_deck.compatibility_links()
        ),
    )
    deck_topologies = tuple(dict.fromkeys((own_topology, *nested_topologies)))
    pieces = (
        piece.strip() for piece in linked_deck.deal(row, variant, deck)
    )
    return DealtCard(
        separator.join(piece for piece in pieces if piece),
        deck_name=deck,
        deck_topologies=deck_topologies,
        pool_names=linked_deck.pool_names,
        compatibility_links=linked_deck.compatibility_links(),
    )


def _code(row: dict[str, Any]) -> str:
    return row["scenario_id"].split(":")[-1][:6].upper()


def _payload(row: dict[str, Any]) -> dict[str, str]:
    return json.loads(row["semantic_payload"])


def _lower_sentence_initial(value: str) -> str:
    """Lower a sentence initial without corrupting an acronym such as RAM."""
    initial = re.match(r"[A-Za-z]+", value)
    if initial is None or initial.group(0).isupper():
        return value
    return value[:1].lower() + value[1:]
