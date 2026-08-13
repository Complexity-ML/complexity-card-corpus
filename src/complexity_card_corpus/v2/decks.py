from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .contracts import RoleSeparatedVariableBy, SurfaceRole


_WORD = re.compile(r"[a-z0-9']+", re.I)


def _index(seed: str, size: int) -> int:
    value = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    return value % size


def _literal_ngrams(template: str, size: int = 3) -> set[tuple[str, ...]]:
    literal = re.sub(r"\{[^{}]+\}", " ", template)
    words = _WORD.findall(literal.casefold())
    return set(zip(*(words[index:] for index in range(size))))


@dataclass(frozen=True)
class V2SubcardPool:
    """Interchangeable surface cards belonging to one model-facing role."""

    name: str
    role: SurfaceRole
    cards: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a V2 subcard pool requires a visible name")
        if not self.cards or not all(card.strip() for card in self.cards):
            raise ValueError(f"V2 subcard pool {self.name!r} is empty")
        if len(self.cards) != len(set(self.cards)):
            raise ValueError(f"V2 subcard pool {self.name!r} contains duplicates")


@dataclass(frozen=True)
class V2DealtPair:
    prompt: str
    thinking: str
    answer: str
    prompt_subcards: tuple[str, ...]
    thinking_subcards: tuple[str, ...]
    answer_subcards: tuple[str, ...]
    variable_indices: dict[str, dict[str, int]]
    variable_card_counts: dict[str, dict[str, int]]
    dependency_graph: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class V2RoleSeparatedDeck:
    """Compose prompts and answers from disjoint, role-checked subcard pools."""

    name: str
    variables: RoleSeparatedVariableBy
    prompt_pools: tuple[V2SubcardPool, ...]
    answer_pools: tuple[V2SubcardPool, ...]
    thinking_pools: tuple[V2SubcardPool, ...] = ()
    separator: str = " "

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a V2 deck requires a visible name")
        if not self.prompt_pools or not self.answer_pools:
            raise ValueError("a V2 deck needs prompt and answer subcards")
        pools = (*self.prompt_pools, *self.thinking_pools, *self.answer_pools)
        names = tuple(pool.name for pool in pools)
        if len(names) != len(set(names)):
            raise ValueError("V2 subcard pool names must be unique inside a deck")
        if any(pool.role is not SurfaceRole.PROMPT for pool in self.prompt_pools):
            raise ValueError("prompt pools must carry the prompt role")
        if any(pool.role is not SurfaceRole.ANSWER for pool in self.answer_pools):
            raise ValueError("answer pools must carry the answer role")
        if any(pool.role is not SurfaceRole.THINKING for pool in self.thinking_pools):
            raise ValueError("thinking pools must carry the thinking role")
        self.variables.validate(
            SurfaceRole.PROMPT,
            tuple(card for pool in self.prompt_pools for card in pool.cards),
        )
        self.variables.validate(
            SurfaceRole.ANSWER,
            tuple(card for pool in self.answer_pools for card in pool.cards),
        )
        if self.thinking_pools:
            if SurfaceRole.THINKING not in self.variables.matrix.table:
                raise ValueError("thinking subcards require a thinking VariableBy2D axis")
            self.variables.validate(
                SurfaceRole.THINKING,
                tuple(card for pool in self.thinking_pools for card in pool.cards),
            )
        prompt_surfaces = tuple(
            card
            for cards in self.variables.matrix.table[SurfaceRole.PROMPT].values()
            for card in cards
        )
        answer_surfaces = tuple(
            card
            for cards in self.variables.matrix.table[SurfaceRole.ANSWER].values()
            for card in cards
        )
        thinking_surfaces = tuple(
            card
            for cards in self.variables.matrix.table.get(
                SurfaceRole.THINKING, {}
            ).values()
            for card in cards
        )
        prompt_grams = set().union(
            *(
                _literal_ngrams(card)
                for card in (
                    *prompt_surfaces,
                    *(card for pool in self.prompt_pools for card in pool.cards),
                )
            )
        )
        answer_grams = set().union(
            *(
                _literal_ngrams(card)
                for card in (
                    *answer_surfaces,
                    *(card for pool in self.answer_pools for card in pool.cards),
                )
            )
        )
        overlap = prompt_grams & answer_grams
        if overlap:
            rendered = " ".join(" ".join(gram) for gram in sorted(overlap)[:5])
            raise ValueError(
                "prompt and answer subcards share literal trigrams: " + rendered
            )
        thinking_grams = set().union(
            *(
                _literal_ngrams(card)
                for card in (
                    *thinking_surfaces,
                    *(card for pool in self.thinking_pools for card in pool.cards),
                )
            )
        )
        role_overlap = (
            (prompt_grams & thinking_grams) | (thinking_grams & answer_grams)
        )
        if role_overlap:
            rendered = " ".join(
                " ".join(gram) for gram in sorted(role_overlap)[:5]
            )
            raise ValueError(
                "thinking subcards share literal trigrams with another role: "
                + rendered
            )

    def deal(self, seed: str) -> V2DealtPair:
        variable_seed = f"{self.name}:{seed}:variables"
        indices = self.variables.matrix.deal_indices(variable_seed)
        dealt = self.variables.matrix.deal(variable_seed)

        def render_pools(
            role: SurfaceRole,
            pools: tuple[V2SubcardPool, ...],
        ) -> tuple[str, ...]:
            pieces = []
            for pool in pools:
                card = pool.cards[_index(f"{self.name}:{seed}:{pool.name}", len(pool.cards))]
                pieces.append(self.variables.matrix.render(card, dealt).strip())
            return tuple(piece for piece in pieces if piece)

        prompt_subcards = render_pools(SurfaceRole.PROMPT, self.prompt_pools)
        thinking_subcards = render_pools(
            SurfaceRole.THINKING, self.thinking_pools
        )
        answer_subcards = render_pools(SurfaceRole.ANSWER, self.answer_pools)
        return V2DealtPair(
            prompt=self.separator.join(prompt_subcards),
            thinking=self.separator.join(thinking_subcards),
            answer=self.separator.join(answer_subcards),
            prompt_subcards=prompt_subcards,
            thinking_subcards=thinking_subcards,
            answer_subcards=answer_subcards,
            variable_indices=indices,
            variable_card_counts={
                axis: {
                    sense: len(cards)
                    for sense, cards in senses.items()
                }
                for axis, senses in self.variables.matrix.table.items()
            },
            dependency_graph=self.variables.matrix.dependency_graph(),
        )


__all__ = ("V2DealtPair", "V2RoleSeparatedDeck", "V2SubcardPool")
