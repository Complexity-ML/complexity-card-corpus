from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .contracts import (
    AnswerPlan,
    PlanCompatibility,
    PromptPlan,
    RoleSeparatedVariableBy,
    SurfaceRole,
    ThinkingBudget,
    ThinkingPlan,
)


_WORD = re.compile(r"[a-z0-9']+", re.I)


def _index(seed: str, size: int) -> int:
    value = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    return value % size


def _literal_ngrams(template: str, size: int = 3) -> set[tuple[str, ...]]:
    literal = re.sub(r"\{[^{}]+\}", " ", template)
    words = _WORD.findall(literal.casefold())
    return set(zip(*(words[index:] for index in range(size))))


def prompt_variant_plans(
    *,
    sense: str,
    pool_name: str,
    functions: tuple[tuple[str, ...], ...],
    tones: tuple[str, ...] = (),
) -> tuple[PromptPlan, ...]:
    """Author one explicit prompt plan per surface realization."""

    if tones and len(tones) != len(functions):
        raise ValueError("prompt variant tones must match function sequences")
    return tuple(
        PromptPlan(
            name=f"prompt-{sense}-{index}",
            pool_names=(pool_name,),
            functions=sequence,
            user_tone=tones[index] if tones else "neutral",
            cell_choices={f"prompt[{sense}]": index},
        )
        for index, sequence in enumerate(functions)
    )


def answer_variant_plans(
    *,
    sense: str,
    pool_name: str,
    functions: tuple[tuple[str, ...], ...],
) -> tuple[AnswerPlan, ...]:
    """Author one explicit answer plan per behavioral realization."""

    return tuple(
        AnswerPlan(
            name=f"answer-{sense}-{index}",
            pool_names=(pool_name,),
            functions=sequence,
            cell_choices={f"answer[{sense}]": index},
        )
        for index, sequence in enumerate(functions)
    )


def thinking_variant_plans(
    *,
    sense: str,
    pool_name: str,
    functions: tuple[tuple[str, ...], ...],
    budget: ThinkingBudget,
) -> tuple[ThinkingPlan, ...]:
    """Author one explicit reasoning plan per behavioral realization."""

    return tuple(
        ThinkingPlan(
            name=f"thinking-{sense}-{index}",
            pool_names=(pool_name,),
            functions=sequence,
            budget=budget,
            cell_choices={f"thinking[{sense}]": index},
        )
        for index, sequence in enumerate(functions)
    )


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
    prompt_plan: str
    answer_plan: str
    thinking_plan: str
    prompt_functions: tuple[str, ...]
    answer_functions: tuple[str, ...]
    thinking_functions: tuple[str, ...]
    user_tone: str
    thinking_budget: ThinkingBudget
    allowed_prompt_answer_edges: tuple[tuple[str, str], ...]
    allowed_answer_thinking_edges: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class V2RoleSeparatedDeck:
    """Compose prompts and answers from disjoint, role-checked subcard pools."""

    name: str
    variables: RoleSeparatedVariableBy
    prompt_pools: tuple[V2SubcardPool, ...]
    answer_pools: tuple[V2SubcardPool, ...]
    thinking_pools: tuple[V2SubcardPool, ...] = ()
    separator: str = " "
    prompt_plans: tuple[PromptPlan, ...] = ()
    answer_plans: tuple[AnswerPlan, ...] = ()
    thinking_plans: tuple[ThinkingPlan, ...] = ()
    compatibility: PlanCompatibility = PlanCompatibility()
    default_thinking_budget: ThinkingBudget = ThinkingBudget.SHORT

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
        prompt_plans = self._prompt_plans()
        answer_plans = self._answer_plans()
        thinking_plans = self._thinking_plans()
        self._validate_plans(
            prompt_plans,
            answer_plans,
            thinking_plans,
        )
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

    def _variant_choices(
        self,
        role: SurfaceRole,
        pools: tuple[V2SubcardPool, ...],
    ) -> tuple[dict[str, int], ...]:
        templates = tuple(card for pool in pools for card in pool.cards)
        fields = self.variables.matrix.expand_dependencies(
            self.variables.matrix.validate_templates(templates)
        )
        role_fields = tuple(
            field for field in fields if field.partition("[")[0] == role
        )
        sizes = {
            field: len(
                self.variables.matrix.cards(
                    field.partition("[")[0], field.partition("[")[2][:-1]
                )
            )
            for field in role_fields
        }
        variants = max(sizes.values(), default=1)
        return tuple(
            {field: variant % size for field, size in sizes.items()}
            for variant in range(variants)
        )

    @staticmethod
    def _variant_name(prefix: str, index: int, count: int) -> str:
        return f"{prefix}-default" if count == 1 else f"{prefix}-variant-{index}"

    def _prompt_plans(self) -> tuple[PromptPlan, ...]:
        if self.prompt_plans:
            return self.prompt_plans
        choices = self._variant_choices(SurfaceRole.PROMPT, self.prompt_pools)
        return tuple(
            PromptPlan(
                name=self._variant_name("prompt", index, len(choices)),
                pool_names=tuple(pool.name for pool in self.prompt_pools),
                functions=tuple(pool.name for pool in self.prompt_pools),
                cell_choices=choice,
            )
            for index, choice in enumerate(choices)
        )

    def _answer_plans(self) -> tuple[AnswerPlan, ...]:
        if self.answer_plans:
            return self.answer_plans
        choices = self._variant_choices(SurfaceRole.ANSWER, self.answer_pools)
        return tuple(
            AnswerPlan(
                name=self._variant_name("answer", index, len(choices)),
                pool_names=tuple(pool.name for pool in self.answer_pools),
                functions=tuple(pool.name for pool in self.answer_pools),
                cell_choices=choice,
            )
            for index, choice in enumerate(choices)
        )

    def _thinking_plans(self) -> tuple[ThinkingPlan, ...]:
        if self.thinking_plans:
            return self.thinking_plans
        if self.thinking_pools:
            choices = self._variant_choices(
                SurfaceRole.THINKING, self.thinking_pools
            )
            return tuple(
                ThinkingPlan(
                    name=self._variant_name("thinking", index, len(choices)),
                    pool_names=tuple(pool.name for pool in self.thinking_pools),
                    functions=tuple(pool.name for pool in self.thinking_pools),
                    budget=self.default_thinking_budget,
                    cell_choices=choice,
                )
                for index, choice in enumerate(choices)
            )
        return (
            ThinkingPlan(
                name="thinking-none",
                pool_names=(),
                functions=("direct",),
                budget=ThinkingBudget.NONE,
            ),
        )

    def _validate_plans(
        self,
        prompt_plans: tuple[PromptPlan, ...],
        answer_plans: tuple[AnswerPlan, ...],
        thinking_plans: tuple[ThinkingPlan, ...],
    ) -> None:
        groups = {
            "prompt": (
                SurfaceRole.PROMPT,
                prompt_plans,
                {pool.name for pool in self.prompt_pools},
            ),
            "answer": (
                SurfaceRole.ANSWER,
                answer_plans,
                {pool.name for pool in self.answer_pools},
            ),
            "thinking": (
                SurfaceRole.THINKING,
                thinking_plans,
                {pool.name for pool in self.thinking_pools},
            ),
        }
        available_fields = set(self.variables.matrix.field_names())
        for kind, (role, plans, available) in groups.items():
            names = tuple(plan.name for plan in plans)
            if len(names) != len(set(names)):
                raise ValueError(f"{kind} plan names must be unique inside a deck")
            for plan in plans:
                unknown = set(plan.pool_names) - available
                if unknown:
                    raise ValueError(
                        f"{kind} plan {plan.name!r} references unknown pools: "
                        + ", ".join(sorted(unknown))
                    )
                for field, index in plan.cell_choices.items():
                    if field not in available_fields:
                        raise ValueError(
                            f"{kind} plan {plan.name!r} chooses unknown cell {field}"
                        )
                    if field.partition("[")[0] != role:
                        raise ValueError(
                            f"{kind} plan {plan.name!r} crosses the surface boundary"
                        )
                    axis, _, sense = field.partition("[")
                    size = len(self.variables.matrix.cards(axis, sense[:-1]))
                    if index >= size:
                        raise ValueError(
                            f"{kind} plan {plan.name!r} chooses card {index} "
                            f"from a {size}-card cell"
                        )

        prompt_names = {plan.name for plan in prompt_plans}
        answer_names = {plan.name for plan in answer_plans}
        thinking_names = {plan.name for plan in thinking_plans}
        unknown_prompt = set(self.compatibility.prompt_to_answers) - prompt_names
        unknown_answer_sources = (
            set(self.compatibility.answer_to_thinking) - answer_names
        )
        unknown_answer_targets = {
            target
            for targets in self.compatibility.prompt_to_answers.values()
            for target in targets
            if target not in answer_names
        }
        unknown_thinking_targets = {
            target
            for targets in self.compatibility.answer_to_thinking.values()
            for target in targets
            if target not in thinking_names
        }
        if unknown_prompt or unknown_answer_sources:
            raise ValueError("plan compatibility references an unknown source plan")
        if unknown_answer_targets or unknown_thinking_targets:
            raise ValueError("plan compatibility references an unknown target plan")
        for prompt in prompt_plans:
            if not self.compatibility.answers_for(
                prompt.name, tuple(plan.name for plan in answer_plans)
            ):
                raise ValueError(f"prompt plan {prompt.name!r} has no compatible answer")
        for answer in answer_plans:
            if not self.compatibility.thinking_for(
                answer.name, tuple(plan.name for plan in thinking_plans)
            ):
                raise ValueError(f"answer plan {answer.name!r} has no thinking policy")

    def deal(self, seed: str) -> V2DealtPair:
        prompt_plans = self._prompt_plans()
        answer_plans = self._answer_plans()
        thinking_plans = self._thinking_plans()
        prompt_plan = prompt_plans[
            _index(f"{self.name}:{seed}:prompt-plan", len(prompt_plans))
        ]
        compatible_answers = self.compatibility.answers_for(
            prompt_plan.name,
            tuple(plan.name for plan in answer_plans),
        )
        answer_options = tuple(
            plan for plan in answer_plans if plan.name in compatible_answers
        )
        answer_plan = answer_options[
            _index(f"{self.name}:{seed}:answer-plan", len(answer_options))
        ]
        compatible_thinking = self.compatibility.thinking_for(
            answer_plan.name,
            tuple(plan.name for plan in thinking_plans),
        )
        thinking_options = tuple(
            plan for plan in thinking_plans if plan.name in compatible_thinking
        )
        thinking_plan = thinking_options[
            _index(f"{self.name}:{seed}:thinking-plan", len(thinking_options))
        ]
        allowed_prompt_answer_edges = tuple(
            (prompt.name, answer_name)
            for prompt in prompt_plans
            for answer_name in self.compatibility.answers_for(
                prompt.name,
                tuple(plan.name for plan in answer_plans),
            )
        )
        allowed_answer_thinking_edges = tuple(
            (answer.name, thinking_name)
            for answer in answer_plans
            for thinking_name in self.compatibility.thinking_for(
                answer.name,
                tuple(plan.name for plan in thinking_plans),
            )
        )
        cell_choices = {
            **prompt_plan.cell_choices,
            **answer_plan.cell_choices,
            **thinking_plan.cell_choices,
        }
        variable_seed = f"{self.name}:{seed}:variables"
        indices = self.variables.matrix.deal_indices(variable_seed, cell_choices)
        dealt = self.variables.matrix.deal(variable_seed, cell_choices)

        def render_pools(
            pools: tuple[V2SubcardPool, ...],
            pool_names: tuple[str, ...],
        ) -> tuple[str, ...]:
            by_name = {pool.name: pool for pool in pools}
            pieces = []
            for name in pool_names:
                pool = by_name[name]
                card = pool.cards[_index(f"{self.name}:{seed}:{pool.name}", len(pool.cards))]
                pieces.append(self.variables.matrix.render(card, dealt).strip())
            return tuple(piece for piece in pieces if piece)

        prompt_subcards = render_pools(
            self.prompt_pools,
            prompt_plan.pool_names,
        )
        thinking_subcards = render_pools(
            self.thinking_pools,
            thinking_plan.pool_names,
        )
        answer_subcards = render_pools(
            self.answer_pools,
            answer_plan.pool_names,
        )
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
            prompt_plan=prompt_plan.name,
            answer_plan=answer_plan.name,
            thinking_plan=thinking_plan.name,
            prompt_functions=prompt_plan.functions,
            answer_functions=answer_plan.functions,
            thinking_functions=thinking_plan.functions,
            user_tone=prompt_plan.user_tone,
            thinking_budget=thinking_plan.budget,
            allowed_prompt_answer_edges=allowed_prompt_answer_edges,
            allowed_answer_thinking_edges=allowed_answer_thinking_edges,
        )


__all__ = (
    "V2DealtPair",
    "V2RoleSeparatedDeck",
    "V2SubcardPool",
    "answer_variant_plans",
    "prompt_variant_plans",
    "thinking_variant_plans",
)
