from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from ..variable_by import VariableBy2D


class SurfaceRole(StrEnum):
    """Model-facing side allowed to consume one surface reservoir."""

    PROMPT = "prompt"
    THINKING = "thinking"
    ANSWER = "answer"


class ThinkingBudget(StrEnum):
    """Amount and purpose of model-facing reasoning warranted by one task."""

    NONE = "none"
    MINIMAL = "minimal"
    SHORT = "short"
    VERIFICATION = "verification"
    UNCERTAINTY = "uncertainty"


@dataclass(frozen=True)
class ConversationTurn:
    """One authored history turn preceding the current user request."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise ValueError("conversation history roles must be user or assistant")
        if not self.content.strip():
            raise ValueError("conversation history turns require visible content")


@dataclass(frozen=True)
class SemanticFrame:
    """Text-independent truth record consumed by prompt and answer realizers.

    The frame is deliberately model-invisible.  It gives both sides access to
    the same facts without allowing either side to reuse the other's wording.
    """

    intent: str
    facts: Mapping[str, Any]
    constraints: tuple[str, ...] = ()
    expected_outcome: Any = None
    uncertainty: str = "none"
    user_tone: str = "neutral"
    history: tuple[ConversationTurn, ...] = ()
    history_required_facts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.intent.strip():
            raise ValueError("a semantic frame requires an intent")
        if not self.uncertainty.strip() or not self.user_tone.strip():
            raise ValueError("semantic frame uncertainty and tone must be visible")
        if self.history and self.history[-1].role != "assistant":
            raise ValueError(
                "history must end with an assistant turn before the current user turn"
            )
        if self.history and not self.history_required_facts:
            raise ValueError(
                "multi-turn semantic frames must name facts required from history"
            )
        if self.history_required_facts and not self.history:
            raise ValueError("history-required facts need conversation history")
        unknown_history_facts = set(self.history_required_facts) - set(self.facts)
        if unknown_history_facts:
            raise ValueError(
                "history-required facts are absent from the semantic frame: "
                + ", ".join(sorted(unknown_history_facts))
            )
        if len(self.history_required_facts) != len(set(self.history_required_facts)):
            raise ValueError("history-required facts must be unique")
        previous = None
        for turn in self.history:
            if turn.role == previous:
                raise ValueError("conversation history roles must alternate")
            previous = turn.role
        object.__setattr__(self, "facts", MappingProxyType(dict(self.facts)))

    def as_metadata(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "facts": dict(self.facts),
            "constraints": list(self.constraints),
            "expected_outcome": self.expected_outcome,
            "uncertainty": self.uncertainty,
            "user_tone": self.user_tone,
            "history": [
                {"role": turn.role, "content": turn.content}
                for turn in self.history
            ],
            "history_required_facts": list(self.history_required_facts),
        }


def _validate_plan(
    *,
    kind: str,
    name: str,
    pool_names: tuple[str, ...],
    functions: tuple[str, ...],
    cell_choices: Mapping[str, int],
) -> None:
    if not name.strip():
        raise ValueError(f"a {kind} plan requires a visible name")
    if len(pool_names) != len(set(pool_names)):
        raise ValueError(f"{kind} plan {name!r} repeats a subcard pool")
    if not functions or not all(function.strip() for function in functions):
        raise ValueError(f"{kind} plan {name!r} requires visible functions")
    if any(not field.strip() or index < 0 for field, index in cell_choices.items()):
        raise ValueError(f"{kind} plan {name!r} has an invalid cell choice")


@dataclass(frozen=True)
class PromptPlan:
    """A user-side discourse plan, independent from answer wording."""

    name: str
    pool_names: tuple[str, ...]
    functions: tuple[str, ...]
    user_tone: str = "neutral"
    cell_choices: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_plan(
            kind="prompt",
            name=self.name,
            pool_names=self.pool_names,
            functions=self.functions,
            cell_choices=self.cell_choices,
        )
        if not self.pool_names:
            raise ValueError(f"prompt plan {self.name!r} requires a subcard pool")
        if not self.user_tone.strip():
            raise ValueError("prompt plan tone must be visible")
        object.__setattr__(
            self, "cell_choices", MappingProxyType(dict(self.cell_choices))
        )


@dataclass(frozen=True)
class AnswerPlan:
    """An assistant-side discourse plan selected after prompt planning."""

    name: str
    pool_names: tuple[str, ...]
    functions: tuple[str, ...]
    cell_choices: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_plan(
            kind="answer",
            name=self.name,
            pool_names=self.pool_names,
            functions=self.functions,
            cell_choices=self.cell_choices,
        )
        if not self.pool_names:
            raise ValueError(f"answer plan {self.name!r} requires a subcard pool")
        object.__setattr__(
            self, "cell_choices", MappingProxyType(dict(self.cell_choices))
        )


@dataclass(frozen=True)
class ThinkingPlan:
    """Reasoning policy kept distinct from the final answer plan."""

    name: str
    pool_names: tuple[str, ...]
    functions: tuple[str, ...]
    budget: ThinkingBudget
    cell_choices: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_plan(
            kind="thinking",
            name=self.name,
            pool_names=self.pool_names,
            functions=self.functions,
            cell_choices=self.cell_choices,
        )
        if self.budget is ThinkingBudget.NONE and self.pool_names:
            raise ValueError("a no-thinking plan cannot render thinking subcards")
        if self.budget is not ThinkingBudget.NONE and not self.pool_names:
            raise ValueError("a reasoning budget requires thinking subcards")
        object.__setattr__(
            self, "cell_choices", MappingProxyType(dict(self.cell_choices))
        )


@dataclass(frozen=True)
class PlanCompatibility:
    """Allowed plan edges; omitted entries mean every destination is valid."""

    prompt_to_answers: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    answer_to_thinking: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prompt_to_answers",
            MappingProxyType(
                {key: tuple(value) for key, value in self.prompt_to_answers.items()}
            ),
        )
        object.__setattr__(
            self,
            "answer_to_thinking",
            MappingProxyType(
                {key: tuple(value) for key, value in self.answer_to_thinking.items()}
            ),
        )

    @staticmethod
    def _allowed(
        source: str,
        graph: Mapping[str, tuple[str, ...]],
        destinations: tuple[str, ...],
    ) -> tuple[str, ...]:
        requested = graph.get(source, destinations)
        return tuple(destination for destination in destinations if destination in requested)

    def answers_for(
        self,
        prompt_name: str,
        answer_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        return self._allowed(prompt_name, self.prompt_to_answers, answer_names)

    def thinking_for(
        self,
        answer_name: str,
        thinking_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        return self._allowed(answer_name, self.answer_to_thinking, thinking_names)


@dataclass(frozen=True)
class RoleSeparatedVariableBy:
    """VariableBy2D with a hard prompt/answer surface boundary.

    Shared semantic facts live below ``scenario[...]``. Wording that teaches how
    to ask lives below ``prompt[...]`` and wording that teaches how to answer
    lives below ``answer[...]``. A renderer may never cross that boundary.
    """

    matrix: VariableBy2D
    shared_axes: tuple[str, ...] = ("scenario",)

    def __post_init__(self) -> None:
        axes = set(self.matrix.table)
        missing = {SurfaceRole.PROMPT, SurfaceRole.ANSWER} - axes
        if missing:
            raise ValueError(
                "role-separated VariableBy2D requires prompt and answer axes"
            )
        unknown_shared = set(self.shared_axes) - axes
        if unknown_shared:
            raise ValueError(
                "unknown shared VariableBy2D axes: "
                + ", ".join(sorted(unknown_shared))
            )

    def validate(self, role: SurfaceRole, templates: tuple[str, ...]) -> None:
        requested = self.matrix.expand_dependencies(
            self.matrix.validate_templates(templates)
        )
        allowed = {role, *self.shared_axes}
        wrong = tuple(
            field for field in requested if field.partition("[")[0] not in allowed
        )
        if wrong:
            raise ValueError(
                f"{role} templates cross the prompt/answer boundary: "
                + ", ".join(wrong)
            )

    def render(
        self,
        role: SurfaceRole,
        template: str,
        *,
        seed: str,
    ) -> str:
        self.validate(role, (template,))
        return self.matrix.render(template, self.matrix.deal(seed))


__all__ = (
    "AnswerPlan",
    "ConversationTurn",
    "PlanCompatibility",
    "PromptPlan",
    "RoleSeparatedVariableBy",
    "SemanticFrame",
    "SurfaceRole",
    "ThinkingBudget",
    "ThinkingPlan",
)
