from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..variable_by import VariableBy2D


class SurfaceRole(StrEnum):
    """Model-facing side allowed to consume one surface reservoir."""

    PROMPT = "prompt"
    THINKING = "thinking"
    ANSWER = "answer"


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


__all__ = ("RoleSeparatedVariableBy", "SurfaceRole")
