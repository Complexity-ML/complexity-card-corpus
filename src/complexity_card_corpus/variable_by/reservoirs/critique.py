from __future__ import annotations

from collections.abc import Mapping


def critique_reservoir(
    code: str,
    *,
    weakness: str | None = None,
    revision: str | None = None,
    consequences: tuple[str, ...] = (),
) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    """Build nested critique instructions for one candidate."""

    table: dict[str, dict[str, tuple[str, ...]]] = {
        "scenario": {"candidate": (f"candidate {code}",)},
        "critique": {
            "diagnosis": (
                "Identify {scenario[candidate]}'s highest-impact weakness.",
                "Name {scenario[candidate]}'s main evidence or clarity problem.",
                "Diagnose {scenario[candidate]}'s most consequential flaw.",
            ),
            "revision": (
                "Provide a faithful two-sentence revision.",
                "Rewrite the text in exactly two sentences.",
                "Revise the text without inventing any fact.",
            ),
        },
        "constraint": {
            "evidence": (
                "Keep every claim within the supplied evidence.",
                "Preserve the documented meaning and evidentiary limit.",
                "Add no unsupported fact or stronger conclusion.",
            )
        },
    }
    if weakness is not None and revision is not None and consequences:
        table["scenario"].update(
            {
                "weakness": (weakness,),
                "revision": (revision,),
            }
        )
        table["critique"].update(
            {
                "weakness": (
                    "Weakness: {scenario[weakness]}.",
                    "Weakness: {scenario[weakness]}; the wording exceeds the evidence.",
                    "Weakness: {scenario[weakness]}, making verification difficult.",
                ),
                "revised_text": (
                    "Revision: {scenario[revision]}",
                    "Faithful Revision: {scenario[revision]}",
                    "Bounded Revision: {scenario[revision]}",
                ),
            }
        )
        table["consequence"] = {"effect": consequences}
    return table
