from __future__ import annotations

from collections.abc import Mapping


_OPENING_PATTERNS = (
    "Begin by isolating {focus}.",
    "Start directly from {focus}.",
    "First identify {focus}.",
    "Use {focus} as the initial boundary.",
    "Anchor the analysis in {focus}.",
    "Separate {focus} from secondary details.",
    "Check {focus} before selecting a conclusion.",
    "Map the supplied evidence onto {focus}.",
    "Work outward from {focus}.",
    "Establish {focus} before evaluating the result.",
    "Test the candidate answer against {focus}.",
    "Resolve {focus} before considering optional details.",
    "Identify what follows directly from {focus}.",
    "Take the shortest route through {focus}.",
    "Trace a supported path through {focus}.",
    "Keep every step tied to {focus}.",
    "Apply {focus} before weighing alternatives.",
    "Locate {focus} before drafting the response.",
    "Preserve the given facts while checking {focus}.",
    "Distinguish {focus} from illustrative detail.",
    "Define {focus} before comparing outcomes.",
    "Reconstruct the result using only {focus}.",
    "Reduce the request to {focus}.",
    "Verify the conclusion independently from {focus}.",
)

_OPENING_FOCUS = {
    "reasoning_verification": "the supplied quantities and their required relation",
    "planning_comparison": "the hard constraints that determine viability",
    "explanation_learning": "the mechanism the learner must understand",
    "critique_revision": "the defect that most affects correctness",
    "troubleshooting": "the safest test that can separate possible causes",
}


_THINK_STRUCTURES = {
    task: (
        f"{{opening[{task}]}} {{scenario[analysis]}} {{scenario[verification]}}",
        f"{{opening[{task}]}} {{scenario[verification]}} {{scenario[analysis]}}",
        f"{{opening[{task}]}} First, {{scenario[analysis_inline]}}. Then, {{scenario[verification_inline]}}.",
        f"{{opening[{task}]}} {{scenario[analysis]}} Independently, {{scenario[verification_inline]}}.",
        f"{{opening[{task}]}} {{scenario[analysis]}} The separate check is that {{scenario[verification_inline]}}.",
        f"{{opening[{task}]}} {{scenario[verification]}} On that basis, {{scenario[analysis_inline]}}.",
        f"{{opening[{task}]}} {{scenario[analysis]}}\n{{scenario[verification]}}",
        f"{{opening[{task}]}} {{scenario[analysis]}}\n\n{{scenario[verification]}}",
    )
    for task in (
        "reasoning_verification",
        "planning_comparison",
        "explanation_learning",
        "critique_revision",
        "troubleshooting",
    )
}


def reasoning_envelope_reservoir(
    task: str,
    *,
    analysis: str,
    analysis_inline: str,
    verification: str,
    verification_inline: str,
    final_variants: tuple[str, ...],
) -> dict[str, Mapping[str, tuple[str, ...]]]:
    """Return a nested think/final matrix grounded in one scenario answer."""

    try:
        think_cards = _THINK_STRUCTURES[task]
    except KeyError as error:
        raise ValueError(f"unsupported reasoning-envelope task: {task}") from error
    if len(final_variants) < 3:
        raise ValueError("reasoning-envelope final deck requires at least three cards")
    return {
        "opening": {
            task: tuple(
                pattern.format(focus=_OPENING_FOCUS[task])
                for pattern in _OPENING_PATTERNS
            )
        },
        "scenario": {
            "analysis": (analysis,),
            "analysis_inline": (analysis_inline,),
            "verification": (verification,),
            "verification_inline": (verification_inline,),
            **{
                f"final_{index}": (variant,)
                for index, variant in enumerate(final_variants)
            },
        },
        "think": {task: think_cards},
        "final": {
            task: tuple(
                f"{{scenario[final_{index}]}}"
                for index in range(len(final_variants))
            )
        },
    }
