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
    "Find the decisive implication of {focus}.",
    "Use only {focus} to establish the next inference.",
    "Make {focus} the basis of the first check.",
    "Determine the result controlled by {focus}.",
    "Before choosing, apply {focus} to the candidates.",
    "Follow the evidence carried by {focus}.",
    "Treat {focus} as the constraint on every inference.",
    "Derive the supported outcome from {focus}.",
)

_OPENING_FOCUS = {
    "reasoning_verification": (
        "the supplied quantities and their required relation",
        "the numerical relationship stated in the problem",
        "the operation connecting the given values",
        "the quantities that determine the result",
        "the calculation implied by the supplied facts",
        "the stated values and the operation between them",
        "the numerical model supported by the prompt",
        "the arithmetic relation that must be evaluated",
        "the given amounts and their mathematical connection",
        "the exact operation required by the question",
        "the value relationship available in the evidence",
        "the numerical path from the inputs to the result",
        "the quantities that belong in the calculation",
        "the equation grounded in the stated information",
        "the relevant inputs and how they combine",
        "the mathematical structure of the supplied case",
        "the operation that maps the inputs to the answer",
        "the stated numerical facts and their consequence",
        "the checkable relation among the given quantities",
        "the calculation supported by the available numbers",
        "the input values and the result they entail",
        "the arithmetic dependency inside the request",
        "the numerical evidence needed for the conclusion",
        "the shortest valid calculation from the given facts",
        "the supplied amounts and the operation joining them",
        "the numerical relationship fixed by the question",
        "the given quantities and their exact consequence",
        "the arithmetic model supported by the stated facts",
        "the values that control the requested computation",
        "the equation and the result it determines",
        "the quantities required for the direct calculation",
        "the numerical dependency established in the prompt",
    ),
    "planning_comparison": (
        "the hard constraints that determine viability",
        "the non-negotiable conditions governing the choice",
        "the criteria that eliminate infeasible options",
        "the fixed limits that define an acceptable plan",
        "the binding requirements behind the comparison",
        "the deadline and resource gates on the decision",
        "the conditions every viable option must satisfy",
        "the constraints that separate eligible choices",
        "the mandatory tests applied to each alternative",
        "the decision boundaries stated in the request",
        "the feasibility rules that control the shortlist",
        "the required conditions and their practical effect",
        "the criteria that narrow the available alternatives",
        "the hard limits before any preference is considered",
        "the eligibility checks attached to the decision",
        "the fixed requirements that shape the sequence",
        "the constraints that rule options in or out",
        "the mandatory boundaries on cost timing and scope",
        "the supplied gates for selecting a workable option",
        "the requirements that the final plan cannot violate",
        "the comparison criteria with disqualifying force",
        "the limits that preserve a feasible course of action",
        "the decision rules supported by the supplied facts",
        "the binding conditions for a reversible choice",
    ),
    "explanation_learning": "the mechanism the learner must understand",
    "critique_revision": "the defect that most affects correctness",
    "troubleshooting": (
        "the safest test that can separate possible causes",
        "the reversible check that best isolates the fault",
        "the smallest experiment that distinguishes likely causes",
        "the controlled test with the clearest diagnostic value",
        "the next safe observation that can narrow the cause",
        "the least disruptive way to isolate the failure",
        "the reversible change that separates competing explanations",
        "the diagnostic step that preserves the current state",
        "the safest comparison between baseline and failure",
        "the controlled check most likely to localize the issue",
        "the minimum-risk test of the leading hypothesis",
        "the observation that can eliminate the most causes",
        "the next check that keeps rollback available",
        "the isolated test needed to identify the failing layer",
        "the lowest-risk experiment with a decisive outcome",
        "the evidence-producing step that avoids broad changes",
        "the diagnostic boundary between the plausible causes",
        "the reversible probe that can locate the fault",
        "the controlled reproduction needed before changing more",
        "the preserved-state check that can distinguish the competing causes",
        "the single-variable probe with a reversible outcome",
        "the next observation that compares the fault with a known baseline",
        "the bounded experiment that reveals which layer diverges first",
        "the least invasive comparison capable of rejecting a hypothesis",
        "the isolated replay that produces evidence without risking source data",
        "the safe diagnostic move that keeps every unrelated condition fixed",
        "the recoverable test whose result meaningfully narrows the fault",
        "the smallest safe test of the current explanation",
        "the baseline comparison that can expose the difference",
        "the next observation that would reduce uncertainty most",
        "the focused diagnostic action that protects existing data",
        "the check that distinguishes cause from coincidence",
    ),
}

_VERIFICATION_PATTERNS = (
    "A separate boundary is that {verification}.",
    "Independent verification shows that {verification}.",
    "The distinct check is this: {verification}.",
    "One remaining constraint is that {verification}.",
    "A second route confirms that {verification}.",
    "The result must also respect that {verification}.",
    "Another supported observation is that {verification}.",
    "The cross-check establishes that {verification}.",
    "Separately, the evidence indicates that {verification}.",
    "The independent condition remains: {verification}.",
    "Verification depends on the fact that {verification}.",
    "A further evidence boundary is that {verification}.",
    "The conclusion is checked by noting that {verification}.",
    "One additional test confirms that {verification}.",
    "The separate limiting fact is that {verification}.",
    "As an independent check, {verification}.",
    "The available evidence also requires that {verification}.",
    "A second supported path finds that {verification}.",
    "The verification step preserves this condition: {verification}.",
    "The outcome remains consistent because {verification}.",
    "Another checkable fact is that {verification}.",
    "A separate check confirms the result because {verification}.",
    "The independent evidence leaves one limiting condition: {verification}.",
    "A final cross-check supports that {verification}.",
)


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
        f"{{opening[{task}]}} {{scenario[analysis]}} As a cross-check, {{scenario[verification_inline]}}.",
        f"{{opening[{task}]}} {{scenario[verification]}} Separately, {{scenario[analysis_inline]}}.",
        f"{{opening[{task}]}} {{scenario[analysis]}} Confirmation comes from this: {{scenario[verification_inline]}}.",
        f"{{opening[{task}]}} {{scenario[verification]}} The supported route is that {{scenario[analysis_inline]}}.",
        f"{{opening[{task}]}} {{scenario[analysis]}} A distinct verification shows {{scenario[verification_inline]}}.",
        f"{{opening[{task}]}} {{scenario[verification]}} This is consistent with the fact that {{scenario[analysis_inline]}}.",
        f"{{opening[{task}]}} {{scenario[analysis]}} To verify it independently, note that {{scenario[verification_inline]}}.",
        f"{{opening[{task}]}} {{scenario[verification]}} The main inference remains {{scenario[analysis_inline]}}.",
        f"{{opening[{task}]}} {{scenario[analysis]}} The confirming observation is: {{scenario[verification_inline]}}.",
        f"{{opening[{task}]}} {{scenario[verification]}} From the same evidence, {{scenario[analysis_inline]}}.",
        f"{{opening[{task}]}} {{scenario[analysis]}} One separate test establishes that {{scenario[verification_inline]}}.",
        f"{{opening[{task}]}} {{scenario[verification]}} The corresponding analysis is that {{scenario[analysis_inline]}}.",
        f"{{opening[{task}]}} {{scenario[analysis]}} Validation follows because {{scenario[verification_inline]}}.",
        f"{{opening[{task}]}} {{scenario[verification]}} In turn, the evidence supports that {{scenario[analysis_inline]}}.",
        f"{{opening[{task}]}} {{scenario[analysis]}} The independent result agrees: {{scenario[verification_inline]}}.",
        f"{{opening[{task}]}} {{scenario[verification]}} Accordingly, {{scenario[analysis_inline]}}.",
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
        think_cards = tuple(
            card.replace(
                "{scenario[verification]}",
                f"{{verification[{task}]}}",
            )
            for card in _THINK_STRUCTURES[task]
        )
    except KeyError as error:
        raise ValueError(f"unsupported reasoning-envelope task: {task}") from error
    if len(final_variants) < 3:
        raise ValueError("reasoning-envelope final deck requires at least three cards")
    return {
        "focus": {
            task: (
                _OPENING_FOCUS[task]
                if isinstance(_OPENING_FOCUS[task], tuple)
                else (_OPENING_FOCUS[task],)
            )
        },
        "opening": {
            task: tuple(
                pattern.replace("{focus}", f"{{focus[{task}]}}")
                for pattern in _OPENING_PATTERNS
            )
        },
        "verification": {
            task: tuple(
                pattern.replace(
                    "{verification}", "{scenario[verification_inline]}"
                )
                for pattern in _VERIFICATION_PATTERNS
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
