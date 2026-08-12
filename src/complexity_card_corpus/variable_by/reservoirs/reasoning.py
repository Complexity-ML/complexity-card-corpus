from __future__ import annotations

from collections.abc import Mapping


def reasoning_reservoir(
    *,
    equation: str,
    total: str | tuple[str, ...],
    check: str | tuple[str, ...],
    quantity_roles: tuple[str, ...],
    domain: str,
    code: str,
    data: str,
) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    """Build nested calculation language from verified scenario values."""

    return {
        "scenario": {
            "equation": (equation,),
            "total": (total,) if isinstance(total, str) else total,
            "check": (check,) if isinstance(check, str) else check,
            "domain": (domain,),
            "code": (code,),
        },
        "label": {
            "problem": ("Problem", "Calculation card", "Supplied values"),
            "goal": ("Calculation goal", "Verification task", "Reasoning objective"),
            "situation": ("Calculation context", "Reasoning case", "Verified problem"),
        },
        "problem": {"statement": (data,)},
        "goal": {
            "instruction": (
                "Calculate {scenario[domain]} problem {scenario[code]} and show the equation.",
                "Give the equation and total for {scenario[code]}.",
                "Solve supplied {scenario[domain]} problem {scenario[code]}.",
            )
        },
        "constraint": {
            "verification": (
                "Verify the result with an independent check.",
                "Confirm the result through a second calculation.",
                "Include one independent numerical check.",
            )
        },
        "situation": {
            "calculation": (
                "The supplied values define a complete, independently checkable calculation.",
                "The quantities provide everything needed for a checkable result.",
                "The problem contains a complete calculation and a separate verification route.",
            )
        },
        "calculation": {
            "equation": (
                "Equation: {scenario[equation]}.",
                "Equation: using the supplied values, {scenario[equation]}.",
                "Equation: the direct calculation is {scenario[equation]}.",
                "Equation: represent the required operation as {scenario[equation]}.",
                "Equation: evaluating the quantities gives {scenario[equation]}.",
                "Equation: the numerical relation is {scenario[equation]}.",
                "Equation: mapping each supplied quantity to its role gives {scenario[equation]}.",
                "Equation: following the stated order of operations yields {scenario[equation]}.",
                "Equation: the quantities combine in this form: {scenario[equation]}.",
                "Equation: a direct numerical model of the prompt is {scenario[equation]}.",
                "Equation: the requested quantities evaluate as {scenario[equation]}.",
                "Equation: writing the supported arithmetic gives {scenario[equation]}.",
                "Equation: the values enter the calculation as {scenario[equation]}.",
                "Equation: the operation grounded in the prompt is {scenario[equation]}.",
                "Equation: a concise representation is {scenario[equation]}.",
                "Equation: applying the stated relationship yields {scenario[equation]}.",
                "Equation: the inputs form the expression {scenario[equation]}.",
                "Equation: the numerical steps can be written {scenario[equation]}.",
                "Equation: the checkable arithmetic is {scenario[equation]}.",
                "Equation: translating the problem into numbers gives {scenario[equation]}.",
                "Equation: the required computation takes the form {scenario[equation]}.",
                "Equation: the complete expression evaluates to {scenario[equation]}.",
                "Equation: combining the given values produces {scenario[equation]}.",
                "Equation: the direct supported expression reads {scenario[equation]}.",
            ),
            "total": (
                "Total: {scenario[total]}.",
                "Total: this gives {scenario[total]}.",
                "Total: the result is {scenario[total]}.",
                "Total: the computed value is {scenario[total]}.",
                "Total: therefore, {scenario[total]}.",
                "Total: the supplied values produce {scenario[total]}.",
                "Total: completing those operations produces {scenario[total]}.",
                "Total: after applying every stated quantity, the answer is {scenario[total]}.",
                "Total: the resulting quantity is {scenario[total]}.",
                "Total: the final evaluated amount is {scenario[total]}.",
            ),
        },
        "verification": {
            "check": (
                "Check: {scenario[check]}.",
                "Check: independently, {scenario[check]}.",
                "Check: a second view confirms that {scenario[check]}.",
                "Check: verify the result by noting that {scenario[check]}.",
                "Check: reversing or decomposing the operation shows that {scenario[check]}.",
                "Check: an independent reconstruction confirms that {scenario[check]}.",
                "Check: the quantities remain consistent because {scenario[check]}.",
                "Check: a separate numerical route establishes that {scenario[check]}.",
            )
        },
        "explanation": {"quantity_role": quantity_roles},
    }
