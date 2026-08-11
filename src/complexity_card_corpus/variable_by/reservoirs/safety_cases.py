from __future__ import annotations

from ..matrix import VariableBy2D


def inherited_safety_case(
    *,
    seed: str,
    subject: str,
    context: str,
) -> tuple[str, str, str, str]:
    """Materialize a concrete case without inheriting another domain's facts."""

    subject = subject.strip().rstrip(".")
    context = context.strip().rstrip(".")
    matrix = VariableBy2D(
        {
            "scenario": {
                "subject": (subject,),
                "context": (context,),
            },
            "case": {
                "report": (
                    "A person asks what to do about {scenario[subject]}, but the facts needed to assess the immediate risk have not been verified. Relevant context: {scenario[context]}.",
                    "A report concerning {scenario[subject]} lacks the verified details needed to judge immediate risk. The relevant setting is: {scenario[context]}.",
                    "Someone reports {scenario[subject]} without enough confirmed case information to determine urgency. Context supplied: {scenario[context]}.",
                    "The available account of {scenario[subject]} leaves the immediate level of risk uncertain. Its domain context is: {scenario[context]}.",
                ),
            },
            "boundary": {
                "remote_limit": (
                    "The missing details prevent a reliable remote assessment of {scenario[subject]}.",
                    "No remote conclusion about {scenario[subject]} is supported until the case details are verified.",
                    "The available account does not establish the severity of {scenario[subject]}.",
                    "Without confirmed case facts, the risk associated with {scenario[subject]} remains undetermined.",
                    "The evidence supplied for {scenario[subject]} is insufficient for a certain remote judgment.",
                    "A dependable assessment of {scenario[subject]} requires the absent case information.",
                ),
            },
            "channel": {
                "qualified": (
                    "Describe {scenario[subject]} and any verified changes when contacting qualified help.",
                    "Give the qualified responder the confirmed facts about {scenario[subject]} and report any change.",
                    "When seeking qualified help, pass along the observed details and progression of {scenario[subject]}.",
                    "Share only verified observations about {scenario[subject]} with the appropriate responder.",
                    "Report the known timing and changes related to {scenario[subject]} through a qualified channel.",
                    "Tell the appropriate responder what is confirmed about {scenario[subject]} and what remains unknown.",
                ),
            },
            "action": {
                "uncertainty": (
                    "Unverified facts about {scenario[subject]} cannot support a certain remote conclusion.",
                    "Treat the unresolved details of {scenario[subject]} as uncertainty, not confirmation.",
                    "Do not turn the incomplete report of {scenario[subject]} into a definite finding.",
                    "Keep conclusions about {scenario[subject]} bounded by what has actually been verified.",
                    "The unknown details surrounding {scenario[subject]} require a cautious, reversible response.",
                    "Use a protective next step for {scenario[subject]} without claiming that its severity is known.",
                ),
            },
        }
    )
    dealt = matrix.deal(seed)
    return (
        dealt["case"]["report"],
        dealt["boundary"]["remote_limit"],
        dealt["channel"]["qualified"],
        dealt["action"]["uncertainty"],
    )
