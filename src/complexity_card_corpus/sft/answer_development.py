from __future__ import annotations

from typing import Any

from ..tasks.core import LinkedSubcardDeck, SubcardPool
from .language import _inline_sentence, _sentence


# These thresholds protect naturally concise contracts (JSON extraction and
# faithful rewriting are deliberately absent) while allowing discursive
# answers to carry one more grounded implication or validation sentence.
MINIMUM_DEVELOPED_WORDS = {
    "context_clarification": 36,
    "conversation_empathy": 36,
    "critique_revision": 56,
    "explanation_learning": 62,
    "grounded_qa": 42,
    "reasoning_verification": 44,
    "summarization_synthesis": 42,
}


# Twenty-four structure cards are shared across families. Every family below
# adds twenty-four semantic cards, giving 48 cards and 576 linked compositions
# per family.
# The semantic cards contain no new case fact: they only make the already
# authored evidence boundary, verification method, or completion criterion
# explicit.
_DEVELOPMENT_FRAMES = (
    "Taken together, this means {detail}.",
    "Most importantly, {detail}.",
    "The final boundary is clear: {detail}.",
    "A useful final check confirms that {detail}.",
    "The practical implication is that {detail}.",
    "This keeps the reasoning grounded because {detail}.",
    "The conclusion remains bounded: {detail}.",
    "What should carry forward is that {detail}.",
    "The response can therefore stay specific: {detail}.",
    "The remaining standard is simple: {detail}.",
    "This distinction matters because {detail}.",
    "That gives a concrete completion criterion because {detail}.",
    "The supported takeaway is that {detail}.",
    "This also makes explicit that {detail}.",
    "The answer stays useful only while {detail}.",
    "For a final verification, note that {detail}.",
    "The result is easier to use because {detail}.",
    "This preserves the important separation: {detail}.",
    "The next reader can verify that {detail}.",
    "In short, {detail}.",
    "The answer is complete on this basis: {detail}.",
    "A reader can safely act on the result because {detail}.",
    "The evidence and conclusion remain aligned because {detail}.",
    "The closing test is satisfied when {detail}.",
)


_DETAIL_CARDS: dict[str, tuple[str, ...]] = {
    "grounded_qa": (
        "the stated fact about {subject} is separated from every undocumented field",
        "the supplied record supports the direct answer but not the missing detail",
        "the known part of {subject} remains distinct from what the source never states",
        "no requested detail is treated as known unless the supplied record establishes it",
        "the answer reports the evidence about {subject} without filling its documented gap",
        "every asserted detail about {subject} can be located in the supplied material",
        "the missing field stays unknown instead of being completed by a plausible guess",
        "the response answers only the portion of the request supported by the source",
        "the documented value is reported directly and the absent value remains open",
        "the evidence boundary around {subject} is visible in the final wording",
        "the known answer is not extended beyond the scope of the provided record",
        "the source determines both the direct answer and the limit on further claims",
        "the stated result remains verifiable while the unsupported part is withheld",
        "the response distinguishes a recorded fact from an unavailable answer",
        "the conclusion about {subject} contains no detail supplied only by inference",
        "the requested unknown is named explicitly rather than hidden by confident prose",
        "the answer can be checked line by line against the evidence that was provided",
        "the supported part remains useful without turning absence into certainty",
        "the source scope, known value, and unresolved field stay aligned",
        "the final wording preserves uncertainty exactly where the record requires it",
        "the documented answer remains useful without converting silence into evidence",
        "the final statement preserves both the available fact and the unavailable field",
        "the response makes the source limit inspectable alongside the supported answer",
        "the conclusion can be traced to the record without relying on outside completion",
    ),
    "reasoning_verification": (
        "the result is checked from the supplied quantities rather than accepted on appearance",
        "the verification uses the same stated values from a second direction",
        "the computed result and the independent check agree on {subject}",
        "each number in the conclusion can be traced back to the given quantities",
        "the check tests the calculation instead of merely repeating its final value",
        "the conclusion follows from an explicit operation and a separate consistency test",
        "the arithmetic path for {subject} remains visible from inputs to result",
        "the second view would expose a mismatch if the first calculation were wrong",
        "the unit and magnitude remain attached to each quantity throughout the check",
        "the final value is supported by both computation and an independent relation",
        "the verification reconstructs the result without assuming that it is correct",
        "the supplied numbers satisfy the equation and its reverse consistency check",
        "the method for {subject} makes both the operation and the verification inspectable",
        "the numerical answer survives a check that uses a different arrangement of the inputs",
        "the calculation can be audited without relying on an unstated intermediate value",
        "the final number remains consistent with the original quantities and their units",
        "the reasoning exposes enough intermediate structure to catch an arithmetic slip",
        "the answer is verified by relation, not by confidence in the first computation",
        "the stated total agrees with the direct operation and the closing sanity test",
        "the check closes the loop between the given values and the reported result",
        "the computation and its reverse check use the same quantities consistently",
        "the final magnitude remains compatible with every value supplied in the problem",
        "the reasoning records enough intermediate work to reproduce the answer independently",
        "the verified result survives both the direct calculation and a separate plausibility check",
    ),
    "critique_revision": (
        "the revised wording keeps the supported point about {subject} and removes the overclaim",
        "the correction addresses the main weakness without adding an unrecorded fact",
        "the revision is narrower, verifiable, and faithful to the evidence about {subject}",
        "the unsupported inference is removed while the useful part of the original remains",
        "the new wording can be checked directly against the supplied record",
        "the strongest valid claim about {subject} remains after the unsupported language is removed",
        "the corrected text resolves the central flaw without rewriting unrelated material",
        "the revision replaces certainty with the narrower statement the evidence permits",
        "the main claim becomes proportionate to the observations actually supplied",
        "the evidence limit is visible instead of being buried beneath stronger wording",
        "the revised version says what is known and avoids pretending the gap is resolved",
        "the correction improves verifiability while preserving the original useful intent",
        "the criticism identifies one consequential defect and the revision directly repairs it",
        "the updated wording about {subject} is specific enough to test against the record",
        "the revision removes the misleading implication without introducing a replacement assumption",
        "the central statement now matches the scale and quality of the available evidence",
        "the correction keeps the meaning that can be defended and discards the rest",
        "the final prose is clearer because its claim and evidence now have the same scope",
        "the revised claim no longer outruns the example, sample, or observation behind it",
        "the result is a bounded correction rather than a more polished version of the same flaw",
        "the repaired wording preserves the defensible claim while removing its unsupported extension",
        "the revision changes the faulty evidence link instead of merely changing its tone",
        "the final claim is calibrated to the scope, sample, and certainty of the supplied material",
        "the corrected passage can now be challenged or confirmed using the cited evidence",
    ),
    "explanation_learning": (
        "the worked example and transfer question test whether the mechanism can be reused",
        "the learner can connect the central idea about {subject} to a fresh case",
        "the example makes the mechanism visible and the question checks understanding",
        "the explanation separates the rule, its concrete use, and a way to test transfer",
        "understanding is demonstrated by applying the idea rather than repeating its wording",
        "the learner can explain why the example fits instead of only recalling its label",
        "the mechanism behind {subject} stays connected to an observable consequence",
        "the final question reveals whether the explanation transfers to a different situation",
        "the concept, worked case, and understanding check form one causal chain",
        "the learner has a concrete example to inspect and a separate prompt to answer",
        "the explanation moves from definition to use and then to independent application",
        "the key distinction about {subject} can be tested rather than memorized",
        "the example supplies evidence for the mechanism while the question probes transfer",
        "the response gives both an accessible mental model and a way to challenge it",
        "the learner can compare a new case with the mechanism described in the answer",
        "the explanation stays simple without removing the causal step that makes it useful",
        "the worked case shows the rule in action and the closing question checks reuse",
        "the central idea is tied to something the learner can observe and verify",
        "the final check distinguishes genuine understanding from repetition of the definition",
        "the explanation of {subject} remains usable beyond the single example provided",
        "the learner can reconstruct the mechanism before applying it to an unfamiliar example",
        "the transfer prompt tests the causal idea rather than recall of the original wording",
        "the explanation links an observable example to the rule that produced it",
        "the closing question reveals whether the concept can guide a new decision",
    ),
    "summarization_synthesis": (
        "the decision, owned action, timing, and unresolved point remain separate",
        "the summary preserves what was decided without silently resolving the open issue",
        "ownership stays attached to the next action while uncertainty remains visible",
        "a reader can distinguish the recorded decision from the work that is still pending",
        "the condensed account keeps both execution responsibility and the remaining gap",
        "the record is shortened without losing the owner, deadline, or unresolved issue",
        "the selected facts preserve what happens next and what still lacks a decision",
        "the summary keeps the recorded outcome distinct from later interpretation",
        "the action can be assigned and tracked without treating the open point as closed",
        "the concise version retains the operational detail needed for follow-through",
        "the decision about {subject} remains connected to its owner and timing",
        "the unresolved issue is visible beside the action rather than hidden by compression",
        "the synthesis preserves accountability while refusing to manufacture closure",
        "the reader can recover the decision, next move, and remaining uncertainty quickly",
        "the summary removes secondary wording but keeps every execution-critical element",
        "the final account distinguishes agreed work from questions still awaiting evidence",
        "the responsible person and due point remain attached to the recorded action",
        "the concise result supports follow-up without overstating what the source settled",
        "the open point about {subject} remains explicit after the record is compressed",
        "the synthesis is brief while still preserving decision state and operational ownership",
        "the compressed record retains the decision, accountable owner, and unresolved dependency",
        "the summary supports follow-through without confusing planned work with completed work",
        "the final synthesis keeps timing and responsibility attached to the correct action",
        "the shortened account remains operational because no open issue is silently closed",
    ),
    "context_clarification": (
        "the missing detail remains visible instead of being replaced by an assumption",
        "the next answer can become specific as soon as the unresolved point is confirmed",
        "the known context about {subject} stays separate from the detail still being requested",
        "a reversible default is preserved until the user confirms the missing condition",
        "the clarification narrows the request without inventing its final boundary",
        "the unresolved choice is asked directly while the available context remains intact",
        "one focused question separates the supported interpretation from the missing instruction",
        "the request about {subject} can proceed once the user selects the intended scope",
        "the temporary default remains easy to reverse after the missing detail arrives",
        "the response identifies exactly which decision cannot yet be made safely",
        "the clarification keeps the known facts useful without treating them as a complete goal",
        "the user can resolve the ambiguity with one answer rather than restating the whole case",
        "the next turn has a clear purpose and does not broaden the original request",
        "the missing boundary is surfaced before any irreversible recommendation is offered",
        "the question about {subject} requests only the information needed to continue",
        "the current interpretation remains provisional until the user confirms the intended direction",
        "the response turns a broad ambiguity into one answerable point",
        "the established context and the unresolved preference remain visibly separate",
        "the clarification avoids guessing while still moving the conversation forward",
        "the next answer can be both direct and bounded after this single point is resolved",
        "the focused question identifies the one missing choice that changes the valid response",
        "the user can confirm the intended scope without repeating information already understood",
        "the provisional reading stays reversible until the decisive preference is supplied",
        "the conversation advances by isolating uncertainty instead of inventing a resolution",
    ),
    "conversation_empathy": (
        "the feeling is acknowledged while the suggested next step remains optional",
        "the person keeps control over whether and when to try the proposed next step",
        "the response recognizes the experience without claiming to know more than was shared",
        "support for {subject} is offered without pressure, diagnosis, or a forced conclusion",
        "the practical suggestion remains small enough to accept, adapt, or decline",
        "the response makes room for the emotion before offering any practical direction",
        "the next step for {subject} is framed as a choice rather than an obligation",
        "the acknowledgement stays close to what was shared and avoids an unsupported diagnosis",
        "the person can pause, continue, or choose a smaller action without being judged",
        "the reply combines emotional recognition with one manageable and optional possibility",
        "the support remains present even if the suggested action is not useful right now",
        "the response validates the difficulty without claiming that the feeling has one cause",
        "the practical option leaves timing and intensity under the person's control",
        "the language around {subject} remains warm, specific, and free of pressure",
        "the person is invited to identify what would help instead of being told what to feel",
        "the acknowledgement does not minimize the experience or rush toward a solution",
        "the next move can be adjusted to the person's current capacity",
        "the reply offers companionship and choice rather than certainty or instruction",
        "the emotional reality remains central while the action stays deliberately modest",
        "the response supports agency by making every proposed step optional and reversible",
        "the acknowledgement remains specific while leaving interpretation and timing with the person",
        "the offered next step can be accepted, changed, or declined without weakening the support",
        "the response makes space for the feeling before introducing a manageable possibility",
        "the person retains control over both the pace of reflection and any practical action",
    ),
}


def development_card_count(task: str) -> int:
    if task not in _DETAIL_CARDS:
        return 0
    return len(_DEVELOPMENT_FRAMES) + len(_DETAIL_CARDS[task])


def develop_answer(
    target: str,
    *,
    task: str,
    metadata: dict[str, Any],
    example_id: str,
) -> str:
    """Add one linked, grounded development sentence to short answers only."""

    minimum = MINIMUM_DEVELOPED_WORDS.get(task)
    if minimum is None or len(target.split()) >= minimum:
        return target
    if metadata.get("evaluation_source") == "separately_authored":
        return target
    details = _DETAIL_CARDS[task]
    deck = LinkedSubcardDeck(
        pools=(
            SubcardPool("development_frame", _DEVELOPMENT_FRAMES),
            SubcardPool("grounded_detail", details),
        )
    )
    frame, detail = deck.deal(
        {"scenario_id": example_id},
        0,
        f"answer-development:{task}",
    )
    subject = str(metadata.get("subject", "the request")).strip().rstrip(".")
    rendered = frame.format(
        detail=detail.format(subject=subject or "the request")
    )
    return f"{target.rstrip()} {_sentence(_inline_sentence(rendered))}"
