from __future__ import annotations

from typing import Final


# These are original Complexity formulations.  The source fallback remains the
# semantic contract; these variants change only its surface realization.
FALLBACK_ACTIONS: Final[dict[str, tuple[str, ...]]] = {
    "Pause before commitment and preserve the current confirmed state": (
        "pause before making a commitment and leave the latest confirmed state intact",
        "hold back commitment while keeping the current verified state unchanged",
        "defer commitment and preserve the most recent confirmed state",
        "keep the decision open without altering what has already been confirmed",
        "avoid committing yet and protect the last verified state",
        "leave the confirmed state untouched until commitment is justified",
    ),
    "Return to a smaller causal model with fewer moving parts": (
        "return to a smaller causal model with fewer interacting parts",
        "reduce the analysis to a smaller causal model with fewer moving pieces",
        "step back to a simpler causal model that contains fewer dependencies",
        "use a narrower causal model with fewer components to reason about",
        "rebuild the explanation around a smaller causal chain",
        "simplify the causal account until only the essential moving parts remain",
    ),
    "Stop changing state, preserve logs, and record the last confirmed boundary": (
        "stop further state changes, preserve the logs, and mark the last confirmed boundary",
        "freeze state changes, retain the logs, and document the last verified boundary",
        "make no additional changes; keep the logs and record the last known boundary",
        "halt mutation, protect the logs, and note the most recent verified boundary",
        "leave state untouched while preserving logs and writing down the confirmed boundary",
        "suspend further changes, retain the evidence, and capture the latest confirmed boundary",
    ),
    "Stay with acknowledgment and do not add an action step yet": (
        "acknowledge what was expressed without introducing an action step yet",
        "remain with acknowledgment and postpone action planning",
        "focus on recognition for now rather than adding a next step",
        "reflect the experience without moving into problem-solving yet",
        "keep the response at acknowledgment until an action is invited",
        "validate the stated experience and leave action for later",
    ),
    "Request the missing fact instead of inventing a bridge sentence": (
        "ask for the missing fact rather than filling the gap with an invented connection",
        "request the needed detail instead of guessing how the pieces connect",
        "identify and ask for the absent fact before composing a bridge",
        "leave the gap visible and seek the missing evidence",
        "pause composition to obtain the specific missing fact",
        "replace a speculative bridge with a direct clarification request",
    ),
    "Use the provider's official support channel with a concise evidence summary": (
        "contact the provider's official support channel with a concise evidence summary",
        "send a compact evidence summary through the provider's verified support route",
        "use the provider's authorized help channel and include only the key evidence",
        "escalate through official provider support with a brief documented account",
        "present the concise evidence to the provider's recognized support service",
        "route the issue to the provider's official team with a focused summary",
    ),
    "Retain the caveat explicitly when shortening the text": (
        "keep the caveat explicit while reducing the length",
        "preserve the qualification in any shorter version",
        "shorten the text without removing its stated caveat",
        "carry the uncertainty notice into the concise wording",
        "make the caveat visible even after compression",
        "retain the limiting qualification in the edited text",
    ),
    "Choose a smaller reversible scope that still tests the plan": (
        "choose a smaller reversible scope that still tests the plan",
        "reduce the scope to a reversible trial that can test the proposal",
        "use a narrower experiment that preserves reversibility",
        "test the plan through a smaller step that can be undone",
        "select a limited reversible action that still produces evidence",
        "shrink the trial while keeping its result informative",
    ),
    "Defer commitment until the decisive unknown is resolved": (
        "defer commitment until the decisive unknown is resolved",
        "leave the commitment open until the key uncertainty is settled",
        "postpone the decision until the decisive fact is known",
        "withhold commitment while the central unknown remains unresolved",
        "wait for the missing decisive evidence before committing",
        "keep the choice provisional until the critical uncertainty is closed",
    ),
    "Provide only general educational information and verification steps": (
        "provide only general educational information and verification steps",
        "limit the response to general guidance and ways to verify it",
        "offer broad educational context together with checking steps",
        "keep the answer informational and point to independent verification",
        "share general principles without personalized direction, plus checks",
        "restrict assistance to educational material and verification methods",
    ),
    "Use a counterexample to expose the exact point of confusion": (
        "use a counterexample to expose the exact point of confusion",
        "test the misconception with a counterexample that isolates the error",
        "show one contrasting case that reveals where the reasoning fails",
        "locate the confusion through a carefully chosen counterexample",
        "introduce an exception that makes the mistaken step visible",
        "contrast the claim with a case that pinpoints the faulty inference",
    ),
    "Recommend immediate local emergency help when there is imminent danger": (
        "recommend immediate local emergency help when danger is imminent",
        "direct the person to immediate nearby emergency assistance",
        "prioritize urgent local help when the risk is immediate",
        "advise contacting local emergency services without delay",
        "move first toward immediate in-person emergency support",
        "state the need for urgent local assistance when danger is present",
    ),
    "Direct the user to an official or qualified support channel": (
        "direct the user to an official or qualified support channel",
        "refer the request to a recognized source of qualified support",
        "point the user toward the appropriate official assistance route",
        "move the question to a verified professional support channel",
        "identify an authorized or qualified service that can take over",
        "guide the user to the relevant official support provider",
    ),
    "Name the unresolved claim and suggest an authoritative source check": (
        "name the unresolved claim and suggest an authoritative source check",
        "identify the unsupported claim and verify it against an authoritative source",
        "make the open claim explicit before recommending a trusted reference check",
        "flag the unresolved assertion and point to a primary source for confirmation",
        "state what remains unverified and propose an authoritative check",
        "isolate the disputed claim and seek confirmation from a trusted source",
    ),
}


FALLBACK_FRAMES: Final[tuple[str, ...]] = (
    "If support is insufficient, {action}.",
    "When the evidence remains incomplete, {action}.",
    "If the acceptance check fails, {action}.",
    "If verification does not hold, {action}.",
    "Should the record stay inconclusive, {action}.",
    "If material uncertainty remains, {action}.",
    "When a decisive fact is missing, {action}.",
    "If the proposed result cannot be confirmed, {action}.",
    "When support falls short of the decision threshold, {action}.",
    "If the available facts do not justify completion, {action}.",
    "If the boundary cannot be verified, {action}.",
    "When the result is not adequately supported, {action}.",
    "If the final check exposes a gap, {action}.",
    "If evidence conflicts at the point of decision, {action}.",
    "When the answer would otherwise require a guess, {action}.",
    "If confidence is not backed by the record, {action}.",
    "When the requested conclusion exceeds the evidence, {action}.",
    "If a reliable result is still out of reach, {action}.",
    "When the confirmation step remains unresolved, {action}.",
    "If the next move would overstate what is known, {action}.",
    "When the evidence does not close the case, {action}.",
    "If review leaves the central uncertainty open, {action}.",
    "When the completion criterion is not met, {action}.",
    "If the supported scope is narrower than requested, {action}.",
)


CONCLUSION_FRAMES: Final[tuple[str, ...]] = (
    "The required result is clear: {outcome}.",
    "Use this as the completion criterion: {outcome}.",
    "The acceptance check is explicit: {outcome}.",
    "Completion depends on one observable result: {outcome}.",
    "The response is ready only after confirming this: {outcome}.",
    "The final review should establish the following: {outcome}.",
    "Treat the task as complete when this result holds: {outcome}.",
    "The evidence should support this concrete outcome: {outcome}.",
    "Close the task only after verifying this result: {outcome}.",
    "The decision rule is whether this has been achieved: {outcome}.",
    "A valid answer produces this checkable result: {outcome}.",
    "The last checkpoint is straightforward: {outcome}.",
    "Success is bounded by this observable condition: {outcome}.",
    "The result can be accepted once this is true: {outcome}.",
    "Verification should end with this finding: {outcome}.",
    "The closing evidence must demonstrate this: {outcome}.",
    "Resolve the request against this criterion: {outcome}.",
    "The answer remains provisional until this is confirmed: {outcome}.",
    "Use the following condition to decide whether to finish: {outcome}.",
    "The practical definition of done is this: {outcome}.",
    "One result determines acceptance: {outcome}.",
    "Before concluding, verify this outcome: {outcome}.",
    "The response should leave behind this confirmed result: {outcome}.",
    "The final evidence test is the following: {outcome}.",
)


def fallback_actions(label: str) -> tuple[str, ...]:
    """Return original, human-authored realizations for one semantic fallback."""
    normalized = label.rstrip(".")
    try:
        return FALLBACK_ACTIONS[normalized]
    except KeyError as exc:
        raise ValueError(f"missing post-training fallback language for: {label}") from exc
