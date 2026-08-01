from __future__ import annotations

import random
from collections import Counter
from typing import Protocol

from .english_morphology import (
    correct_indefinite_articles,
    verb_forms,
)


NARRATIVE_FRAME_IDS = tuple(f"frame_{index:02d}" for index in range(1, 13))
QUESTION_FRAME_IDS = frozenset({"frame_02", "frame_06", "frame_10"})


def uses_question_surface(frame_id: str) -> bool:
    """Reserve one quarter of narrative frames for direct questions."""

    return frame_id in QUESTION_FRAME_IDS


class DomainLike(Protocol):
    domain_id: str
    label: str
    subject: str
    context: str


class IntentLike(Protocol):
    label: str


class AtomLike(Protocol):
    atom_id: str
    label: str


def lower_first(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    return value[:1].lower() + value[1:]


def upper_first(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    return value[:1].upper() + value[1:]


class DynamicNarrativeComposer:
    """Choose compatible language dynamically while controlling repetition."""

    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed)
        self._global_usage: Counter[str] = Counter()
        self._domain_usage: Counter[tuple[str, str]] = Counter()
        self._previous_frame: str | None = None

    def _choose_frame(self, domain_id: str) -> tuple[int, str]:
        candidates = []
        for index, frame_id in enumerate(NARRATIVE_FRAME_IDS):
            score = (
                self._domain_usage[(domain_id, frame_id)],
                self._global_usage[frame_id],
                int(frame_id == self._previous_frame),
            )
            candidates.append((score, index, frame_id))
        best_score = min(item[0] for item in candidates)
        tied = [item for item in candidates if item[0] == best_score]
        _, index, frame_id = self._random.choice(tied)
        self._global_usage[frame_id] += 1
        self._domain_usage[(domain_id, frame_id)] += 1
        self._previous_frame = frame_id
        return index, frame_id

    @property
    def frame_counts(self) -> dict[str, int]:
        return dict(self._global_usage)

    def compose(
        self,
        family_id: str,
        domain: DomainLike,
        intent: IntentLike,
        constraint: AtomLike,
        state: AtomLike,
        outcome: AtomLike,
        fallback: AtomLike,
    ) -> tuple[str, str, str]:
        """Compose prose after semantic compatibility has already been resolved."""
        frame_index, frame_id = self._choose_frame(domain.domain_id)
        return _render_frame(
            frame_index,
            frame_id,
            family_id,
            domain,
            intent,
            constraint,
            state,
            outcome,
            fallback,
        )


def _render_frame(
    frame_index: int,
    frame_id: str,
    family_id: str,
    domain: DomainLike,
    intent: IntentLike,
    constraint: AtomLike,
    state: AtomLike,
    outcome: AtomLike,
    fallback: AtomLike,
) -> tuple[str, str, str]:
    """Render four compact sentences with each hard semantic anchor stated once."""
    subject = domain.subject
    context_text = domain.context.rstrip(".")
    state_text = state.label.rstrip(".")
    constraint_text = constraint.label.rstrip(".")
    outcome_text = outcome.label.rstrip(".")
    fallback_text = fallback.label.rstrip(".")
    intent_forms = verb_forms(intent.label)
    intent_base = intent_forms["base"]

    triggers = (
        f"A decision about {subject} changes because {lower_first(state_text)}.",
        f"The request concerning {subject} reaches a turning point: {state_text}.",
        f"One update changes the task: {state_text}.",
        f"Routine progress on {subject} stops because {lower_first(state_text)}.",
        f"A review of {subject} reveals the key condition: {state_text}.",
        f"One verified fact shapes the choice for {subject}: {state_text}.",
        f"Handling {subject} becomes bounded by this update: {state_text}.",
        f"The immediate issue for {subject} is specific: {state_text}.",
        f"A checkpoint for {subject} exposes the condition: {state_text}.",
        f"The current state calls for a new approach: {state_text}.",
        f"One update makes {subject} actionable: {state_text}.",
        f"The scenario for {subject} begins with a change: {state_text}.",
    )
    contexts = (
        f"Context for {intent_base} in {domain.label}: {context_text}.",
        f"Evidence for {intent_base} in {domain.label} comes from here: {context_text}.",
        f"In {domain.label}, {intent_base} depends on this background: {context_text}.",
        f"Context bounds {intent_base} in {domain.label}: {context_text}.",
        f"This setting narrows {intent_base} in {domain.label}: {context_text}.",
        f"Reviewing {intent_base} in {domain.label} starts here: {context_text}.",
        f"Background for {intent_base} in {domain.label} prevents guesswork: {context_text}.",
        f"Domain context keeps {intent_base} in {domain.label} evidence-bound: {context_text}.",
        f"Known context defines {intent_base} in {domain.label}: {context_text}.",
        f"Observable evidence for {intent_base} in {domain.label} comes from here: {context_text}.",
        f"This background prevents generic {intent_base} in {domain.label}: {context_text}.",
        f"Practical context for {intent_base} in {domain.label} is explicit: {context_text}.",
    )
    tasks = (
        f"For {subject}, {intent_base} using only supported details.",
        f"Make one reviewable attempt to {intent_base} for {subject}.",
        f"The task is to {intent_base} for {subject} without speculation.",
        f"Use known facts to {intent_base} for {subject}.",
        f"The request is to {intent_base} for {subject} with a rationale.",
        f"Take one step to {intent_base} for {subject} with an evidence trail.",
        f"This case needs a careful attempt to {intent_base} for {subject}.",
        f"The objective is to {intent_base} for {subject} with reviewable support.",
        f"Verified information bounds {intent_base} for {subject}.",
        f"Current evidence supports one attempt to {intent_base} for {subject}.",
        f"The task is to {intent_base} for {subject} with a visible basis.",
        f"The task is to {intent_base} for {subject} without overclaiming.",
    )
    boundaries = (
        f"For {intent_base} in {domain.label}, one boundary applies: {constraint_text}.",
        f"One rule remains for {intent_base} in {domain.label}: {constraint_text}.",
        f"In {domain.label}, {intent_base} follows one rule: {constraint_text}.",
        f"No attempt at {intent_base} in {domain.label} may cross: {constraint_text}.",
        f"One condition limits {intent_base} in {domain.label}: {constraint_text}.",
        f"In {domain.label}, {intent_base} preserves this requirement: {constraint_text}.",
        f"The attempt at {intent_base} in {domain.label} carries this limit: {constraint_text}.",
        f"The goal of {intent_base} in {domain.label} follows this rule: {constraint_text}.",
        f"Every option for {intent_base} in {domain.label} must satisfy: {constraint_text}.",
        f"This boundary applies to {intent_base} in {domain.label}: {constraint_text}.",
        f"One requirement remains for {intent_base} in {domain.label}: {constraint_text}.",
        f"In {domain.label}, {intent_base} cannot override this condition: {constraint_text}.",
    )
    outcomes = (
        f"For {intent_base} in {domain.label}, success means {lower_first(outcome_text)}.",
        f"Support for {intent_base} in {domain.label} is sufficient only if {lower_first(outcome_text)}.",
        f"Completing {intent_base} in {domain.label} requires that {lower_first(outcome_text)}.",
        f"The final check on {intent_base} in {domain.label} is whether {lower_first(outcome_text)}.",
        f"A valid endpoint for {intent_base} in {domain.label} shows that {lower_first(outcome_text)}.",
        f"In {domain.label}, {intent_base} is complete once {lower_first(outcome_text)}.",
        f"A result from {intent_base} in {domain.label} is acceptable only if {lower_first(outcome_text)}.",
        f"The completion rule for {intent_base} in {domain.label} is that {lower_first(outcome_text)}.",
        f"Acceptance of {intent_base} in {domain.label} requires that {lower_first(outcome_text)}.",
        f"The outcome of {intent_base} in {domain.label} is valid when {lower_first(outcome_text)}.",
        f"Closing {intent_base} in {domain.label} requires that {lower_first(outcome_text)}.",
        f"The case for {intent_base} in {domain.label} closes once {lower_first(outcome_text)}.",
    )
    statement_endings = (
        f"If {intent_base} for {subject} remains blocked, use this fallback: {fallback_text}.",
        f"If support for {intent_base} around {subject} is incomplete, follow: {fallback_text}.",
        f"Without a safe scope for {intent_base} around {subject}, follow: {fallback_text}.",
        f"If verification of {intent_base} for {subject} fails, follow: {fallback_text}.",
        f"With an insufficient record for {intent_base} around {subject}, follow: {fallback_text}.",
        f"If support for {intent_base} around {subject} disappears, follow: {fallback_text}.",
        f"When evidence for {intent_base} around {subject} remains unresolved, follow: {fallback_text}.",
        f"If the completion check on {intent_base} for {subject} fails, follow: {fallback_text}.",
        f"Without enough proof for {intent_base} around {subject}, follow: {fallback_text}.",
        f"When uncertainty around {intent_base} for {subject} remains, follow: {fallback_text}.",
        f"If support for {intent_base} around {subject} stays weak, follow: {fallback_text}.",
        f"If the case cannot support {intent_base} for {subject}, follow: {fallback_text}.",
    )
    question_endings = (
        f"How would you {intent_base} for {subject} under this fallback condition: {fallback_text}?",
        f"How would you {intent_base} for {subject} with this recovery option: {fallback_text}?",
        f"What grounded approach would {intent_base} for {subject} given this fallback: {fallback_text}?",
        f"How can you {intent_base} for {subject} within this recovery boundary: {fallback_text}?",
        f"What careful approach would {intent_base} for {subject} with this backup step: {fallback_text}?",
        f"How would you {intent_base} for {subject} given this contingency: {fallback_text}?",
        f"What grounded step could {intent_base} for {subject} under this fallback: {fallback_text}?",
        f"How can you {intent_base} for {subject} with this alternative available: {fallback_text}?",
        f"What supported approach would {intent_base} for {subject} given this recovery path: {fallback_text}?",
        f"How would you {intent_base} for {subject} with this safeguard: {fallback_text}?",
        f"What bounded step would {intent_base} for {subject} under this contingency: {fallback_text}?",
        f"How can you {intent_base} for {subject} with this fallback available: {fallback_text}?",
    )
    endings = question_endings if uses_question_surface(frame_id) else statement_endings

    trigger = correct_indefinite_articles(triggers[frame_index])
    situation = " ".join(
        correct_indefinite_articles(value)
        for value in (
            trigger,
            contexts[frame_index],
            tasks[frame_index],
            boundaries[frame_index],
            outcomes[frame_index],
            endings[frame_index],
        )
    )
    return trigger, situation, frame_id


def compose_title(
    domain: DomainLike,
    intent: IntentLike,
    constraint: AtomLike,
    state: AtomLike,
    outcome: AtomLike,
) -> str:
    """Render a readable title that retains every semantic axis."""
    return (
        f"{domain.label} — {intent.label}: "
        f"{state.atom_id.replace('_', ' ')} / {constraint.atom_id.replace('_', ' ')} "
        f"→ {outcome.atom_id.replace('_', ' ')}"
    )
