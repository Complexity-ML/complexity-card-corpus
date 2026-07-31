from __future__ import annotations

import random
from collections import Counter
from typing import Protocol

from .english_morphology import (
    correct_indefinite_articles,
    verb_forms,
)


NARRATIVE_FRAME_IDS = tuple(f"frame_{index:02d}" for index in range(1, 13))


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
        f"A decision about {subject} now requires attention because {lower_first(state_text)}.",
        f"The request concerning {subject} reaches a clear turning point: {state_text}.",
        f"One concrete update changes the direction of the task: {state_text}.",
        f"The {domain.label.lower()} request can no longer proceed routinely because {lower_first(state_text)}.",
        f"A review of {subject} reveals the condition that matters most: {state_text}.",
        f"The immediate choice for {subject} is shaped by one verified fact: {state_text}.",
        f"Handling {subject} becomes a bounded decision with this verified update: {state_text}.",
        f"The immediate issue in this {domain.label.lower()} case is specific: {state_text}.",
        f"A checkpoint for {subject} exposes the operative condition: {state_text}.",
        f"The available state calls for a different approach: {state_text}.",
        f"This request about {subject} becomes actionable from one update: {state_text}.",
        f"The {domain.label.lower()} scenario begins with a decisive change: {state_text}.",
    )
    contexts = (
        f"The surrounding context keeps this {domain.label.lower()} request grounded: {context_text}.",
        f"The setting provides the evidence needed to assess {subject}: {context_text}.",
        f"This decision depends on concrete background: {context_text}.",
        f"The practical setting remains tied to verifiable facts: {context_text}.",
        f"The wider context narrows what can be concluded responsibly: {context_text}.",
        f"This background provides a basis for reviewing {subject}: {context_text}.",
        f"One background fact protects the decision from guesswork: {context_text}.",
        f"The domain context keeps the requested action evidence-bound: {context_text}.",
        f"The available context defines the task without inviting assumptions: {context_text}.",
        f"The request sits within a setting that requires observable support: {context_text}.",
        f"This background rules out a generic treatment of {subject}: {context_text}.",
        f"The case retains a practical context for reviewing {subject}: {context_text}.",
    )
    tasks = (
        f"For {subject}, the objective is to {intent_base} using only supported details.",
        f"The work now focuses on one concrete, reviewable attempt to {intent_base} for {subject}.",
        f"The bounded task is to {intent_base} for {subject} without speculation.",
        f"Known facts provide the basis to {intent_base} for {subject}.",
        f"The requested action is to {intent_base} for {subject} with a clear rationale.",
        f"A practical step is to {intent_base} for {subject} and preserve an evidence trail.",
        f"This case calls for a careful attempt to {intent_base} for {subject}.",
        f"The practical objective is to {intent_base} for {subject} with reviewable support.",
        f"Verified information sets the scope for trying to {intent_base} for {subject}.",
        f"Only the available evidence can support an attempt to {intent_base} for {subject}.",
        f"The task is to {intent_base} for {subject} and make the basis visible.",
        f"The intended task is to {intent_base} for {subject} without overclaiming.",
    )
    boundaries = (
        f"One explicit boundary governs that work: {constraint_text}.",
        f"One non-negotiable boundary must remain intact: {constraint_text}.",
        f"The task remains subject to one firm rule: {constraint_text}.",
        f"No proposed step may cross this stated boundary: {constraint_text}.",
        f"One condition limits the requested action: {constraint_text}.",
        f"This requirement must remain intact: {constraint_text}.",
        f"This case also carries one clear limit: {constraint_text}.",
        f"The practical objective remains bounded by this rule: {constraint_text}.",
        f"Every grounded option must comply with this condition: {constraint_text}.",
        f"This boundary applies to every option: {constraint_text}.",
        f"One requirement must remain intact throughout: {constraint_text}.",
        f"The intended task cannot override this condition: {constraint_text}.",
    )
    outcomes = (
        f"A successful result for this {domain.label.lower()} case means that {lower_first(outcome_text)}.",
        f"The evidence is sufficient only if it shows that {lower_first(outcome_text)}.",
        f"Completion requires evidence that {lower_first(outcome_text)}.",
        f"The final check for {subject} is whether {lower_first(outcome_text)}.",
        f"A valid endpoint for this request makes it true that {lower_first(outcome_text)}.",
        f"The task is complete once the evidence demonstrates that {lower_first(outcome_text)}.",
        f"The result remains acceptable only if {lower_first(outcome_text)}.",
        f"The completion rule for {subject} is that {lower_first(outcome_text)}.",
        f"Acceptance depends on showing through evidence that {lower_first(outcome_text)}.",
        f"The outcome is valid only when the record shows that {lower_first(outcome_text)}.",
        f"The closing evidence needs to demonstrate that {lower_first(outcome_text)}.",
        f"The case closes successfully once the record shows that {lower_first(outcome_text)}.",
    )
    statement_endings = (
        f"Blocked progress on {intent_base} for {subject} calls for this fallback: {fallback_text}.",
        f"Incomplete support for {intent_base} around {subject} calls for this fallback: {fallback_text}.",
        f"An unconfirmed safe scope for {intent_base} around {subject} calls for this fallback: {fallback_text}.",
        f"Failed verification of an attempt to {intent_base} for {subject} calls for this fallback: {fallback_text}.",
        f"An insufficient record for {intent_base} around {subject} calls for this fallback: {fallback_text}.",
        f"Reliable support for {intent_base} around {subject} is no longer available, so use this fallback: {fallback_text}.",
        f"Unresolved evidence around {intent_base} for {subject} calls for this fallback: {fallback_text}.",
        f"A failed completion check for {intent_base} around {subject} calls for this fallback: {fallback_text}.",
        f"Insufficient proof for {intent_base} around {subject} calls for this fallback: {fallback_text}.",
        f"Unresolved uncertainty around {intent_base} for {subject} calls for this fallback: {fallback_text}.",
        f"Weak support for {intent_base} around {subject} calls for this fallback: {fallback_text}.",
        f"A case that cannot support {intent_base} for {subject} calls for this fallback: {fallback_text}.",
    )
    question_endings = (
        f"Given these limits, how would you {intent_base} for {subject} and keep this fallback available for insufficient evidence: {fallback_text}?",
        f"How would you {intent_base} for {subject} with this fallback available for incomplete support: {fallback_text}?",
        f"What grounded approach would {intent_base} for {subject} and retain this fallback: {fallback_text}?",
        f"How can you {intent_base} for {subject} and preserve this fallback for failed verification: {fallback_text}?",
        f"What is a careful way to {intent_base} for {subject}, with this fallback ready for an insufficient record: {fallback_text}?",
        f"How would you {intent_base} for {subject} and retain this fallback for lost support: {fallback_text}?",
        f"What grounded step could {intent_base} for {subject} and preserve this fallback for unresolved evidence: {fallback_text}?",
        f"How can you {intent_base} for {subject} with this fallback for a failed completion check: {fallback_text}?",
        f"What supported approach would {intent_base} for {subject}, with this fallback for insufficient proof: {fallback_text}?",
        f"How would you {intent_base} for {subject} and keep this fallback for unresolved uncertainty: {fallback_text}?",
        f"What bounded step would {intent_base} for {subject} and leave this fallback available: {fallback_text}?",
        f"How can you {intent_base} for {subject} and preserve this fallback for a case that cannot support further work: {fallback_text}?",
    )
    endings = (
        question_endings
        if family_id in {"conversation_empathy", "explanation_learning"}
        else statement_endings
    )

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
