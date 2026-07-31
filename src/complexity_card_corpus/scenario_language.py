from __future__ import annotations

import random
from collections import Counter
from typing import Protocol

from .english_morphology import (
    VerbFeatures,
    realize_clause,
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
    subject = domain.subject
    context_text = domain.context.rstrip(".")
    state_text = state.label.rstrip(".")
    constraint_text = constraint.label.rstrip(".")
    outcome_text = outcome.label.rstrip(".")
    fallback_text = fallback.label.rstrip(".")
    intent_forms = verb_forms(intent.label)
    intent_base = intent_forms["base"]
    intent_third = intent_forms["third_person_singular"]
    intent_negative = realize_clause(
        "the assistant",
        intent.label,
        VerbFeatures(negated=True),
    )
    intent_progressive = realize_clause(
        "the assistant",
        intent.label,
        VerbFeatures(aspect="progressive"),
    )
    intent_should_question = realize_clause(
        "the response",
        intent.label,
        VerbFeatures(modal="should", interrogative=True),
    )
    intent_can_question = realize_clause(
        "the response",
        intent.label,
        VerbFeatures(modal="can", interrogative=True),
    )

    triggers = (
        f"A routine step involving {subject} reaches a decision point when {lower_first(state_text)}.",
        f"The situation around {subject} changes because {lower_first(state_text)}.",
        f"A new obstacle appears while handling {subject}: {state_text}.",
        f"Work on {subject} can no longer continue unchanged after this update: {state_text}.",
        f"The decisive signal in this {domain.label.lower()} case is that {lower_first(state_text)}.",
        f"An otherwise ordinary request about {subject} becomes non-routine when {lower_first(state_text)}.",
        f"Before the next step for {subject}, one fact changes the shape of the request: {state_text}.",
        f"The immediate trigger for this {domain.label.lower()} scenario is simple: {state_text}.",
        f"A checkpoint is reached for {subject} once {lower_first(state_text)}.",
        f"The next decision about {subject} is prompted by one concrete condition: {state_text}.",
        f"A request involving {subject} becomes actionable when {lower_first(state_text)}.",
        f"This {domain.label.lower()} case begins from a specific turning point: {state_text}.",
    )
    developments = (
        f"Given that {lower_first(state_text)}, the relevant context is that {lower_first(context_text)}. The next move for {subject} is to {intent_base} without crossing “{constraint_text}”.",
        f"Here, {lower_first(context_text)}, while {lower_first(state_text)}. For {subject}, the immediate task is to {intent_base} within “{constraint_text}”.",
        f"The setting matters because {lower_first(context_text)} and {lower_first(state_text)}. A useful answer about {subject} should {intent_base} under “{constraint_text}”.",
        f"The operative fact is that {lower_first(state_text)}, in a setting where {lower_first(context_text)}. The request about {subject} is to {intent_base} while retaining “{constraint_text}”.",
        f"The case combines this context—{context_text}—with the fact that {lower_first(state_text)}. Together they call for an answer about {subject} that {intent_third} without violating “{constraint_text}”.",
        f"Assumptions would be unreliable because {lower_first(context_text)} and {lower_first(state_text)}. The response about {subject} should {intent_base} from the stated facts and honor “{constraint_text}”.",
        f"What is known is that {lower_first(context_text)}, while {lower_first(state_text)}. What is needed for {subject} is a way to {intent_base} that respects “{constraint_text}”.",
        f"The current condition—{state_text}—sets the practical boundary in a context where {lower_first(context_text)}. The objective for {subject} is to {intent_base} under “{constraint_text}”.",
        f"The broad subject is {subject}, and the current fact is that {lower_first(state_text)}. The actionable task is to {intent_base} within the limit “{constraint_text}”.",
        f"In this setting, {lower_first(context_text)}, and {lower_first(state_text)}. A grounded response about {subject} must {intent_base} without crossing “{constraint_text}”.",
        f"The context is that {lower_first(context_text)}, combined with the fact that {lower_first(state_text)}. For {subject}, the response {intent_third} while preserving “{constraint_text}”.",
        f"A broad answer about {subject} would miss that {lower_first(state_text)} in a setting where {lower_first(context_text)}. The response must {intent_base} under “{constraint_text}”.",
    )

    if family_id in {"conversation_empathy", "explanation_learning"}:
        endings = (
            f"What response can {intent_base} while respecting “{constraint_text}” and still establish that {lower_first(outcome_text)}?",
            f"How can the answer respect the boundary “{constraint_text}” while helping to {intent_base} and reaching a point where {lower_first(outcome_text)}?",
            f"What would an answer look like if it must {intent_base}, honor “{constraint_text}”, and ensure that {lower_first(outcome_text)}?",
            f"Which response best combines the task to {intent_base} with the boundary “{constraint_text}” and the result that {lower_first(outcome_text)}?",
            f"{upper_first(intent_should_question)} without crossing “{constraint_text}”, so that {lower_first(outcome_text)}?",
            f"What is the clearest way to {intent_base} while keeping “{constraint_text}” intact and making sure that {lower_first(outcome_text)}?",
            f"{upper_first(intent_can_question)}, respect the boundary “{constraint_text}”, and leave the situation such that {lower_first(outcome_text)}?",
            f"What answer would respect “{constraint_text}” yet still {intent_base} and produce the result that {lower_first(outcome_text)}?",
            f"How can one {intent_base} under the rule “{constraint_text}” and verify that {lower_first(outcome_text)}?",
            f"What response would satisfy “{constraint_text}” while carrying out the goal to {intent_base} and establishing that {lower_first(outcome_text)}?",
            f"How should this be handled if the response must {intent_base}, must respect the boundary “{constraint_text}”, and must end with {lower_first(outcome_text)}?",
            f"Which bounded answer can {intent_base} and respect “{constraint_text}” while still producing the result that {lower_first(outcome_text)}?",
        )
    else:
        endings = (
            f"In this {domain.label.lower()} case, given that {lower_first(state_text)}, completion under “{constraint_text}” requires that {lower_first(outcome_text)}. If that result remains unsupported while trying to {intent_base} for {subject}, {lower_first(fallback_text)}.",
            f"Given that {lower_first(state_text)}, a {domain.label.lower()} response respecting “{constraint_text}” should establish that {lower_first(outcome_text)}. If it cannot do so while trying to {intent_base} for {subject}, {lower_first(fallback_text)}.",
            f"From a state where {lower_first(state_text)}, success in this {domain.label.lower()} case means honoring “{constraint_text}” until {lower_first(outcome_text)}. If that result fails while trying to {intent_base} for {subject}, {lower_first(fallback_text)}.",
            f"Two checks govern this {domain.label.lower()} case: “{constraint_text}”, and whether {lower_first(outcome_text)}. Given that {lower_first(state_text)}, failure while trying to {intent_base} for {subject} means that one should {lower_first(fallback_text)}.",
            f"The endpoint in this {domain.label.lower()} case is valid only when “{constraint_text}” is preserved and {lower_first(outcome_text)}. Given that {lower_first(state_text)}, failure while trying to {intent_base} for {subject} means one should {lower_first(fallback_text)}.",
            f"From a state where {lower_first(state_text)}, a {domain.label.lower()} answer must retain “{constraint_text}” and make it true that {lower_first(outcome_text)}. Without that support while trying to {intent_base} for {subject}, {lower_first(fallback_text)}.",
            f"Given that {lower_first(state_text)}, work in this {domain.label.lower()} case remains bounded by “{constraint_text}” until {lower_first(outcome_text)}. Before trying to {intent_base} further for {subject}, {lower_first(fallback_text)}.",
            f"In this {domain.label.lower()} case, given that {lower_first(state_text)}, success requires respecting “{constraint_text}” and establishing that {lower_first(outcome_text)}. If that remains unsupported while trying to {intent_base} for {subject}, {lower_first(fallback_text)}.",
            f"From a state where {lower_first(state_text)}, the acceptance rule for this {domain.label.lower()} case combines “{constraint_text}” with the result that {lower_first(outcome_text)}. Otherwise, before trying to {intent_base} further for {subject}, {lower_first(fallback_text)}.",
            f"Given that {lower_first(state_text)}, reject a {domain.label.lower()} answer that violates “{constraint_text}” or cannot show that {lower_first(outcome_text)}. In that case, when handling {subject}, {intent_negative} until the evidence changes; {lower_first(fallback_text)}.",
            f"From a state where {lower_first(state_text)}, an acceptable {domain.label.lower()} response must honor “{constraint_text}” and leave the case such that {lower_first(outcome_text)}. Otherwise, while {intent_progressive}, the {domain.label.lower()} case remains unresolved; {lower_first(fallback_text)}.",
            f"In this {domain.label.lower()} case, given that {lower_first(state_text)}, the finishing condition is that {lower_first(outcome_text)}, but never at the expense of “{constraint_text}”. If the two cannot be reconciled while trying to {intent_base} for {subject}, {lower_first(fallback_text)}.",
        )

    trigger = triggers[frame_index]
    ending = endings[frame_index]
    if family_id in {"conversation_empathy", "explanation_learning"}:
        ending = (
            f"For {subject}, given that {lower_first(state_text)}, "
            f"{lower_first(ending)}"
        )
    situation = " ".join((trigger, developments[frame_index], ending))
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
