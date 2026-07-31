from __future__ import annotations

import random
from collections import Counter
from typing import Protocol


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
        domain: DomainLike,
        intent: IntentLike,
        constraint: AtomLike,
        state: AtomLike,
        outcome: AtomLike,
    ) -> tuple[str, str, str]:
        """Compose prose after semantic compatibility has already been resolved."""
        frame_index, frame_id = self._choose_frame(domain.domain_id)
        return _render_frame(
            frame_index,
            frame_id,
            domain,
            intent,
            constraint,
            state,
            outcome,
        )


def _render_frame(
    frame_index: int,
    frame_id: str,
    domain: DomainLike,
    intent: IntentLike,
    constraint: AtomLike,
    state: AtomLike,
    outcome: AtomLike,
) -> tuple[str, str, str]:
    subject = domain.subject
    context_text = domain.context.rstrip(".")
    state_text = state.label.rstrip(".")
    constraint_text = constraint.label.rstrip(".")
    outcome_text = outcome.label.rstrip(".")
    frames = (
        (
            f"A routine step involving {subject} reaches a decision point when "
            f"{lower_first(state_text)}.",
            f"{context_text}. The person needs to {intent.label}, while respecting "
            f"this boundary: {constraint_text}. A satisfactory resolution is one "
            f"where {lower_first(outcome_text)}.",
        ),
        (
            f"The situation around {subject} changes because {lower_first(state_text)}.",
            f"The relevant setting is clear: {lower_first(context_text)}. The immediate "
            f"objective is to {intent.label}. The governing condition is: "
            f"{constraint_text}. Success requires that {lower_first(outcome_text)}.",
        ),
        (
            f"A new obstacle appears while handling {subject}: {state_text}.",
            f"In this case, {lower_first(context_text)}. The next response must "
            f"{intent.label}. It must preserve this boundary: {constraint_text}. "
            f"The target result is that {lower_first(outcome_text)}.",
        ),
        (
            f"Work on {subject} can no longer continue unchanged after this update: "
            f"{state_text}.",
            f"{context_text}. The person therefore needs to {intent.label}. The plan "
            f"must account for this condition: {constraint_text}. It should end in a "
            f"state where {lower_first(outcome_text)}.",
        ),
        (
            f"The decisive signal in this {domain.label.lower()} case is that "
            f"{lower_first(state_text)}.",
            f"The case concerns {subject}. {context_text}. The useful task is to "
            f"{intent.label}, subject to this limit: {constraint_text}. Completion "
            f"means that {lower_first(outcome_text)}.",
        ),
        (
            f"An otherwise ordinary request about {subject} becomes non-routine when "
            f"{lower_first(state_text)}.",
            f"{context_text}. The person needs to {intent.label}. "
            f"The response must preserve this rule: {constraint_text}. "
            f"The intended endpoint is one where {lower_first(outcome_text)}.",
        ),
        (
            f"Before the next step for {subject}, one fact changes the shape of the "
            f"request: {state_text}.",
            f"{context_text}. The person is trying to {intent.label}. Any proposal "
            f"must honor this condition: {constraint_text}. It succeeds only if "
            f"{lower_first(outcome_text)}.",
        ),
        (
            f"The immediate trigger for this {domain.label.lower()} scenario is simple: "
            f"{state_text}.",
            f"The subject is {subject}. {context_text}. The requested help "
            f"is to {intent.label}. The hard boundary is: "
            f"{constraint_text}. The desired result is that "
            f"{lower_first(outcome_text)}.",
        ),
        (
            f"A checkpoint is reached for {subject} once {lower_first(state_text)}.",
            f"{context_text}. The task is to {intent.label}. "
            f"The answer must observe this condition: {constraint_text}. It should "
            f"lead to a state where {lower_first(outcome_text)}.",
        ),
        (
            f"The next decision about {subject} is prompted by one concrete condition: "
            f"{state_text}.",
            f"{context_text}. The person now wants to {intent.label}. The response is "
            f"bounded by this requirement: {constraint_text}. The finish line is that "
            f"{lower_first(outcome_text)}.",
        ),
        (
            f"A request involving {subject} becomes actionable at the moment when "
            f"{lower_first(state_text)}.",
            f"The surrounding context is this: {lower_first(context_text)}. The person "
            f"needs to {intent.label}. The solution must respect this condition: "
            f"{constraint_text}. It must establish that {lower_first(outcome_text)}.",
        ),
        (
            f"This {domain.label.lower()} case begins from a specific turning point: "
            f"{state_text}.",
            f"It concerns {subject}. {context_text}. The next useful move is to "
            f"{intent.label}, under this constraint: {constraint_text}. "
            f"The case is resolved when {lower_first(outcome_text)}.",
        ),
    )
    trigger, situation = frames[frame_index]
    return trigger, f"{trigger} {situation}", frame_id


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
