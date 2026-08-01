from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingCards:
    """Invisible conditioning choices used to naturalize one SFT example."""

    surface: str
    dialogue_state: str
    output: str
    evidence: str
    reasoning: str
    style: str
    context_density: str
    noise: str
    uncertainty: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _stable_index(value: str, size: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % size


def _pick(key: str, values: tuple[str, ...]) -> str:
    return values[_stable_index(key, len(values))]


_OUTPUT_BY_TASK = {
    "brainstorming_creativity": "numbered_options",
    "context_clarification": "question_and_default",
    "critique_revision": "critique_and_revision",
    "extraction_classification": "json_or_schema",
    "planning_comparison": "decision_plan",
    "practical_action": "action_plan",
    "reasoning_verification": "equation_and_check",
    "safety_uncertainty": "boundary_and_escalation",
    "summarization_synthesis": "concise_synthesis",
    "troubleshooting": "diagnostic_steps",
    "writing_transformation": "revised_prose",
}

_REASONING_BY_TASK = {
    "brainstorming_creativity": "diverge_then_select",
    "context_clarification": "resolve_ambiguity",
    "critique_revision": "diagnose_then_revise",
    "explanation_learning": "explain_then_check",
    "extraction_classification": "extract_then_validate",
    "grounded_qa": "locate_then_answer",
    "planning_comparison": "compare_then_plan",
    "practical_action": "sequence_and_verify",
    "reasoning_verification": "calculate_then_verify",
    "safety_uncertainty": "bound_then_escalate",
    "summarization_synthesis": "select_then_compress",
    "troubleshooting": "isolate_then_test",
    "writing_transformation": "transform_then_preserve",
}

_STYLE_BY_TASK = {
    "brainstorming_creativity": ("imaginative", "varied", "practical_creative"),
    "conversation_empathy": ("warm", "calm", "supportive"),
    "explanation_learning": ("plain", "pedagogical", "concise"),
    "safety_uncertainty": ("calm", "direct", "cautious"),
    "writing_transformation": ("natural", "audience_aware", "concise"),
}


def _source_text(metadata: dict[str, Any]) -> str:
    return " ".join(
        str(metadata.get(key, ""))
        for key in (
            "state",
            "source_state",
            "constraint",
            "source_constraint",
            "desired_outcome",
            "fallback",
        )
    ).lower()


def _evidence_card(metadata: dict[str, Any]) -> str:
    source = _source_text(metadata)
    realized_state = str(
        metadata.get("state") or metadata.get("source_state", "")
    ).lower()
    if any(
        phrase in realized_state
        for phrase in (
            "statements in the context appear to conflict",
            "records contain conflicting details",
            "sources disagree",
            "records disagree",
        )
    ):
        return "conflicting"
    if any(
        word in source
        for word in ("missing", "partial", "unknown", "unclear", "ambiguous")
    ):
        return "partial"
    if any(word in source for word in ("scattered", "distributed", "mixed")):
        return "distributed"
    return "sufficient"


def _uncertainty_card(task: str, evidence: str, metadata: dict[str, Any]) -> str:
    if task == "safety_uncertainty":
        return "safety_boundary"
    if evidence == "conflicting":
        return "preserve_conflict"
    if evidence == "partial":
        return "state_limits"
    if str(metadata.get("risk_level", "low")) in {"medium", "high"}:
        return "verify_before_action"
    return "answerable"


def deal_training_cards(
    *,
    task: str,
    mode: str,
    example_id: str,
    metadata: dict[str, Any] | None = None,
) -> TrainingCards:
    """Deal compatible, deterministic cards without exposing their labels."""

    metadata = metadata or {}
    evidence = _evidence_card(metadata)
    risk = str(metadata.get("risk_level", "low"))

    if mode == "chat":
        dialogue_choices = (
            "follow_up",
            "constraint_update",
            "clarification_resolved",
            "continued_request",
        )
        surface_choices = (
            "conversational",
            "follow_up",
            "context_first",
            "plain",
        )
    else:
        dialogue_choices = ("new_request", "bounded_request", "direct_request")
        surface_choices = (
            "direct",
            "polite",
            "compact",
            "context_first",
            "plain",
        )

    if evidence == "conflicting":
        dialogue_choices = (*dialogue_choices, "correction")
    if risk in {"medium", "high"}:
        density_choices = ("full", "full", "focused")
    else:
        density_choices = ("full", "focused", "minimal", "full")

    return TrainingCards(
        surface=_pick(f"surface:{example_id}", surface_choices),
        dialogue_state=_pick(f"dialogue:{example_id}", dialogue_choices),
        output=_OUTPUT_BY_TASK.get(task, "direct_prose"),
        evidence=evidence,
        reasoning=_REASONING_BY_TASK.get(task, "direct_response"),
        style=_pick(
            f"style:{example_id}",
            _STYLE_BY_TASK.get(task, ("plain", "concise", "natural")),
        ),
        context_density=_pick(f"density:{example_id}", density_choices),
        noise=_pick(
            f"noise:{example_id}",
            ("none", "none", "none", "none", "secondary_detail"),
        ),
        uncertainty=_uncertainty_card(task, evidence, metadata),
    )
