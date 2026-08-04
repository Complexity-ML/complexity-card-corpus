from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any


# Families whose model-facing renderer materially applies response order and
# layout. Other families retain card metadata, but their output contract is
# atomic (for example JSON extraction or faithful prose rewriting), so hidden
# one-card-away signatures must not be mistaken for visible diversity.
RESPONSE_STRUCTURE_SIBLING_TASKS = frozenset(
    {
        "context_clarification",
        "conversation_empathy",
        "critique_revision",
        "explanation_learning",
        "grounded_qa",
        "planning_comparison",
        "practical_action",
        "reasoning_verification",
        "safety_uncertainty",
        "summarization_synthesis",
        "troubleshooting",
    }
)


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
    response_order: str = "source"
    response_bridge: str = "plain"
    response_layout: str = "paragraph"
    response_opening: str = "bare"
    natural_opening: str = "direct"
    natural_link: str = "clarify"
    natural_update: str = "goal"
    natural_depth: str = "direct"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def response_structure_signature(self) -> str:
        """Return the invisible response-card hand used by the SFT renderer."""

        return "|".join(
            (
                self.response_order,
                self.response_bridge,
                self.response_layout,
                self.response_opening,
            )
        )

    @property
    def response_structure_sibling_signatures(self) -> dict[str, str]:
        """Return the four one-card-away neighbourhoods of this response hand.

        Exact-hand balancing alone can hide a repetitive family: many hands may
        differ only by their opening, bridge, layout, or clause order. Each
        signature removes one axis and groups all hands that are structural
        siblings along that axis.
        """

        axes = {
            "order": self.response_order,
            "bridge": self.response_bridge,
            "layout": self.response_layout,
            "opening": self.response_opening,
        }
        return {
            f"without_{omitted}": "|".join(
                f"{name}={value}"
                for name, value in axes.items()
                if name != omitted
            )
            for omitted in axes
        }


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


_RESPONSE_ORDERS_BY_TASK = {
    "context_clarification": (
        "restatement>question>default",
        "restatement>default>question",
    ),
    "conversation_empathy": (
        "acknowledgment>state_reflection>agency>question",
        "acknowledgment>agency>state_reflection>question",
        "state_reflection>acknowledgment>agency>question",
    ),
    "grounded_qa": (
        "documented>boundary>verification",
        "documented>verification>boundary",
        "boundary>documented>verification",
        "verification>documented>boundary",
        "boundary>verification>documented",
        "verification>boundary>documented",
    ),
    "explanation_learning": (
        "idea>example>check",
        "idea>check>example",
        "example>idea>check",
        "check>idea>example",
    ),
    "reasoning_verification": (
        "equation>total>check",
        "total>equation>check",
        "equation>check>total",
        "check>equation>total",
    ),
    "summarization_synthesis": (
        "decision>action>open_point",
        "action>decision>open_point",
        "open_point>decision>action",
        "decision>open_point>action",
    ),
    "critique_revision": (
        "revision>weakness",
        "weakness>revision",
        "revision",
    ),
    "safety_uncertainty": (
        "action>boundary>escalation",
        "action>escalation>boundary",
    ),
    "practical_action": (
        "step>owner>timing>guard",
        "timing>step>owner>guard",
        "guard>step>owner>timing",
        "step>timing>guard>owner",
    ),
    "planning_comparison": (
        "criteria>choice>sequence>fallback",
        "choice>criteria>sequence>fallback",
        "sequence>criteria>choice>fallback",
        "criteria>sequence>choice>fallback",
    ),
    "troubleshooting": ("steps",),
}

_RESPONSE_LAYOUTS_BY_TASK = {
    "brainstorming_creativity": ("paragraph", "bullets", "numbered"),
    "context_clarification": (
        "paragraph",
        "line_breaks",
        "spaced_lines",
        "opening_break",
    ),
    "conversation_empathy": ("paragraph", "line_breaks"),
    "critique_revision": ("paragraph", "line_breaks"),
    "explanation_learning": ("paragraph", "paragraph", "line_breaks"),
    "grounded_qa": ("paragraph", "line_breaks", "bullets", "numbered"),
    "planning_comparison": ("paragraph", "paragraph", "line_breaks"),
    "practical_action": ("paragraph", "paragraph", "line_breaks"),
    "reasoning_verification": ("paragraph", "paragraph", "line_breaks"),
    "safety_uncertainty": ("paragraph", "line_breaks", "bullets"),
    "summarization_synthesis": ("paragraph", "paragraph", "line_breaks"),
    "troubleshooting": (
        "numbered",
        "bullets",
        "paragraph",
        "line_breaks",
        "spaced_lines",
    ),
}

_RESPONSE_BRIDGES = (
    "plain",
    "compact",
    "guided",
    "analytic",
    "conversational",
    "stepwise",
)

_RESPONSE_OPENINGS = ("bare", "direct", "result_first", "contextual")


# Invisible natural-dialogue subdecks.  The values are metadata labels used by
# the renderer; none of these storage names are emitted into model text.
_NATURAL_DIALOGUE_BY_TASK = {
    "brainstorming_creativity": {
        "opening": ("brief_first", "possibility_first", "constraint_first"),
        "link": ("differentiate", "test_options", "confirm_brief"),
        "update": ("selection_criterion", "creative_limit", "preference"),
    },
    "context_clarification": {
        "opening": ("known_first", "ambiguity_first", "decision_first"),
        "link": ("locate_ambiguity", "ask_decisive_detail", "bound_default"),
        "update": ("missing_detail", "format_choice", "scope_choice"),
    },
    "conversation_empathy": {
        "opening": ("experience_first", "feeling_first", "need_first"),
        "link": ("acknowledge", "invite_detail", "offer_choice"),
        "update": ("felt_need", "preferred_support", "gentle_boundary"),
    },
    "critique_revision": {
        "opening": ("draft_first", "weakness_first", "audience_first"),
        "link": ("identify_priority", "confirm_intent", "test_revision"),
        "update": ("revision_priority", "meaning_to_keep", "audience_need"),
    },
    "explanation_learning": {
        "opening": ("concept_first", "example_first", "learner_gap_first"),
        "link": ("locate_gap", "connect_example", "check_level"),
        "update": ("learning_goal", "known_material", "transfer_check"),
    },
    "extraction_classification": {
        "opening": ("record_first", "schema_first", "field_first"),
        "link": ("confirm_schema", "preserve_missing", "resolve_conflict"),
        "update": ("required_fields", "label_set", "normalization_rule"),
    },
    "grounded_qa": {
        "opening": ("question_first", "evidence_first", "scope_first"),
        "link": ("locate_support", "state_boundary", "resolve_conflict"),
        "update": ("exact_question", "source_limit", "answer_format"),
    },
    "planning_comparison": {
        "opening": ("choice_first", "criteria_first", "options_first"),
        "link": ("identify_criterion", "compare_tradeoff", "confirm_constraint"),
        "update": ("priority", "hard_constraint", "fallback_preference"),
    },
    "practical_action": {
        "opening": ("outcome_first", "situation_first", "next_step_first"),
        "link": ("confirm_owner", "sequence_action", "check_guardrail"),
        "update": ("desired_outcome", "available_resource", "fixed_boundary"),
    },
    "reasoning_verification": {
        "opening": ("inputs_first", "claim_first", "calculation_first"),
        "link": ("check_premise", "request_method", "separate_verification"),
        "update": ("target_value", "allowed_method", "precision_rule"),
    },
    "safety_uncertainty": {
        "opening": ("risk_first", "uncertainty_first", "safe_goal_first"),
        "link": ("check_immediacy", "set_boundary", "choose_escalation"),
        "update": ("current_status", "safe_limit", "available_support"),
    },
    "summarization_synthesis": {
        "opening": ("material_first", "decision_first", "audience_first"),
        "link": ("confirm_scope", "rank_information", "preserve_open_point"),
        "update": ("target_audience", "length_limit", "decision_focus"),
    },
    "troubleshooting": {
        "opening": ("symptom_first", "change_first", "system_first"),
        "link": ("isolate_change", "request_observation", "choose_safe_test"),
        "update": ("last_change", "observed_result", "test_boundary"),
    },
    "writing_transformation": {
        "opening": ("source_first", "audience_first", "purpose_first"),
        "link": ("confirm_meaning", "check_tone", "preserve_fact"),
        "update": ("target_audience", "desired_tone", "meaning_to_keep"),
    },
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
            "objection",
            "correction",
            "validation",
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

    if evidence == "conflicting" and mode != "chat":
        dialogue_choices = (*dialogue_choices, "correction")
    if risk in {"medium", "high"}:
        density_choices = ("full", "full", "focused")
    else:
        density_choices = ("full", "focused", "minimal", "full")

    natural_deck = _NATURAL_DIALOGUE_BY_TASK.get(
        task,
        {
            "opening": ("direct", "context_first"),
            "link": ("clarify", "confirm"),
            "update": ("goal", "constraint"),
        },
    )
    natural_depth_choices = (
        ("direct", "linked", "direct", "linked", "direct")
        if mode == "chat"
        else ("direct",)
    )

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
        response_order=_pick(
            f"response-order:{example_id}",
            _RESPONSE_ORDERS_BY_TASK.get(task, ("source",)),
        ),
        response_bridge=_pick(
            f"response-bridge:{example_id}",
            _RESPONSE_BRIDGES,
        ),
        response_layout=_pick(
            f"response-layout:{example_id}",
            _RESPONSE_LAYOUTS_BY_TASK.get(task, ("paragraph",)),
        ),
        response_opening=_pick(
            f"response-opening:{example_id}",
            _RESPONSE_OPENINGS,
        ),
        natural_opening=_pick(
            f"natural-opening:{task}:{example_id}",
            natural_deck["opening"],
        ),
        natural_link=_pick(
            f"natural-link:{task}:{example_id}",
            natural_deck["link"],
        ),
        natural_update=_pick(
            f"natural-update:{task}:{example_id}",
            natural_deck["update"],
        ),
        natural_depth=_pick(
            f"natural-depth:{task}:{example_id}",
            natural_depth_choices,
        ),
    )


def natural_dialogue_deck() -> dict[str, dict[str, tuple[str, ...]]]:
    """Expose the immutable family deck for audits and documentation."""

    return _NATURAL_DIALOGUE_BY_TASK


def projected_difficulty(
    cards: TrainingCards,
    messages: list[dict[str, str]],
) -> str:
    """Classify the realized request rather than its generator variant.

    Difficulty is Viewer metadata, not a balancing target.  The score uses
    only observable conversation complexity and the conditioning cards that
    produced it; it never depends on a hard-coded variant number.
    """

    user_messages = [
        message["content"]
        for message in messages
        if message.get("role") == "user"
    ]
    user_words = sum(len(content.split()) for content in user_messages)
    score = 0
    if len(user_messages) > 1:
        score += 1
    if user_words >= 60:
        score += 1
    if user_words >= 140:
        score += 1
    if cards.context_density == "full":
        score += 1
    if cards.noise != "none":
        score += 1
    if cards.evidence in {"partial", "distributed"}:
        score += 1
    elif cards.evidence == "conflicting":
        score += 2
    if cards.uncertainty != "answerable":
        score += 1
    if cards.natural_depth == "linked":
        score += 1

    if score >= 5:
        return "hard"
    if score >= 2:
        return "medium"
    return "easy"
