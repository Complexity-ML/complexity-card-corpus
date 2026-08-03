from __future__ import annotations

from typing import Any

# Completion contracts describe the semantic fields that a correct answer must
# realize. They are metadata for validation and auditing, not headings that are
# injected into assistant prose. Closely related intents deliberately share a
# contract when their useful answer shape is the same.
_INTENT_CONTRACTS: dict[str, dict[str, tuple[str, ...]]] = {
    "practical_action": {
        "arrange": ("next_step", "owner", "timing", "confirmation"),
        "change": ("change_scope", "owner", "timing", "reversibility_check"),
        "cancel": ("cancellation_scope", "owner", "consequences", "preserved_state"),
        "verify": ("verification_target", "known_unknown", "owner", "acceptance_check"),
        "resolve": ("discrepancy", "resolution_step", "owner", "confirmation"),
        "document": ("decision_record", "owner", "timing", "supporting_check"),
    },
    "explanation_learning": {
        "mechanism": ("simple_model", "mechanism", "example", "understanding_check"),
        "contrast": ("concept_a", "concept_b", "decisive_difference", "contrast_check"),
        "worked_example": ("principle", "worked_steps", "result", "transfer_check"),
        "misconception": (
            "misconception",
            "correction",
            "example",
            "understanding_check",
        ),
        "application": ("principle", "new_situation", "application", "transfer_check"),
        "diagnose_gap": (
            "known_part",
            "learning_gap",
            "next_explanation",
            "diagnostic_check",
        ),
    },
    "troubleshooting": {
        "diagnose": (
            "preserved_state",
            "discriminating_checks",
            "diagnosis",
            "verification",
        ),
        "reproduce": (
            "preconditions",
            "minimal_steps",
            "observation",
            "control_comparison",
        ),
        "recover": ("preserved_state", "recovery_steps", "rollback", "recovery_check"),
        "verify_fix": ("proposed_fix", "direct_check", "regression_check", "rollback"),
        "prevent": ("root_cause", "prevention_step", "monitoring_check", "rollback"),
        "trace_transition": (
            "known_good_state",
            "state_transition",
            "failure_observation",
            "boundary_check",
        ),
    },
    "writing_transformation": {
        "draft": ("source_facts", "audience", "faithful_draft", "unresolved_points"),
        "revise": ("preserved_claims", "clarity_revision", "owner", "timing"),
        "summarize": ("essential_points", "decisions", "caveats", "open_points"),
        "adapt_tone": (
            "preserved_meaning",
            "audience",
            "adapted_tone",
            "commitment_check",
        ),
        "restructure": (
            "preserved_meaning",
            "action_first_structure",
            "owner",
            "timing",
        ),
        "adapt_register": (
            "preserved_meaning",
            "target_register",
            "revised_text",
            "claim_check",
        ),
    },
    "planning_comparison": {
        "define_options": (
            "hard_constraints",
            "viable_options",
            "excluded_options",
            "decision_gate",
        ),
        "compare": ("criteria", "option_comparison", "tradeoffs", "recommendation"),
        "sequence": (
            "dependencies",
            "ordered_steps",
            "checkpoints",
            "fallback_trigger",
        ),
        "allocate": ("available_resources", "allocation", "tradeoffs", "reserve"),
        "fallback": (
            "primary_path",
            "failure_signal",
            "fallback_path",
            "switch_trigger",
        ),
        "sequence_dependencies": (
            "dependency_graph",
            "critical_order",
            "optional_work",
            "fallback_trigger",
        ),
    },
    "conversation_empathy": {
        "acknowledge": (
            "acknowledgment",
            "state_reflection",
            "agency",
            "optional_question",
        ),
        "clarify_need": ("acknowledgment", "stated_need", "agency", "one_question"),
        "prepare_conversation": (
            "acknowledgment",
            "conversation_goal",
            "opening",
            "boundary",
        ),
        "small_step": (
            "acknowledgment",
            "manageable_step",
            "agency",
            "optional_check_in",
        ),
        "reflect": (
            "acknowledgment",
            "meaning_reflection",
            "mixed_feelings",
            "open_question",
        ),
        "reflect_need": (
            "acknowledgment",
            "expressed_need",
            "no_motive_assumption",
            "agency",
        ),
    },
    "safety_uncertainty": {
        "set_boundary": (
            "risk",
            "safety_boundary",
            "allowed_help",
            "escalation_condition",
        ),
        "clarify_scope": ("risk", "safe_scope", "excluded_scope", "next_safe_step"),
        "safe_alternative": (
            "unsafe_path",
            "boundary",
            "safe_alternative",
            "verification",
        ),
        "preserve_privacy": (
            "sensitive_data_boundary",
            "minimal_information",
            "user_control",
            "safe_channel",
        ),
        "escalate": (
            "risk_signal",
            "immediate_action",
            "qualified_support",
            "emergency_threshold",
        ),
        "triage_next_step": (
            "risk_level",
            "immediate_action",
            "bounded_next_step",
            "escalation_threshold",
        ),
    },
    "grounded_qa": {
        "answer": ("direct_answer", "decisive_evidence", "unknown"),
        "locate": ("evidence_location", "quoted_fact", "supported_answer", "unknown"),
        "compare": ("claim_a", "claim_b", "evidence_comparison", "bounded_conclusion"),
        "infer": ("source_facts", "bounded_inference", "inference_limit", "unknown"),
        "unknown": ("established_facts", "unknown_details", "required_evidence"),
        "reconcile": (
            "source_a",
            "source_b",
            "compatible_facts",
            "unresolved_conflict",
        ),
    },
    "summarization_synthesis": {
        "essentials": ("essential_points", "decision", "action", "open_point"),
        "synthesize": (
            "source_positions",
            "shared_points",
            "disagreement",
            "open_point",
        ),
        "decisions": ("decision", "owner", "action", "open_point"),
        "chronology": ("ordered_events", "latest_state", "action", "open_point"),
        "audience": ("audience_priority", "essential_points", "caveat", "open_point"),
        "trace_decisions": (
            "decision",
            "supporting_evidence",
            "action",
            "unsupported_link",
        ),
    },
    "reasoning_verification": {
        "calculate": ("given_values", "equation", "result", "independent_check"),
        "compare": ("quantities", "common_unit", "comparison", "check"),
        "constraint": (
            "constraint",
            "computed_value",
            "satisfaction_result",
            "boundary_check",
        ),
        "explain": ("given_values", "reasoning_steps", "result", "check"),
        "verify": (
            "proposed_result",
            "independent_method",
            "verification_result",
            "discrepancy",
        ),
        "test_counterexample": (
            "claim",
            "candidate_counterexample",
            "test_result",
            "revised_scope",
        ),
    },
    "critique_revision": {
        "identify": (
            "draft_evidence",
            "primary_weakness",
            "impact",
            "bounded_revision",
        ),
        "revise": ("preserved_intent", "weak_passage", "revision", "support_check"),
        "consistency": ("claim_set", "inconsistency", "affected_scope", "revision"),
        "evidence": ("claim", "current_evidence", "evidence_gap", "revision"),
        "prioritize": (
            "blockers",
            "high_impact_fix",
            "optional_improvements",
            "review_order",
        ),
        "prioritize_revision": ("impact", "evidence", "effort", "ordered_revisions"),
    },
    "brainstorming_creativity": {
        "generate": ("constraints", "distinct_options", "tradeoffs", "selection_rule"),
        "diversify": (
            "current_pattern",
            "variation_axes",
            "distinct_options",
            "selection_rule",
        ),
        "filter": ("candidate_options", "criteria", "excluded_options", "shortlist"),
        "combine": (
            "compatible_elements",
            "combined_options",
            "tradeoffs",
            "test_scope",
        ),
        "develop": ("selected_idea", "bounded_proposal", "assumptions", "first_test"),
        "synthesize_mechanisms": (
            "mechanisms",
            "compatibility",
            "synthesized_options",
            "first_test",
        ),
    },
    "context_clarification": {
        "question": (
            "confirmed_scope",
            "material_ambiguity",
            "one_question",
            "bounded_fallback",
        ),
        "restate": (
            "confirmed_request",
            "provisional_restatement",
            "one_question",
            "reversible_default",
        ),
        "reference": (
            "ambiguous_reference",
            "known_context",
            "one_question",
            "no_action_default",
        ),
        "assumptions": (
            "confirmed_facts",
            "assumptions",
            "one_question",
            "bounded_fallback",
        ),
        "bounded": (
            "confirmed_scope",
            "bounded_interpretation",
            "one_question",
            "reversibility",
        ),
        "confirm_scope": (
            "included_scope",
            "excluded_scope",
            "one_question",
            "no_commitment_default",
        ),
    },
}


def intent_contract_for(row: dict[str, Any]) -> tuple[str, ...]:
    """Return the authored semantic completion contract for a scenario."""

    family = str(row.get("family", ""))
    intent = str(row.get("intent", ""))
    try:
        return _INTENT_CONTRACTS[family][intent]
    except KeyError as error:
        raise ValueError(f"no completion contract for {family}:{intent}") from error


def intent_contract_catalog() -> dict[str, dict[str, tuple[str, ...]]]:
    """Return a defensive copy used by tests and audit tooling."""

    return {family: dict(contracts) for family, contracts in _INTENT_CONTRACTS.items()}
