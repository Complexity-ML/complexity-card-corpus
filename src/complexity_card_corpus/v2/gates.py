from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class V2Gate:
    gate_id: str
    category: str
    implemented: bool


V2_RELEASE_GATES = (
    V2Gate("volume.no_synthetic_cap", "volume", True),
    V2Gate("volume.full_family_capacity", "volume", True),
    V2Gate("volume.complete_family_registry", "volume", True),
    V2Gate("schema.required_model_fields", "schema", True),
    V2Gate("schema.one_assistant_target", "schema", True),
    V2Gate("schema.think_final_envelope_integrity", "schema", True),
    V2Gate("surface.direct_role_boundary", "surface", True),
    V2Gate("surface.nested_role_boundary", "surface", True),
    V2Gate("surface.lexical_role_overlap", "surface", True),
    V2Gate("surface.prompt_final_copy", "surface", True),
    V2Gate("surface.prompt_thinking_copy", "surface", True),
    V2Gate("repetition.final_internal", "repetition", True),
    V2Gate("repetition.final_exact_cross_row", "repetition", True),
    V2Gate("repetition.final_closing_cross_row", "repetition", True),
    V2Gate("repetition.final_semantic_near_duplicate", "repetition", True),
    V2Gate("repetition.prompt_semantic_near_duplicate", "repetition", True),
    V2Gate("repetition.thinking_internal", "repetition", True),
    V2Gate("repetition.thinking_exact_signature", "repetition", True),
    V2Gate("repetition.thinking_fivegram", "repetition", True),
    V2Gate("repetition.thinking_final_overlap", "repetition", True),
    V2Gate("correctness.deterministic_validators", "correctness", True),
    V2Gate("correctness.conflicting_prompt_answers", "correctness", True),
    V2Gate("correctness.arithmetic_recomputation", "correctness", True),
    V2Gate("correctness.instruction_constraints", "correctness", True),
    V2Gate("language.unrendered_placeholders", "language", True),
    V2Gate("language.punctuation_and_casing", "language", True),
    V2Gate("language.grammar_composition", "language", True),
    V2Gate("language.response_length_bands", "language", True),
    V2Gate("distribution.variable_card_entropy", "distribution", True),
    V2Gate("distribution.subcard_edge_coverage", "distribution", True),
    V2Gate("distribution.domain_balance", "distribution", True),
    V2Gate("behavior.direct_anchors", "behavior", True),
    V2Gate("behavior.safety_policy_anchors", "behavior", True),
    V2Gate("behavior.factual_anchors", "behavior", True),
    V2Gate("split.exact_leakage", "split", True),
    V2Gate("split.near_duplicate_leakage", "split", True),
    V2Gate("provenance.source_and_license", "provenance", True),
    V2Gate("tokenization.chat_template_roundtrip", "tokenization", True),
    V2Gate("tokenization.assistant_loss_mask", "tokenization", True),
    V2Gate("tokenization.think_final_tokens", "tokenization", True),
)


def v2_gate_progress() -> dict[str, object]:
    implemented = tuple(gate.gate_id for gate in V2_RELEASE_GATES if gate.implemented)
    missing = tuple(gate.gate_id for gate in V2_RELEASE_GATES if not gate.implemented)
    by_category = {
        category: {
            "implemented": sum(
                gate.implemented
                for gate in V2_RELEASE_GATES
                if gate.category == category
            ),
            "total": sum(gate.category == category for gate in V2_RELEASE_GATES),
        }
        for category in sorted({gate.category for gate in V2_RELEASE_GATES})
    }
    return {
        "implemented": implemented,
        "missing": missing,
        "implemented_count": len(implemented),
        "total_count": len(V2_RELEASE_GATES),
        "implemented_share": round(len(implemented) / len(V2_RELEASE_GATES), 6),
        "by_category": by_category,
        "complete": not missing,
    }


__all__ = ("V2Gate", "V2_RELEASE_GATES", "v2_gate_progress")
