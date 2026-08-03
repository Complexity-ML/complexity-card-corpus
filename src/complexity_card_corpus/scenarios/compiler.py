from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from ..scenario_integrity import (
    creation_hash as _creation_hash,
    deterministic_order as _permuted,
    stable_digest as _digest,
    verification_hash as _verification_hash,
)
from ..scenario_language import DynamicNarrativeComposer, compose_title as _title
from .audit import audit_scenarios
from .schema import (
    SCENARIO_PROVENANCE,
    DomainSpec,
    IntentSpec,
    ScenarioFamilySpec,
    ScenarioForgeRegistry,
    SemanticAtom,
)


def _domain_quotas(
    family: ScenarioFamilySpec,
    seed: int,
    *,
    target: int | None = None,
    capacities: dict[str, int] | None = None,
) -> dict[str, int]:
    resolved_target = family.weight if target is None else target
    order = sorted(
        family.domains,
        key=lambda domain: _digest(f"{seed}:{family.family_id}:{domain.domain_id}"),
    )
    capacities = capacities or {
        domain.domain_id: len(_semantic_combinations(family, domain, seed=seed))
        for domain in family.domains
    }
    if resolved_target > sum(capacities.values()):
        raise ValueError(
            f"family {family.family_id} requests {resolved_target} scenarios but "
            f"its domain capacity is {sum(capacities.values())}"
        )
    quotas = {domain.domain_id: 0 for domain in family.domains}
    remaining = resolved_target
    active = {domain.domain_id for domain in family.domains}
    while remaining:
        share = max(1, remaining // len(active))
        progress = 0
        for domain in order:
            domain_id = domain.domain_id
            if domain_id not in active:
                continue
            addition = min(share, capacities[domain_id] - quotas[domain_id])
            quotas[domain_id] += addition
            progress += addition
            if progress == remaining:
                break
        remaining -= progress
        active = {
            domain_id
            for domain_id in active
            if quotas[domain_id] < capacities[domain_id]
        }
        if progress == 0:
            raise ValueError(f"domain capacity exhausted for {family.family_id}")
    return quotas


def resolve_family_targets(
    registry: ScenarioForgeRegistry,
    target_scenarios: int,
) -> dict[str, int]:
    """Scale family targets by authored weights and compatible capacity.

    Tank ``weight`` values control proportional allocation, not upper bounds.
    A caller may request any total supported by the realized semantic capacity.
    """

    baseline = {family.family_id: family.weight for family in registry.families}
    if target_scenarios < 1:
        raise ValueError("target_scenarios must be positive")
    capacities = {
        family.family_id: family.semantic_signature_capacity()
        for family in registry.families
    }
    if target_scenarios > sum(capacities.values()):
        raise ValueError(
            f"requested {target_scenarios} scenarios but compatible capacity is "
            f"{sum(capacities.values())}"
        )

    allocated = {family_id: 0 for family_id in baseline}
    remaining = target_scenarios
    active = set(baseline)
    while remaining:
        if not active:
            raise ValueError("semantic capacity was exhausted during allocation")
        weight_total = sum(baseline[family_id] for family_id in active)
        ideals = {
            family_id: remaining * baseline[family_id] / weight_total
            for family_id in active
        }
        progress = 0
        for family_id in sorted(active):
            available = capacities[family_id] - allocated[family_id]
            addition = min(available, int(ideals[family_id]))
            allocated[family_id] += addition
            progress += addition
        remaining -= progress
        active = {
            family_id
            for family_id in active
            if allocated[family_id] < capacities[family_id]
        }
        if not remaining:
            break
        if progress == 0 or remaining < len(active):
            order = sorted(
                active,
                key=lambda family_id: (
                    -(ideals.get(family_id, 0.0) % 1.0),
                    _digest(f"{registry.seed}:family-target:{family_id}"),
                ),
            )
            for family_id in order[:remaining]:
                allocated[family_id] += 1
            remaining = 0
    return allocated


def _assign_splits(
    rows: list[dict[str, Any]], validation_percent: int, seed: int
) -> None:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row["split"] = "train"
        families[row["family"]].append(row)

    target = round(len(rows) * validation_percent / 100)
    quotas: dict[str, int] = {}
    remainders: list[tuple[int, bytes, str]] = []
    for family_id, values in families.items():
        numerator = len(values) * validation_percent
        quotas[family_id] = numerator // 100
        remainders.append(
            (
                numerator % 100,
                _digest(f"{seed}:split-quota:{family_id}"),
                family_id,
            )
        )
    remaining = target - sum(quotas.values())
    for _, _, family_id in sorted(remainders, key=lambda value: (-value[0], value[1]))[
        :remaining
    ]:
        quotas[family_id] += 1

    for family_id, values in families.items():
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in values:
            groups[(row["domain"], row["intent"])].append(row)
        ordered = sorted(
            groups.items(),
            key=lambda item: _digest(
                f"{seed}:validation-group:{family_id}:{item[0][0]}:{item[0][1]}"
            ),
        )
        family_target = quotas[family_id]
        subsets: dict[int, tuple[tuple[str, str], ...]] = {0: ()}
        for key, group_rows in ordered:
            for subtotal, selected in tuple(sorted(subsets.items(), reverse=True)):
                candidate = subtotal + len(group_rows)
                if candidate not in subsets:
                    subsets[candidate] = (*selected, key)
        # A complete (family, domain, intent) group is the leakage boundary.
        # At larger scales those groups can be wider than the exact percentage
        # quota, so choose the closest whole-group subset instead of splitting
        # a semantic unit merely to hit an exact row count.
        closest_total = min(
            (subtotal for subtotal in subsets if subtotal > 0),
            key=lambda subtotal: (
                abs(subtotal - family_target),
                subtotal > family_target,
                _digest(f"{seed}:validation-total:{family_id}:{subtotal}"),
            ),
        )
        selected_groups = set(subsets[closest_total])
        for key in selected_groups:
            for row in groups[key]:
                row["split"] = "validation"


def _semantic_combinations(
    family: ScenarioFamilySpec,
    domain: DomainSpec,
    *,
    seed: int,
) -> list[tuple[IntentSpec, SemanticAtom, SemanticAtom, SemanticAtom]]:
    prefix = f"{seed}:{family.family_id}:{domain.domain_id}"
    constraint_by_id = {value.atom_id: value for value in family.constraints}
    outcome_by_id = {value.atom_id: value for value in family.outcomes}
    intent_by_id = {value.atom_id: value for value in family.intents}
    constraints = _permuted(
        [
            constraint_by_id[value]
            for value in family.compatibility.domain_constraints[domain.domain_id]
        ],
        f"{prefix}:constraint",
    )
    states = _permuted(family.states, f"{prefix}:state")
    intents = _permuted(
        [
            intent_by_id[value]
            for value in family.compatibility.domain_intents[domain.domain_id]
        ],
        f"{prefix}:intent",
    )
    combinations = [
        (intent, constraint, state, outcome_by_id[outcome_id])
        for intent in intents
        for constraint in constraints
        for state in states
        for outcome_id in sorted(
            set(family.compatibility.intent_outcomes[intent.atom_id])
            & set(family.compatibility.state_outcomes[state.atom_id])
        )
    ]
    return sorted(
        combinations,
        key=lambda items: _digest(
            prefix + ":" + ":".join(item.atom_id for item in items)
        ),
    )


def _compatible_fallbacks(
    family: ScenarioFamilySpec,
    domain: DomainSpec,
    state: SemanticAtom,
) -> list[SemanticAtom]:
    fallback_by_id = {value.atom_id: value for value in family.fallbacks}
    allowed_ids = sorted(
        set(family.compatibility.state_fallbacks[state.atom_id])
        & set(family.compatibility.risk_fallbacks[domain.risk_level])
    )
    return [fallback_by_id[value] for value in allowed_ids]


def _select_fallback(
    family: ScenarioFamilySpec,
    domain: DomainSpec,
    state: SemanticAtom,
) -> SemanticAtom:
    candidates = {
        value.atom_id: value for value in _compatible_fallbacks(family, domain, state)
    }
    for fallback_id in family.compatibility.state_fallbacks[state.atom_id]:
        if fallback_id in candidates:
            return candidates[fallback_id]
    raise ValueError(
        f"no prioritized fallback for {family.family_id}/{domain.domain_id}/{state.atom_id}"
    )


def _payload(
    family: ScenarioFamilySpec,
    domain: DomainSpec,
    intent: IntentSpec,
    constraint: SemanticAtom,
    state: SemanticAtom,
    outcome: SemanticAtom,
    fallback: SemanticAtom,
) -> dict[str, str]:
    common = {
        "domain_context": domain.context,
        "subject": domain.subject,
        "constraint": constraint.label,
        "current_state": state.label,
        "success_condition": outcome.label,
        "fallback": fallback.label,
    }
    family_specific = {
        "practical_action": {
            "requested_action": intent.label,
            "missing_fact": state.label,
        },
        "explanation_learning": {
            "learning_goal": intent.label,
            "misconception_risk": state.label,
            "check_for_understanding": outcome.label,
        },
        "troubleshooting": {
            "diagnostic_goal": intent.label,
            "observed_symptom": state.label,
            "stop_condition": fallback.label,
        },
        "writing_transformation": {
            "transformation": intent.label,
            "source_state": state.label,
            "audience_result": outcome.label,
        },
        "planning_comparison": {
            "planning_goal": intent.label,
            "options_state": state.label,
            "decision_rule": outcome.label,
        },
        "conversation_empathy": {
            "conversational_goal": intent.label,
            "emotion_state": state.label,
            "support_boundary": constraint.label,
        },
        "safety_uncertainty": {
            "safe_goal": intent.label,
            "risk_state": state.label,
            "safety_boundary": constraint.label,
            "safe_alternative": fallback.label,
        },
        "grounded_qa": {
            "question_goal": intent.label,
            "evidence_state": state.label,
            "source_boundary": constraint.label,
        },
        "summarization_synthesis": {
            "summary_goal": intent.label,
            "source_state": state.label,
            "retention_check": outcome.label,
        },
        "extraction_classification": {
            "extraction_goal": intent.label,
            "record_state": state.label,
            "schema_check": outcome.label,
        },
        "reasoning_verification": {
            "reasoning_goal": intent.label,
            "problem_state": state.label,
            "verification_check": outcome.label,
        },
        "critique_revision": {
            "critique_goal": intent.label,
            "draft_state": state.label,
            "revision_check": outcome.label,
        },
        "brainstorming_creativity": {
            "ideation_goal": intent.label,
            "idea_state": state.label,
            "selection_check": outcome.label,
        },
        "context_clarification": {
            "clarification_goal": intent.label,
            "ambiguity_state": state.label,
            "scope_check": outcome.label,
        },
    }
    if family.family_id not in family_specific:
        raise ValueError(f"no payload contract for family {family.family_id}")
    return {**common, **family_specific[family.family_id]}


def compile_scenarios(
    registry: ScenarioForgeRegistry,
    *,
    target_scenarios: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    language = DynamicNarrativeComposer(seed=registry.seed)
    family_targets = resolve_family_targets(registry, target_scenarios)
    domain_targets: dict[str, dict[str, int]] = {}
    for family in registry.families:
        domain_combinations = {
            domain.domain_id: _semantic_combinations(
                family,
                domain,
                seed=registry.seed,
            )
            for domain in family.domains
        }
        quotas = _domain_quotas(
            family,
            registry.seed,
            target=family_targets[family.family_id],
            capacities={
                domain_id: len(combinations)
                for domain_id, combinations in domain_combinations.items()
            },
        )
        domain_targets[family.family_id] = {
            domain_id: count for domain_id, count in quotas.items() if count
        }
        for domain in family.domains:
            combinations = domain_combinations[domain.domain_id]
            for intent, constraint, state, outcome in combinations[
                : quotas[domain.domain_id]
            ]:
                signature = "|".join(
                    (
                        family.family_id,
                        domain.domain_id,
                        intent.atom_id,
                        constraint.atom_id,
                        state.atom_id,
                        outcome.atom_id,
                    )
                )
                creation_hash = _creation_hash(signature)
                scenario_id = f"scenario:{creation_hash[:24]}"
                fallback = _select_fallback(family, domain, state)
                trigger, situation, narrative_frame = language.compose(
                    family.family_id,
                    domain,
                    intent,
                    constraint,
                    state,
                    outcome,
                    fallback,
                )
                base_goal = intent.goal_template.format(
                    subject=domain.subject,
                    context=domain.context,
                )
                goal = (
                    f"{base_goal} Account for this state: {state.label.rstrip('.')}. "
                    f"Respect this boundary: {constraint.label.rstrip('.')}. "
                    f"Establish this result: {outcome.label.rstrip('.')}."
                )
                payload = _payload(
                    family,
                    domain,
                    intent,
                    constraint,
                    state,
                    outcome,
                    fallback,
                )
                row = {
                    "scenario_id": scenario_id,
                    "creation_hash": creation_hash,
                    "verification_hash": "",
                    "family": family.family_id,
                    "domain": domain.domain_id,
                    "intent": intent.atom_id,
                    "title": _title(domain, intent, constraint, state, outcome),
                    "trigger": trigger,
                    "situation": situation,
                    "narrative_frame": narrative_frame,
                    "goal": goal,
                    "constraint": constraint.label,
                    "state": state.label,
                    "desired_outcome": outcome.label,
                    "fallback": fallback.label,
                    "risk_level": domain.risk_level,
                    "split": "",
                    "semantic_signature": signature,
                    "semantic_payload": json.dumps(payload, sort_keys=True),
                    "response_contract": list(family.response_contract),
                    "source_structure_keys": [
                        f"family:{family.family_id}",
                        f"domain:{domain.domain_id}",
                        f"intent:{intent.atom_id}",
                        f"constraint:{constraint.atom_id}",
                        f"state:{state.atom_id}",
                        f"outcome:{outcome.atom_id}",
                        f"fallback:{fallback.atom_id}",
                        f"risk:{domain.risk_level}",
                    ],
                    "source_structure_links": [
                        f"family:{family.family_id}->domain:{domain.domain_id}",
                        f"family:{family.family_id}->intent:{intent.atom_id}",
                        f"domain:{domain.domain_id}->intent:{intent.atom_id}",
                        f"domain:{domain.domain_id}->constraint:{constraint.atom_id}",
                        f"domain:{domain.domain_id}->state:{state.atom_id}",
                        f"intent:{intent.atom_id}->outcome:{outcome.atom_id}",
                        f"constraint:{constraint.atom_id}->outcome:{outcome.atom_id}",
                        f"constraint:{constraint.atom_id}->fallback:{fallback.atom_id}",
                        f"state:{state.atom_id}->outcome:{outcome.atom_id}",
                        f"state:{state.atom_id}->fallback:{fallback.atom_id}",
                        f"risk:{domain.risk_level}->state:{state.atom_id}",
                        f"risk:{domain.risk_level}->fallback:{fallback.atom_id}",
                    ],
                    "model_generated_dialogue": False,
                    "provenance": SCENARIO_PROVENANCE,
                    "license": registry.metadata.license,
                    "version": registry.metadata.version,
                }
                rows.append(row)
    rows.sort(key=lambda row: row["scenario_id"])
    _assign_splits(rows, registry.validation_percent, registry.seed)
    for row in rows:
        row["verification_hash"] = _verification_hash(row)
    audit_scenarios(
        rows,
        registry,
        expected_family_counts=family_targets,
        expected_domain_counts=domain_targets,
    )
    return rows
