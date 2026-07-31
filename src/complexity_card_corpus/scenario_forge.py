from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .build import file_sha256
from .english_morphology import audit_verb_phrases
from .scenario_integrity import (
    creation_hash as _creation_hash,
    deterministic_order as _permuted,
    stable_digest as _digest,
    verification_hash as _verification_hash,
)
from .scenario_language import (
    NARRATIVE_FRAME_IDS,
    DynamicNarrativeComposer,
    compose_title as _title,
)
from .scenario_surface import audit_scenario_surface


SCENARIO_FORGE_VERSION = "scenario-forge-v1"
SCENARIO_PROVENANCE = (
    "Complexity original authored semantic taxonomy and narrative frames; "
    "no third-party source utterances and no model-generated dialogue."
)
MIN_UNIQUE_SENTENCE_RATE = 0.45
MIN_QUESTION_RATE = 0.25
MAX_QUESTION_RATE = 0.30
MIN_MEAN_SENTENCE_WORDS = 14.0
MAX_MEAN_SENTENCE_WORDS = 20.0
MIN_TRANSITIONS_PER_SENTENCE = 0.10
MAX_TRANSITIONS_PER_SENTENCE = 0.30

SCENARIO_SCHEMA = pa.schema(
    [
        ("scenario_id", pa.string()),
        ("creation_hash", pa.string()),
        ("verification_hash", pa.string()),
        ("family", pa.string()),
        ("domain", pa.string()),
        ("intent", pa.string()),
        ("title", pa.string()),
        ("trigger", pa.string()),
        ("situation", pa.string()),
        ("narrative_frame", pa.string()),
        ("goal", pa.string()),
        ("constraint", pa.string()),
        ("state", pa.string()),
        ("desired_outcome", pa.string()),
        ("fallback", pa.string()),
        ("risk_level", pa.string()),
        ("split", pa.string()),
        ("semantic_signature", pa.string()),
        ("semantic_payload", pa.string()),
        ("response_contract", pa.list_(pa.string())),
        ("source_structure_keys", pa.list_(pa.string())),
        ("model_generated_dialogue", pa.bool_()),
        ("provenance", pa.string()),
        ("license", pa.string()),
        ("version", pa.string()),
    ]
)


class SemanticAtom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atom_id: str = Field(alias="id")
    label: str

    @field_validator("atom_id", "label")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("semantic atoms cannot be empty")
        return value


class IntentSpec(SemanticAtom):
    goal_template: str = Field(alias="goalTemplate")

    @field_validator("goal_template")
    @classmethod
    def valid_template(cls, value: str) -> str:
        value = value.strip()
        if "{subject}" not in value:
            raise ValueError("intent goalTemplate must contain {subject}")
        return value


class DomainSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain_id: str = Field(alias="id")
    label: str
    subject: str
    context: str
    risk_level: str = Field(default="low", alias="riskLevel")

    @field_validator("domain_id", "label", "subject", "context", "risk_level")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("domain fields cannot be empty")
        return value


class CompatibilitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_outcomes: dict[str, list[str]] = Field(alias="intentOutcomes")
    domain_intents: dict[str, list[str]] = Field(alias="domainIntents")
    domain_constraints: dict[str, list[str]] = Field(alias="domainConstraints")
    state_outcomes: dict[str, list[str]] = Field(alias="stateOutcomes")
    state_fallbacks: dict[str, list[str]] = Field(alias="stateFallbacks")
    risk_fallbacks: dict[str, list[str]] = Field(alias="riskFallbacks")

    @field_validator(
        "intent_outcomes",
        "domain_intents",
        "domain_constraints",
        "state_outcomes",
        "state_fallbacks",
        "risk_fallbacks",
    )
    @classmethod
    def non_empty_matrix(cls, matrix: dict[str, list[str]]) -> dict[str, list[str]]:
        if not matrix:
            raise ValueError("compatibility matrices cannot be empty")
        for key, values in matrix.items():
            if not key.strip() or not values or len(values) != len(set(values)):
                raise ValueError(
                    "compatibility rows require a non-empty key and unique values"
                )
        return matrix


class ScenarioFamilySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family_id: str = Field(alias="id")
    label: str
    target: int = Field(gt=0)
    domains: list[DomainSpec]
    intents: list[IntentSpec]
    constraints: list[SemanticAtom]
    states: list[SemanticAtom]
    outcomes: list[SemanticAtom]
    fallbacks: list[SemanticAtom]
    compatibility: CompatibilitySpec
    response_contract: list[str] = Field(alias="responseContract")

    @field_validator("family_id", "label")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("scenario family fields cannot be empty")
        return value

    @field_validator(
        "domains", "intents", "constraints", "states", "outcomes", "fallbacks"
    )
    @classmethod
    def populated_axes(cls, values: list[Any]) -> list[Any]:
        if not values:
            raise ValueError("scenario family axes cannot be empty")
        return values

    @field_validator("response_contract")
    @classmethod
    def clean_contract(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value.strip()]
        if not cleaned or len(cleaned) != len(set(cleaned)):
            raise ValueError("responseContract must contain unique non-empty rules")
        return cleaned

    @model_validator(mode="after")
    def unique_axes_and_capacity(self) -> "ScenarioFamilySpec":
        axes: dict[str, list[Any]] = {
            "domains": self.domains,
            "intents": self.intents,
            "constraints": self.constraints,
            "states": self.states,
            "outcomes": self.outcomes,
            "fallbacks": self.fallbacks,
        }
        for name, values in axes.items():
            identifiers = [
                value.domain_id if isinstance(value, DomainSpec) else value.atom_id
                for value in values
            ]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"duplicate identifiers in {self.family_id}.{name}")

        intent_ids = {value.atom_id for value in self.intents}
        domain_ids = {value.domain_id for value in self.domains}
        constraint_ids = {value.atom_id for value in self.constraints}
        state_ids = {value.atom_id for value in self.states}
        outcome_ids = {value.atom_id for value in self.outcomes}
        fallback_ids = {value.atom_id for value in self.fallbacks}
        risk_levels = {value.risk_level for value in self.domains}
        matrices = (
            (
                "intentOutcomes",
                self.compatibility.intent_outcomes,
                intent_ids,
                outcome_ids,
            ),
            (
                "domainIntents",
                self.compatibility.domain_intents,
                domain_ids,
                intent_ids,
            ),
            (
                "domainConstraints",
                self.compatibility.domain_constraints,
                domain_ids,
                constraint_ids,
            ),
            (
                "stateOutcomes",
                self.compatibility.state_outcomes,
                state_ids,
                outcome_ids,
            ),
            (
                "stateFallbacks",
                self.compatibility.state_fallbacks,
                state_ids,
                fallback_ids,
            ),
            (
                "riskFallbacks",
                self.compatibility.risk_fallbacks,
                risk_levels,
                fallback_ids,
            ),
        )
        for name, matrix, expected_keys, allowed_values in matrices:
            if set(matrix) != expected_keys:
                raise ValueError(
                    f"{self.family_id}.{name} must cover exactly {sorted(expected_keys)}"
                )
            unknown = {
                value
                for values in matrix.values()
                for value in values
                if value not in allowed_values
            }
            if unknown:
                raise ValueError(
                    f"{self.family_id}.{name} references unknown IDs: {sorted(unknown)}"
                )

        for domain in self.domains:
            for state in self.states:
                allowed = set(self.compatibility.state_fallbacks[state.atom_id]) & set(
                    self.compatibility.risk_fallbacks[domain.risk_level]
                )
                if not allowed:
                    raise ValueError(
                        f"{self.family_id} has no fallback for domain "
                        f"{domain.domain_id}, state {state.atom_id}, risk {domain.risk_level}"
                    )

        signature_capacity = 0
        for domain in self.domains:
            for intent_id in self.compatibility.domain_intents[domain.domain_id]:
                for state in self.states:
                    compatible_outcomes = set(
                        self.compatibility.intent_outcomes[intent_id]
                    ) & set(self.compatibility.state_outcomes[state.atom_id])
                    signature_capacity += len(
                        self.compatibility.domain_constraints[domain.domain_id]
                    ) * len(compatible_outcomes)
        if self.target > signature_capacity:
            raise ValueError(
                f"family {self.family_id} requests {self.target} scenarios but "
                f"its semantic signature capacity is {signature_capacity}"
            )
        return self


class ScenarioForgeMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    dataset_id: str = Field(alias="datasetId")
    title: str
    version: str
    language: str = "en"
    license: str
    source: str
    description: str

    @field_validator(
        "dataset_id", "title", "version", "language", "license", "source", "description"
    )
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Scenario Forge metadata cannot be empty")
        return value


class ScenarioForgeRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: str
    seed: int = 42
    validation_percent: int = Field(default=5, alias="validationPercent", ge=1, le=25)
    metadata: ScenarioForgeMetadata
    families: list[ScenarioFamilySpec]

    @model_validator(mode="after")
    def valid_registry(self) -> "ScenarioForgeRegistry":
        if self.format != SCENARIO_FORGE_VERSION:
            raise ValueError(f"unsupported Scenario Forge format: {self.format}")
        identifiers = [family.family_id for family in self.families]
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError(
                "Scenario Forge family identifiers must be non-empty and unique"
            )
        return self


def _domain_quotas(family: ScenarioFamilySpec, seed: int) -> dict[str, int]:
    base, remainder = divmod(family.target, len(family.domains))
    order = sorted(
        family.domains,
        key=lambda domain: _digest(f"{seed}:{family.family_id}:{domain.domain_id}"),
    )
    extra = {domain.domain_id for domain in order[:remainder]}
    return {
        domain.domain_id: base + int(domain.domain_id in extra)
        for domain in family.domains
    }


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
    for _, _, family_id in sorted(
        remainders, key=lambda value: (-value[0], value[1])
    )[
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
                if candidate <= family_target and candidate not in subsets:
                    subsets[candidate] = (*selected, key)
        if family_target not in subsets:
            sizes = [len(group_rows) for _, group_rows in ordered]
            raise ValueError(
                f"cannot form exact validation holdout of {family_target} rows for "
                f"{family_id} from domain-intent group sizes {sizes}"
            )
        selected_groups = set(subsets[family_target])
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
    }
    if family.family_id not in family_specific:
        raise ValueError(f"no payload contract for family {family.family_id}")
    return {**common, **family_specific[family.family_id]}


def compile_scenarios(registry: ScenarioForgeRegistry) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    language = DynamicNarrativeComposer(seed=registry.seed)
    for family in registry.families:
        quotas = _domain_quotas(family, registry.seed)
        for domain in family.domains:
            combinations = _semantic_combinations(family, domain, seed=registry.seed)
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
    audit_scenarios(rows, registry)
    return rows


def audit_scenarios(
    rows: list[dict[str, Any]], registry: ScenarioForgeRegistry
) -> dict[str, Any]:
    expected_total = sum(family.target for family in registry.families)
    if len(rows) != expected_total:
        raise ValueError(f"expected {expected_total} scenarios, found {len(rows)}")
    if len({row["scenario_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate Scenario Forge IDs")
    if len({row["semantic_signature"] for row in rows}) != len(rows):
        raise ValueError("duplicate Scenario Forge semantic signatures")
    if len({row["semantic_payload"] for row in rows}) != len(rows):
        raise ValueError("duplicate Scenario Forge semantic payloads")
    if len({row["situation"] for row in rows}) != len(rows):
        raise ValueError("duplicate Scenario Forge situations")
    for field in ("title", "goal"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError(f"duplicate Scenario Forge {field} values")
    if any(row["model_generated_dialogue"] for row in rows):
        raise ValueError("Scenario Forge must not contain model-generated dialogue")
    if any(row["provenance"] != SCENARIO_PROVENANCE for row in rows):
        raise ValueError("Scenario Forge provenance mismatch")
    for row in rows:
        expected_creation = _creation_hash(row["semantic_signature"])
        if row["creation_hash"] != expected_creation:
            raise ValueError(f"creation hash mismatch for {row['scenario_id']}")
        if row["scenario_id"] != f"scenario:{expected_creation[:24]}":
            raise ValueError(f"scenario ID mismatch for {row['scenario_id']}")
        if row["verification_hash"] != _verification_hash(row):
            raise ValueError(f"verification hash mismatch for {row['scenario_id']}")

    expected_by_family = {
        family.family_id: family.target for family in registry.families
    }
    actual_by_family = Counter(row["family"] for row in rows)
    if dict(actual_by_family) != expected_by_family:
        raise ValueError(
            f"family allocation mismatch: expected {expected_by_family}, "
            f"found {dict(actual_by_family)}"
        )

    payload_contracts = {
        "practical_action": {"requested_action", "missing_fact"},
        "explanation_learning": {
            "learning_goal",
            "misconception_risk",
            "check_for_understanding",
        },
        "troubleshooting": {
            "diagnostic_goal",
            "observed_symptom",
            "stop_condition",
        },
        "writing_transformation": {
            "transformation",
            "source_state",
            "audience_result",
        },
        "planning_comparison": {
            "planning_goal",
            "options_state",
            "decision_rule",
        },
        "conversation_empathy": {
            "conversational_goal",
            "emotion_state",
            "support_boundary",
        },
        "safety_uncertainty": {
            "safe_goal",
            "risk_state",
            "safety_boundary",
            "safe_alternative",
        },
    }
    common_payload = {
        "domain_context",
        "subject",
        "constraint",
        "current_state",
        "success_condition",
        "fallback",
    }
    for row in rows:
        payload = json.loads(row["semantic_payload"])
        required = common_payload | payload_contracts[row["family"]]
        if not required <= set(payload):
            missing = sorted(required - set(payload))
            raise ValueError(
                f"scenario {row['scenario_id']} is missing payload fields: {missing}"
            )

    domain_counts: dict[str, dict[str, int]] = defaultdict(dict)
    axis_coverage: dict[str, dict[str, int]] = {}
    compatibility_violations: list[str] = []
    for family in registry.families:
        family_rows = [row for row in rows if row["family"] == family.family_id]
        constraint_by_label = {
            value.label: value.atom_id for value in family.constraints
        }
        state_by_label = {value.label: value.atom_id for value in family.states}
        outcome_by_label = {value.label: value.atom_id for value in family.outcomes}
        fallback_by_label = {value.label: value.atom_id for value in family.fallbacks}
        domain_by_id = {value.domain_id: value for value in family.domains}
        for row in family_rows:
            intent_id = row["intent"]
            constraint_id = constraint_by_label[row["constraint"]]
            state_id = state_by_label[row["state"]]
            outcome_id = outcome_by_label[row["desired_outcome"]]
            fallback_id = fallback_by_label[row["fallback"]]
            domain = domain_by_id[row["domain"]]
            if outcome_id not in family.compatibility.intent_outcomes[intent_id]:
                compatibility_violations.append(f"{row['scenario_id']}:intent_outcome")
            if intent_id not in family.compatibility.domain_intents[domain.domain_id]:
                compatibility_violations.append(f"{row['scenario_id']}:domain_intent")
            if (
                constraint_id
                not in family.compatibility.domain_constraints[domain.domain_id]
            ):
                compatibility_violations.append(
                    f"{row['scenario_id']}:domain_constraint"
                )
            if outcome_id not in family.compatibility.state_outcomes[state_id]:
                compatibility_violations.append(f"{row['scenario_id']}:state_outcome")
            if (
                fallback_id not in family.compatibility.state_fallbacks[state_id]
                or fallback_id
                not in family.compatibility.risk_fallbacks[domain.risk_level]
            ):
                compatibility_violations.append(
                    f"{row['scenario_id']}:risk_state_fallback"
                )
        counts = Counter(row["domain"] for row in family_rows)
        domain_counts[family.family_id] = dict(sorted(counts.items()))
        if max(counts.values()) - min(counts.values()) > 1:
            raise ValueError(f"unbalanced domains in family {family.family_id}")
        coverage = {
            "domains": len({row["domain"] for row in family_rows}),
            "intents": len({row["intent"] for row in family_rows}),
            "constraints": len({row["constraint"] for row in family_rows}),
            "states": len({row["state"] for row in family_rows}),
            "outcomes": len({row["desired_outcome"] for row in family_rows}),
        }
        expected_coverage = {
            "domains": len(family.domains),
            "intents": len(family.intents),
            "constraints": len(family.constraints),
            "states": len(family.states),
            "outcomes": len(family.outcomes),
        }
        if coverage != expected_coverage:
            raise ValueError(
                f"incomplete semantic coverage in {family.family_id}: "
                f"expected {expected_coverage}, found {coverage}"
            )
        axis_coverage[family.family_id] = coverage

    if compatibility_violations:
        raise ValueError(
            "Scenario Forge compatibility violations: "
            + ", ".join(compatibility_violations[:10])
        )

    split_counts = Counter(row["split"] for row in rows)
    expected_validation = round(len(rows) * registry.validation_percent / 100)
    if split_counts["validation"] != expected_validation:
        raise ValueError(
            f"expected exactly {expected_validation} validation rows, found "
            f"{split_counts['validation']}"
        )
    train_groups = {
        (row["family"], row["domain"], row["intent"])
        for row in rows
        if row["split"] == "train"
    }
    validation_groups = {
        (row["family"], row["domain"], row["intent"])
        for row in rows
        if row["split"] == "validation"
    }
    leaking_groups = train_groups & validation_groups
    if leaking_groups:
        raise ValueError(
            "validation domain-intent groups leak into training: "
            + ", ".join(map(str, sorted(leaking_groups)[:10]))
        )

    surface_audit = audit_scenario_surface(rows)
    if surface_audit["issues"]:
        raise ValueError(
            "Scenario Forge surface composition lint failed: "
            + ", ".join(
                f"{issue['scenario_id']}:{issue['kind']}"
                for issue in surface_audit["issues"][:10]
            )
        )
    surface_stats = surface_audit["stats"]
    if surface_stats["unique_sentence_rate"] < MIN_UNIQUE_SENTENCE_RATE:
        raise ValueError(
            "Scenario Forge surface language is too repetitive: "
            f"{surface_stats['unique_sentence_rate']:.3f} < "
            f"{MIN_UNIQUE_SENTENCE_RATE:.3f}"
        )
    if not MIN_QUESTION_RATE <= surface_stats["question_rate"] <= MAX_QUESTION_RATE:
        raise ValueError(
            "Scenario Forge question rate is outside the intended range: "
            f"{surface_stats['question_rate']:.3f}"
        )
    if not (
        MIN_MEAN_SENTENCE_WORDS
        <= surface_stats["mean_sentence_words"]
        <= MAX_MEAN_SENTENCE_WORDS
    ):
        raise ValueError(
            "Scenario Forge mean sentence length is outside the post-training "
            f"target: {surface_stats['mean_sentence_words']:.3f}"
        )
    if not (
        MIN_TRANSITIONS_PER_SENTENCE
        <= surface_stats["transitions_per_sentence"]
        <= MAX_TRANSITIONS_PER_SENTENCE
    ):
        raise ValueError(
            "Scenario Forge transition density is outside the post-training "
            f"target: {surface_stats['transitions_per_sentence']:.3f}"
        )

    morphology_audit = audit_verb_phrases(
        [intent.label for family in registry.families for intent in family.intents]
    )

    return {
        "scenarios": len(rows),
        "unique_ids": len({row["scenario_id"] for row in rows}),
        "unique_semantic_signatures": len({row["semantic_signature"] for row in rows}),
        "unique_semantic_payloads": len({row["semantic_payload"] for row in rows}),
        "unique_situations": len({row["situation"] for row in rows}),
        "unique_titles": len({row["title"] for row in rows}),
        "unique_goals": len({row["goal"] for row in rows}),
        "unique_triggers": len({row["trigger"] for row in rows}),
        "unique_creation_hashes": len({row["creation_hash"] for row in rows}),
        "unique_verification_hashes": len({row["verification_hash"] for row in rows}),
        "model_generated_dialogue_rows": 0,
        "family_counts": dict(sorted(actual_by_family.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "axis_coverage": dict(sorted(axis_coverage.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "split_holdout_unit": "family+domain+intent",
        "split_group_overlap": 0,
        "validation_family_counts": dict(
            sorted(
                Counter(
                    row["family"] for row in rows if row["split"] == "validation"
                ).items()
            )
        ),
        "risk_counts": dict(sorted(Counter(row["risk_level"] for row in rows).items())),
        "surface_stats": surface_stats,
        "surface_language_audit": {
            key: value for key, value in surface_audit.items() if key != "stats"
        },
        "morphology_audit": morphology_audit,
        "payload_contract_match_ratio": 1.0,
        "compatibility_match_ratio": 1.0,
    }


def load_scenario_registry(path: Path) -> ScenarioForgeRegistry:
    return ScenarioForgeRegistry.model_validate(json.loads(path.read_text()))


def build_scenario_forge(
    registry_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    registry = load_scenario_registry(registry_path)
    rows = compile_scenarios(registry)
    audit = audit_scenarios(rows, registry)

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    parquet_path = temporary / "scenarios.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCENARIO_SCHEMA),
        parquet_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    jsonl_path = temporary / "scenarios.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    audit_path = temporary / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    files = {}
    for path in (parquet_path, jsonl_path, audit_path):
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    manifest = {
        "format": SCENARIO_FORGE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": registry.metadata.model_dump(mode="json", by_alias=True),
        "seed": registry.seed,
        "validation_percent": registry.validation_percent,
        "input": {
            "path": str(registry_path),
            "sha256": file_sha256(registry_path),
        },
        "counts": {
            "scenarios": len(rows),
            "families": len(registry.families),
            "by_family": audit["family_counts"],
            "by_split": audit["split_counts"],
        },
        "generation": {
            "model_generated_dialogue": False,
            "third_party_utterances_accessed": False,
            "language_selection": "seeded_dynamic_least_used",
            "narrative_frames": len(NARRATIVE_FRAME_IDS),
            "provenance": SCENARIO_PROVENANCE,
        },
        "audit": audit,
        "files": files,
    }
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return manifest
