from __future__ import annotations

import hashlib
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


SCENARIO_FORGE_VERSION = "scenario-forge-v1"
SCENARIO_PROVENANCE = (
    "Complexity original authored semantic taxonomy; no source utterances and "
    "no model-generated prose."
)

SCENARIO_SCHEMA = pa.schema(
    [
        ("scenario_id", pa.string()),
        ("creation_hash", pa.string()),
        ("verification_hash", pa.string()),
        ("family", pa.string()),
        ("domain", pa.string()),
        ("intent", pa.string()),
        ("title", pa.string()),
        ("situation", pa.string()),
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
        ("surface_text_generated", pa.bool_()),
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
    domain_constraints: dict[str, list[str]] = Field(alias="domainConstraints")
    state_fallbacks: dict[str, list[str]] = Field(alias="stateFallbacks")
    risk_fallbacks: dict[str, list[str]] = Field(alias="riskFallbacks")

    @field_validator(
        "intent_outcomes", "domain_constraints", "state_fallbacks", "risk_fallbacks"
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
            ("intentOutcomes", self.compatibility.intent_outcomes, intent_ids, outcome_ids),
            (
                "domainConstraints",
                self.compatibility.domain_constraints,
                domain_ids,
                constraint_ids,
            ),
            ("stateFallbacks", self.compatibility.state_fallbacks, state_ids, fallback_ids),
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

        signature_capacity = sum(
            len(self.compatibility.domain_constraints[domain.domain_id])
            * len(self.states)
            * sum(
                len(self.compatibility.intent_outcomes[intent.atom_id])
                for intent in self.intents
            )
            for domain in self.domains
        )
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
            raise ValueError("Scenario Forge family identifiers must be non-empty and unique")
        return self


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _stable_index(value: str, size: int) -> int:
    return int.from_bytes(_digest(value)[:8], "big") % size


def _split(scenario_id: str, validation_percent: int) -> str:
    bucket = _stable_index(f"split:{scenario_id}", 100)
    return "validation" if bucket < validation_percent else "train"


def _creation_hash(signature: str) -> str:
    return hashlib.sha256(signature.encode()).hexdigest()


def _verification_hash(row: dict[str, Any]) -> str:
    excluded = {"verification_hash"}
    canonical = {key: value for key, value in row.items() if key not in excluded}
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _permuted(values: list[Any], key: str) -> list[Any]:
    return sorted(values, key=lambda value: _digest(f"{key}:{value.atom_id}"))


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


def _semantic_combinations(
    family: ScenarioFamilySpec,
    domain: DomainSpec,
    *,
    seed: int,
) -> list[tuple[IntentSpec, SemanticAtom, SemanticAtom, SemanticAtom]]:
    prefix = f"{seed}:{family.family_id}:{domain.domain_id}"
    intents = _permuted(family.intents, f"{prefix}:intent")
    constraint_by_id = {value.atom_id: value for value in family.constraints}
    outcome_by_id = {value.atom_id: value for value in family.outcomes}
    constraints = _permuted(
        [
            constraint_by_id[value]
            for value in family.compatibility.domain_constraints[domain.domain_id]
        ],
        f"{prefix}:constraint",
    )
    states = _permuted(family.states, f"{prefix}:state")
    combinations = [
        (intent, constraint, state, outcome_by_id[outcome_id])
        for intent in intents
        for constraint in constraints
        for state in states
        for outcome_id in family.compatibility.intent_outcomes[intent.atom_id]
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


def _situation(
    domain: DomainSpec,
    intent: IntentSpec,
    constraint: SemanticAtom,
    state: SemanticAtom,
    outcome: SemanticAtom,
) -> str:
    return " ".join(
        (
            domain.context,
            f"The immediate trigger is this: {state.label}",
            f"The person now needs to {intent.label}.",
            f"The governing boundary is: {constraint.label}",
            f"Success in this situation means: {outcome.label}",
        )
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
    for family in registry.families:
        quotas = _domain_quotas(family, registry.seed)
        for domain in family.domains:
            combinations = _semantic_combinations(
                family, domain, seed=registry.seed
            )
            for rank, (intent, constraint, state, outcome) in enumerate(
                combinations[: quotas[domain.domain_id]]
            ):
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
                fallback_candidates = _compatible_fallbacks(family, domain, state)
                fallback = fallback_candidates[
                    _stable_index(
                        f"{registry.seed}:{signature}:fallback",
                        len(fallback_candidates),
                    )
                ]
                goal = intent.goal_template.format(
                    subject=domain.subject,
                    context=domain.context,
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
                        "title": f"{family.label}: {domain.label} — {intent.label}",
                        "situation": _situation(
                            domain, intent, constraint, state, outcome
                        ),
                        "goal": goal,
                        "constraint": constraint.label,
                        "state": state.label,
                        "desired_outcome": outcome.label,
                        "fallback": fallback.label,
                        "risk_level": domain.risk_level,
                        "split": _split(scenario_id, registry.validation_percent),
                        "semantic_signature": signature,
                        "semantic_payload": json.dumps(payload, sort_keys=True),
                        "response_contract": list(family.response_contract),
                        "source_structure_keys": [
                            f"family:{family.family_id}",
                            f"domain:{domain.domain_id}",
                            f"intent:{intent.atom_id}",
                        ],
                        "surface_text_generated": False,
                        "provenance": SCENARIO_PROVENANCE,
                        "license": registry.metadata.license,
                        "version": registry.metadata.version,
                    }
                row["verification_hash"] = _verification_hash(row)
                rows.append(row)
    rows.sort(key=lambda row: row["scenario_id"])
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
    if any(row["surface_text_generated"] for row in rows):
        raise ValueError("Scenario Forge must not generate surface text")
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

    expected_by_family = {family.family_id: family.target for family in registry.families}
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
                compatibility_violations.append(
                    f"{row['scenario_id']}:intent_outcome"
                )
            if constraint_id not in family.compatibility.domain_constraints[domain.domain_id]:
                compatibility_violations.append(
                    f"{row['scenario_id']}:domain_constraint"
                )
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

    return {
        "scenarios": len(rows),
        "unique_ids": len({row["scenario_id"] for row in rows}),
        "unique_semantic_signatures": len(
            {row["semantic_signature"] for row in rows}
        ),
        "unique_semantic_payloads": len({row["semantic_payload"] for row in rows}),
        "unique_situations": len({row["situation"] for row in rows}),
        "unique_creation_hashes": len({row["creation_hash"] for row in rows}),
        "unique_verification_hashes": len(
            {row["verification_hash"] for row in rows}
        ),
        "surface_text_rows": 0,
        "family_counts": dict(sorted(actual_by_family.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "axis_coverage": dict(sorted(axis_coverage.items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "risk_counts": dict(
            sorted(Counter(row["risk_level"] for row in rows).items())
        ),
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
        "surface_text": {
            "generated": False,
            "source_utterances_accessed": False,
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
