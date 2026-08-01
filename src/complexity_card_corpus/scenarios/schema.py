from __future__ import annotations

from typing import Any

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
        ("source_structure_links", pa.list_(pa.string())),
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
    includes: list[str] = Field(default_factory=list)
    families: list[ScenarioFamilySpec]

    @field_validator("includes")
    @classmethod
    def valid_includes(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("Scenario Forge includes must be unique non-empty paths")
        return cleaned

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
