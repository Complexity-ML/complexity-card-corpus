from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]*$")


class Relation(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    type: str
    target_key: str = Field(alias="targetKey")
    target_dataset_id: str | None = Field(default=None, alias="targetDatasetId")
    detail: str | None = None

    @field_validator("type", "target_key")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("relation fields cannot be empty")
        return value


class Card(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    kind: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    summary: str
    description: str | None = None
    facts: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    relations: list[Relation] = Field(default_factory=list)

    @field_validator("key")
    @classmethod
    def valid_key(cls, value: str) -> str:
        value = value.strip()
        if not KEY_RE.fullmatch(value):
            raise ValueError(
                "keys must use lowercase letters, digits, dots, colons, underscores or hyphens"
            )
        return value

    @field_validator("kind", "name", "summary")
    @classmethod
    def required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required card text cannot be empty")
        return value

    @field_validator("aliases", "facts", "tags")
    @classmethod
    def clean_list(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("card lists cannot contain duplicates")
        return cleaned


class DatasetMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    dataset_id: str = Field(alias="datasetId")
    title: str
    domain: str
    language: str = "en"
    version: str
    split: Literal["train", "validation", "test"]
    source: str
    source_urls: list[str] = Field(default_factory=list, alias="sourceUrls")
    license: str
    description: str

    @field_validator(
        "dataset_id", "title", "domain", "language", "version", "source", "license", "description"
    )
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("dataset metadata cannot be empty")
        return value


class CardDataset(BaseModel):
    metadata: DatasetMetadata
    cards: list[Card]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str
    source_message_id: str

    @field_validator("content", "source_message_id")
    @classmethod
    def message_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("chat message fields cannot be empty")
        return value


class AlignmentCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    mode: Literal["instruct", "chat"]
    split: Literal["train", "validation", "test"]
    language: str = "en"
    messages: list[ChatMessage]
    rendered_text: str
    quality_score: float
    source_dataset: str
    source_revision: str
    source_tree_id: str
    license: str

    @model_validator(mode="after")
    def valid_conversation(self) -> "AlignmentCard":
        if len(self.messages) < 2:
            raise ValueError("alignment cards require at least one user/assistant pair")
        expected = "user"
        for message in self.messages:
            if message.role != expected:
                raise ValueError("chat roles must alternate and begin with user")
            expected = "assistant" if expected == "user" else "user"
        if self.messages[-1].role != "assistant":
            raise ValueError("alignment cards must end with an assistant response")
        if self.mode == "instruct" and len(self.messages) != 2:
            raise ValueError("instruct cards contain exactly one user/assistant pair")
        if self.mode == "chat" and len(self.messages) < 4:
            raise ValueError("chat cards contain at least two user/assistant turns")
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be between zero and one")
        return self

