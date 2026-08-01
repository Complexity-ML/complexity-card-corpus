from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskHand:
    """A concrete, solvable hand of cards for one training scenario."""

    data: str
    goal: str
    answer: str
    contract: tuple[str, ...]
    situation_title: str | None = None
    situation: str | None = None
    rule: str | None = None


def _number(key: str, low: int, high: int) -> int:
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return low + value % (high - low + 1)


def _pick(key: str, values: tuple[str, ...]) -> str:
    return values[_number(key, 0, len(values) - 1)]


def _card_pick(
    row: dict[str, Any], variant: int, deck: str, values: tuple[str, ...]
) -> str:
    """Deal one deterministic surface card from a compatible deck."""
    return _pick(f"{deck}:{row['scenario_id']}:{variant}", values)


def _code(row: dict[str, Any]) -> str:
    return row["scenario_id"].split(":")[-1][:6].upper()


def _payload(row: dict[str, Any]) -> dict[str, str]:
    return json.loads(row["semantic_payload"])


def _lower_sentence_initial(value: str) -> str:
    """Lower a sentence initial without corrupting an acronym such as RAM."""
    initial = re.match(r"[A-Za-z]+", value)
    if initial is None or initial.group(0).isupper():
        return value
    return value[:1].lower() + value[1:]
