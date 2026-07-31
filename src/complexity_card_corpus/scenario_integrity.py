from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol, TypeVar


class IdentifiedAtom(Protocol):
    atom_id: str


AtomT = TypeVar("AtomT", bound=IdentifiedAtom)


def stable_digest(value: str) -> bytes:
    """Return the canonical digest used for deterministic allocation."""
    return hashlib.sha256(value.encode()).digest()


def creation_hash(signature: str) -> str:
    """Bind a scenario identity to its semantic signature."""
    return hashlib.sha256(signature.encode()).hexdigest()


def verification_hash(row: dict[str, Any]) -> str:
    """Bind a compiled scenario to all final canonical fields."""
    canonical = {key: value for key, value in row.items() if key != "verification_hash"}
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def deterministic_order(values: list[AtomT], key: str) -> list[AtomT]:
    """Order semantic atoms reproducibly without assigning semantic meaning."""
    return sorted(values, key=lambda value: stable_digest(f"{key}:{value.atom_id}"))
