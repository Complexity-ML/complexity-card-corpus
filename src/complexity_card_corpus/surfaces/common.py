from __future__ import annotations

import hashlib
import re
from math import gcd
from typing import Any


DATASET_ID = "complexity-original-conversation-v1"


SURFACE_VERSION = "conversation-surface-v1"


SURFACE_LICENSE = "CC BY-NC 4.0"


SURFACE_SOURCE = "Complexity original authored conversation cards"


_VARIANT_RADIX = 32


_PLACEHOLDER = re.compile(r"\{[^{}]+\}")


_WORD = re.compile(r"[a-z0-9']+")


_LOWERCASE_I = re.compile(r"(?:^|[.!?]\s+)i(?:\s|['’])")


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _stable_index(value: str, size: int) -> int:
    if size < 1:
        raise ValueError("cannot select from an empty sequence")
    return int.from_bytes(_digest(value)[:8], "big") % size


def _split(source_card_id: str, validation_percent: int) -> str:
    return (
        "validation"
        if _stable_index(f"split:scenario-card:{source_card_id}", 100)
        < validation_percent
        else "train"
    )


def _lower_first(value: str) -> str:
    if re.match(r"^I(?:\s|['’])", value):
        return value
    return value[:1].lower() + value[1:] if value else value


def _recommendation_from_choice(value: str) -> str:
    prefix = "I will "
    recommendation = value[len(prefix) :] if value.startswith(prefix) else value
    return recommendation[:1].upper() + recommendation[1:]


def _render_messages(messages: list[dict[str, str]]) -> str:
    labels = {"user": "User", "assistant": "Assistant"}
    return "\n".join(
        f"{labels[message['role']]}: {message['content']}" for message in messages
    )


def _length_bucket(value: str) -> str:
    words = len(value.split())
    if words <= 4:
        return "very_short"
    if words <= 12:
        return "short"
    if words <= 30:
        return "medium"
    return "long"


def _position_variant_shift(attempt: int, position: int, total: int) -> int:
    if attempt == 0:
        return 0
    attempt -= 1
    if position == 0:
        return 1 + attempt % _VARIANT_RADIX
    if position == total - 1:
        return 1 + (attempt // _VARIANT_RADIX) % _VARIANT_RADIX
    return 1 + (attempt + position * 7) % _VARIANT_RADIX


def _default_variant_shift(
    blueprint: dict[str, Any], card: dict[str, str], position: int
) -> int:
    return 1 + _stable_index(
        f"default-surface:{blueprint['blueprint_id']}:{card['card_id']}:{position}",
        _VARIANT_RADIX,
    )


def _render_candidates(
    templates: tuple[str, ...],
    frames: tuple[str, ...],
    values: dict[str, str],
) -> list[str]:
    base_candidates = [template.format(**values).strip() for template in templates]
    candidates: list[str] = []
    for base_index, base in enumerate(base_candidates):
        selected_frames = frames if base_index == 0 else ("{}",)
        for frame in selected_frames:
            value = base if frame == "{}" else frame.format(_lower_first(base))
            if value not in candidates:
                candidates.append(value)
    return candidates


def _choose_variant(
    templates: tuple[str, ...],
    *,
    frames: tuple[str, ...],
    rank: int,
    stage: str,
    style: str,
    target_length: str,
    values: dict[str, str],
    variant_shift: int = 0,
) -> str:
    candidates = _render_candidates(templates, frames, values)
    exact = [value for value in candidates if _length_bucket(value) == target_length]
    if not exact:
        order = {"very_short": 0, "short": 1, "medium": 2, "long": 3}
        target = order[target_length]
        distance = min(
            abs(order[_length_bucket(value)] - target) for value in candidates
        )
        exact = [
            value
            for value in candidates
            if abs(order[_length_bucket(value)] - target) == distance
        ]
    if style in {"concise_practical", "concise_empathetic"}:
        exact.sort(key=lambda value: (len(value.split()), value))
    elif style == "stepwise_helpful":
        exact.sort(key=lambda value: (-len(value.split()), value))
    else:
        exact.sort(
            key=lambda value: _digest(
                f"style-order:{style}:{stage}:{values['card_id']}:{value}"
            )
        )
    offset = _stable_index(f"{stage}:{style}:{values['card_id']}", len(exact))
    strides = [
        value for value in range(1, len(exact) + 1) if gcd(value, len(exact)) == 1
    ]
    stride = strides[_stable_index(f"surface-stride:{stage}", len(strides))]
    return exact[(rank + offset + variant_shift * stride) % len(exact)]
