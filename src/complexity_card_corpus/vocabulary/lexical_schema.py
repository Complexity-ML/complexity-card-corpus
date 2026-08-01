from __future__ import annotations

import hashlib
import math
import re

import pyarrow as pa


LEXICAL_MINE_VERSION = "aggregate-lexical-mine-v1"


TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")


VALID_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


LEXICON_SCHEMA = pa.schema(
    [
        ("token", pa.string()),
        ("role", pa.string()),
        ("source_dataset", pa.string()),
        ("source_license", pa.string()),
        ("source_revision", pa.string()),
        ("occurrences", pa.int64()),
        ("document_count", pa.int64()),
        ("mined_unit", pa.string()),
        ("source_text_retained", pa.bool_()),
        ("release_ready", pa.bool_()),
        ("extraction_version", pa.string()),
    ]
)


TRANSITIONS = {
    "after",
    "although",
    "before",
    "because",
    "finally",
    "first",
    "following",
    "however",
    "instead",
    "meanwhile",
    "next",
    "once",
    "otherwise",
    "then",
    "therefore",
    "unless",
    "until",
    "when",
    "while",
}


INTENT_CUES = {"can", "could", "help", "let", "must", "please", "should", "to", "would"}


STATE_CUES = {
    "am",
    "are",
    "became",
    "become",
    "becomes",
    "feel",
    "feels",
    "is",
    "remain",
    "remains",
    "seem",
    "seems",
    "was",
    "were",
}


CONSTRAINT_CUES = {"avoid", "cannot", "can't", "must", "never", "should", "without"}


OUTCOME_CUES = {
    "achieve",
    "ensure",
    "result",
    "results",
    "so",
    "successful",
    "successfully",
}


BLOCKED_TOKENS = {
    "assistant",
    "http",
    "https",
    "speaker",
    "system",
    "unknown",
    "user",
    "www",
}


class _ApproxDistinct:
    """Fixed-memory linear counter for aggregate diversity statistics."""

    def __init__(self, bit_power: int = 24) -> None:
        self._bits = bytearray(1 << (bit_power - 3))
        self._mask = (1 << bit_power) - 1
        self._set_bits = 0

    def add(self, value: str) -> None:
        index = (
            int.from_bytes(
                hashlib.blake2b(value.encode(), digest_size=8).digest(), "little"
            )
            & self._mask
        )
        byte_index, bit_index = divmod(index, 8)
        bit = 1 << bit_index
        if not self._bits[byte_index] & bit:
            self._bits[byte_index] |= bit
            self._set_bits += 1

    def estimate(self) -> int:
        slots = self._mask + 1
        empty = slots - self._set_bits
        if empty <= 0:
            return slots
        return round(-slots * math.log(empty / slots))


def _words(text: str) -> list[tuple[str, bool]]:
    return [
        (match.group(0).replace("’", "'").lower(), match.group(0)[:1].isupper())
        for match in TOKEN_PATTERN.finditer(text)
    ]


def _valid_word(token: str) -> bool:
    return (
        3 <= len(token) <= 24
        and token not in BLOCKED_TOKENS
        and VALID_TOKEN.fullmatch(token) is not None
    )
