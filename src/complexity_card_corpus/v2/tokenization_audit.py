from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .chat import (
    chat_template_contract,
    render_system_prefix,
    render_user_turn,
)
from .tokenizer import IGNORE_INDEX, load_encoding


_MARKERS = ("<think>", "</think>", "<final>", "</final>")


def audit_v2_tokenization(
    rows: Iterable[dict[str, Any]],
    tokenizer_root: str | Path,
    *,
    maximum_examples: int = 128,
) -> dict[str, Any]:
    encoding, tokenizer_config = load_encoding(Path(tokenizer_root))
    contract = chat_template_contract()
    eos_token = str(tokenizer_config.get("eos_token", contract["eos_token"]))
    try:
        eos_id = encoding.encode_single_token(eos_token)
    except KeyError:
        eos_id = encoding.eot_token
    unknown_id = None
    unknown_token = tokenizer_config.get("unk_token")
    if isinstance(unknown_token, str) and unknown_token:
        try:
            unknown_id = encoding.encode_single_token(unknown_token)
        except KeyError:
            pass

    marker_tokens = {
        marker: encoding.encode(marker, disallowed_special=())
        for marker in _MARKERS
    }
    marker_failures = [
        marker
        for marker, tokens in marker_tokens.items()
        if not tokens or (unknown_id is not None and unknown_id in tokens)
    ]

    failures = []
    examples = 0
    roundtrip_failures = 0
    loss_mask_failures = 0
    envelope_failures = 0
    for row in rows:
        if row.get("split", "train") != "train":
            continue
        if examples >= maximum_examples:
            break
        prompt = str(row.get("prompt", ""))
        response = str(row.get("response", row.get("final_response", "")))
        prefix_text = (
            render_system_prefix(contract)
            + render_user_turn(prompt, contract)
            + contract["assistant_prefix"]
        )
        prefix_ids = encoding.encode(prefix_text, disallowed_special=())
        response_ids = encoding.encode(response, disallowed_special=())
        full_ids = [*prefix_ids, *response_ids, eos_id]
        target_labels = [
            *([IGNORE_INDEX] * len(prefix_ids)),
            *response_ids,
            eos_id,
        ]
        inputs = full_ids[:-1]
        labels = target_labels[1:]
        decoded = encoding.decode(full_ids)
        if encoding.encode(decoded, disallowed_special=()) != full_ids:
            roundtrip_failures += 1
        supervised = [label for label in labels if label != IGNORE_INDEX]
        if (
            len(inputs) != len(labels)
            or labels[: max(0, len(prefix_ids) - 1)]
            != [IGNORE_INDEX] * max(0, len(prefix_ids) - 1)
            or supervised != [*response_ids, eos_id]
        ):
            loss_mask_failures += 1
        has_any_marker = any(marker in response for marker in _MARKERS)
        if has_any_marker and not all(marker in decoded for marker in _MARKERS):
            envelope_failures += 1
        examples += 1

    if not examples:
        failures.append("no train examples were available for tokenization")
    if roundtrip_failures:
        failures.append("chat serialization does not round-trip through tokenizer")
    if loss_mask_failures:
        failures.append("assistant-only loss mask is misaligned")
    if marker_failures or envelope_failures:
        failures.append("think/final markers are not preserved by tokenizer")
    return {
        "format": "complexity-card-corpus-v2-tokenization-audit-v1",
        "passed": not failures,
        "failures": failures,
        "examples": examples,
        "roundtrip_failures": roundtrip_failures,
        "loss_mask_failures": loss_mask_failures,
        "envelope_failures": envelope_failures,
        "marker_failures": marker_failures,
        "marker_token_ids": marker_tokens,
        "chat_template_id": contract["id"],
        "tokenizer": tokenizer_config.get("encoding_name"),
    }


__all__ = ("audit_v2_tokenization",)
