from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Iterable


REQUIRED_MODEL_FIELDS = (
    "example_id",
    "task",
    "split",
    "messages",
    "prompt",
    "response",
    "final_response",
    "source",
    "license",
    "version",
)
_PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]+\])?\}")
_RENDERING_DEFECTS = {
    "double_terminal_period": re.compile(r"(?<!\.)\.\.(?!\.)"),
    "missing_sentence_space": re.compile(r"(?<=[a-z0-9][.!?])(?=[A-Z])"),
    "duplicated_determiner": re.compile(r"\b(?:the a|the an|a the|an the)\b", re.I),
    "broken_relative_clause": re.compile(r"\bbecause it which\b", re.I),
    "same_plus_article": re.compile(r"\bthe same (?:a|an)\b", re.I),
    "capitalized_verb_after_connector": re.compile(
        r"\b(?:First|Directly|Specifically|Next|Then),\s+"
        r"(?:Reduce|Use|Contact|Tell|Keep|Call|Move|Avoid|Check|Stop|Leave|Ask)\b"
    ),
    "sentence_inside_noun_slot": re.compile(
        r"\bthe (?:Tell|Use|Contact|Reduce|Keep|Call|Move|Avoid|Check) the\b"
    ),
    "capitalized_clause_after_comma": re.compile(r",\s+The\b"),
    "orphan_dash_label": re.compile(r"-\s+—"),
}
_POSITIVE_VERDICT = re.compile(
    r"\b(?:yes|correct|true|accurate|confirmed|confirms|valid|holds|passes|"
    r"agrees|right|matches|stands|succeeds)\b|"
    r"\bchecks out\b|\bno correction\b|\bno change\b",
    re.I,
)
_NEGATIVE_VERDICT = re.compile(
    r"\b(?:no|not|incorrect|wrong|false|correction|correcting|replace|rejects|"
    r"fails|off|disagrees|fix|inaccurate|misses|corrects)\b|"
    r"\bnot accurate\b|\bdoes not\b|\brather than\b",
    re.I,
)


def render_think_final(thinking: str, final: str) -> str:
    thinking = thinking.strip()
    final = final.strip()
    if not thinking or not final:
        raise ValueError("think/final envelopes require visible thinking and final text")
    return f"<think>\n{thinking}\n</think>\n<final>\n{final}\n</final>"


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _assistant_messages(row: dict[str, Any]) -> list[str]:
    return [
        str(message.get("content", ""))
        for message in row.get("messages", [])
        if message.get("role") == "assistant"
    ]


def _contains_number(text: str, value: int) -> bool:
    return bool(re.search(rf"(?<!\d){re.escape(str(value))}(?!\d)", text))


def _reasoning_error(row: dict[str, Any], final: str) -> str | None:
    if row.get("task") != "reasoning_verification":
        return None
    try:
        representation = json.loads(str(row["source_representation"]))
        facts = representation["facts"]
        operation = facts["operation"]
        left = int(facts["left"])
        right = int(facts["right"])
        stored = int(facts["result"])
        candidate = int(facts["candidate"])
        kind = facts["kind"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "invalid reasoning source representation"
    calculated = {
        "addition": lambda: left + right,
        "subtraction": lambda: left - right,
        "multiplication": lambda: left * right,
        "division": lambda: left // right if right and left % right == 0 else None,
    }.get(operation)
    if calculated is None or calculated() != stored:
        return "stored arithmetic result does not recompute"
    if kind == "calculate" and not _contains_number(final, stored):
        return "calculation final omits the computed result"
    if kind == "verify_correct":
        if candidate != stored or not _POSITIVE_VERDICT.search(final):
            return "positive verification is inconsistent"
    elif kind == "verify_incorrect":
        if candidate == stored or not _NEGATIVE_VERDICT.search(final):
            return "negative verification is inconsistent"
        if not _contains_number(final, stored):
            return "correction final omits the computed result"
    elif kind != "calculate":
        return "unknown reasoning validation kind"
    return None


def _validator_error(row: dict[str, Any], final: str) -> str | None:
    try:
        representation = json.loads(str(row.get("source_representation", "")))
        validator = representation["validator"]
        kind = str(validator["kind"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return "validator metadata is unavailable"
    if kind == "arithmetic":
        return _reasoning_error(row, final)
    if kind == "exact":
        return None if final == str(validator["expected"]) else "exact answer mismatch"
    if kind == "contains":
        required = [str(value).casefold() for value in validator.get("required", [])]
        return (
            None
            if required and all(value in final.casefold() for value in required)
            else "required answer content is missing"
        )
    if kind == "regex":
        return (
            None
            if re.fullmatch(str(validator["pattern"]), final, re.S)
            else "answer format does not match its regex"
        )
    if kind == "json_equal":
        try:
            value = json.loads(final)
        except json.JSONDecodeError:
            return "answer is not valid JSON"
        return None if value == validator["expected"] else "JSON answer mismatch"
    if kind == "natural":
        words = final.split()
        minimum = int(validator.get("minimum_words", 1))
        maximum = int(validator.get("maximum_words", 512))
        forbidden = [str(value).casefold() for value in validator.get("forbidden", [])]
        if not minimum <= len(words) <= maximum:
            return "natural answer length is outside its contract"
        if any(value in final.casefold() for value in forbidden):
            return "natural answer contains forbidden content"
        return None
    return f"unsupported validator kind {kind!r}"


def audit_v2_integrity(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Audit schema, envelope, rendering, provenance, and exact correctness."""

    total = 0
    missing_fields: list[str] = []
    duplicate_ids: list[str] = []
    seen_ids: set[str] = set()
    envelope_errors: list[str] = []
    field_mismatches: list[str] = []
    placeholder_errors: list[str] = []
    rendering_errors: dict[str, list[str]] = defaultdict(list)
    provenance_errors: list[str] = []
    prompt_answers: dict[str, set[str]] = defaultdict(set)
    arithmetic_errors: list[str] = []
    validator_errors: list[str] = []

    for row in rows:
        if row.get("split", "train") != "train":
            continue
        total += 1
        example_id = str(row.get("example_id", f"row-{total}"))
        absent = [field for field in REQUIRED_MODEL_FIELDS if not row.get(field)]
        if absent:
            missing_fields.append(f"{example_id}: {', '.join(absent)}")
        if example_id in seen_ids:
            duplicate_ids.append(example_id)
        seen_ids.add(example_id)
        assistants = _assistant_messages(row)
        if len(assistants) != 1:
            envelope_errors.append(f"{example_id}: expected one assistant message")
            continue
        assistant = assistants[0].strip()
        prompt = str(row.get("prompt", "")).strip()
        response = str(row.get("response", "")).strip()
        thinking = str(row.get("reasoning_trace", "")).strip()
        final = str(row.get("final_response", "")).strip()
        expected = render_think_final(thinking, final) if thinking and final else final
        if assistant != expected or response != assistant:
            envelope_errors.append(example_id)
        users = [
            str(message.get("content", "")).strip()
            for message in row.get("messages", [])
            if message.get("role") == "user"
        ]
        if not users or (users[-1] != prompt and users[-1] not in prompt):
            field_mismatches.append(example_id)
        prompt_answers[_normalized(prompt)].add(_normalized(final))
        for text in (prompt, thinking, final):
            if _PLACEHOLDER.search(text):
                placeholder_errors.append(example_id)
            for name, pattern in _RENDERING_DEFECTS.items():
                if pattern.search(text):
                    rendering_errors[name].append(example_id)
        if not all(str(row.get(field, "")).strip() for field in ("source", "license", "version")):
            provenance_errors.append(example_id)
        arithmetic_error = _reasoning_error(row, final)
        if arithmetic_error is not None:
            arithmetic_errors.append(f"{example_id}: {arithmetic_error}")
        validator_error = _validator_error(row, final)
        if validator_error is not None:
            validator_errors.append(f"{example_id}: {validator_error}")

    conflicting_prompts = {
        prompt: sorted(answers)
        for prompt, answers in prompt_answers.items()
        if len(answers) > 1
    }
    violations = []
    for condition, message in (
        (missing_fields, "required model fields are missing"),
        (duplicate_ids, "example IDs are duplicated"),
        (envelope_errors, "assistant envelopes or response fields are inconsistent"),
        (field_mismatches, "prompt fields do not match the user message"),
        (placeholder_errors, "unrendered placeholders remain"),
        (rendering_errors, "punctuation or composition defects remain"),
        (provenance_errors, "source, license, or version is missing"),
        (conflicting_prompts, "identical prompts have conflicting answers"),
        (arithmetic_errors, "arithmetic validation failed"),
        (validator_errors, "deterministic response validation failed"),
    ):
        if condition:
            violations.append(message)
    return {
        "format": "complexity-card-corpus-v2-integrity-audit-v1",
        "passed": not violations,
        "violations": violations,
        "rows": total,
        "missing_field_count": len(missing_fields),
        "duplicate_id_count": len(duplicate_ids),
        "envelope_error_count": len(envelope_errors),
        "field_mismatch_count": len(field_mismatches),
        "placeholder_error_count": len(placeholder_errors),
        "rendering_error_counts": {
            name: len(ids) for name, ids in sorted(rendering_errors.items())
        },
        "provenance_error_count": len(provenance_errors),
        "conflicting_prompt_count": len(conflicting_prompts),
        "arithmetic_error_count": len(arithmetic_errors),
        "validator_error_count": len(validator_errors),
        "examples": {
            "missing_fields": missing_fields[:20],
            "duplicate_ids": duplicate_ids[:20],
            "envelope_errors": envelope_errors[:20],
            "field_mismatches": field_mismatches[:20],
            "placeholder_errors": placeholder_errors[:20],
            "rendering_errors": {
                name: ids[:20] for name, ids in sorted(rendering_errors.items())
            },
            "provenance_errors": provenance_errors[:20],
            "conflicting_prompts": dict(list(conflicting_prompts.items())[:20]),
            "arithmetic_errors": arithmetic_errors[:20],
            "validator_errors": validator_errors[:20],
        },
    }


__all__ = ("REQUIRED_MODEL_FIELDS", "audit_v2_integrity", "render_think_final")
