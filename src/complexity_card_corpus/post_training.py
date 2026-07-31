from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .build import file_sha256
from .english_morphology import correct_indefinite_articles
from .instruct import INSTRUCTION_SCHEMA
from .post_training_language import (
    CONCLUSION_FRAMES,
    FAMILY_ACTION_FRAMES,
    FAMILY_CONSTRAINT_FRAMES,
    FAMILY_OPENINGS,
    FALLBACK_FRAMES,
    RESPONSE_ORDERS,
    fallback_actions,
)


DATASET_ID = "complexity-original-post-training-v1"
DATASET_LICENSE = "CC BY-NC 4.0"
DATASET_SOURCE = "Complexity original Scenario Forge conversations"
REVIEW_GRADES = (
    "semantic_accuracy",
    "constraint_following",
    "language_quality",
    "individualization",
    "safety",
)
_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_REVIEW_STATUSES = frozenset({"pending", "approved", "rejected"})
_REVIEW_GRADE_VALUES = frozenset({"", "pass", "fail"})
_MAX_SURFACE_FORMULATION_SHARE = 0.05
_FORBIDDEN_ASSISTANT_META_PHRASES = (
    "the response should",
    "the response must",
    "the response can",
    "the answer should",
    "the answer must",
    "the answer can",
    "the explanation should",
    "the explanation must",
    "the final review should",
    "a valid answer",
    "a worked response",
)
_FORBIDDEN_USER_META_PHRASES = (
    "the assistant must",
    "the assistant should",
    "the response should",
    "the response must",
    "the answer should",
    "how should an assistant",
    "what response would",
    "which response would",
    "what should the response cover",
    "how can the response",
    "the response must reflect",
)

_INTENT_FIELD = {
    "practical_action": "requested_action",
    "explanation_learning": "learning_goal",
    "troubleshooting": "diagnostic_goal",
    "writing_transformation": "transformation",
    "planning_comparison": "planning_goal",
    "conversation_empathy": "conversational_goal",
    "safety_uncertainty": "safe_goal",
}

_ACKNOWLEDGEMENTS = (
    "Thanks, that gives me a concrete starting point. I will keep the next step tied to what we can verify.",
    "Understood. I will separate what is confirmed from what still needs checking.",
    "That helps. We can handle the immediate task first and keep the final decision reversible.",
    "I follow. There is enough context for a bounded next step, but not for unsupported certainty.",
    "Got it. I will focus on the result you need and keep a fallback available if evidence is missing.",
    "That is a useful starting point. We can make one practical choice instead of a broad assumption.",
    "Understood. I will keep the guidance specific enough to verify.",
    "That makes sense. We can move from the known facts to one cautious action.",
    "Thanks for the update. I will keep the remaining uncertainty visible.",
    "I understand. We can preserve your control while narrowing the next decision.",
    "Got it. I will use the confirmed facts and leave unsupported details open.",
    "That gives us a clear scope: one objective, one limit, and one check at the end.",
)

_PROMPT_REQUESTS = (
    "Please help me {intent} for {subject}.",
    "How can I {intent} for {subject} without guessing?",
    "Can you walk me through how to {intent} for {subject}?",
    "I need a bounded way to {intent} for {subject}. What should I do first?",
    "Please show me how to {intent} for {subject} using only the confirmed details.",
    "What is a practical way to {intent} for {subject} in this situation?",
    "Help me {intent} for {subject} without treating uncertainty as fact.",
    "How can I {intent} for {subject} while keeping control of the decision?",
    "I want to {intent} for {subject}. What would a grounded approach look like?",
    "Please give me a careful way to {intent} for {subject} from the available facts.",
    "How would you {intent} for {subject} with the information available?",
    "What is the simplest safe way to {intent} for {subject}?",
)

_CHAT_OPENERS = (
    "Here is the current state for {subject}: {state}.",
    "Before we decide about {subject}, one fact is established: {state}.",
    "I want to work through {subject}, starting from this update: {state}.",
    "The point I can confirm about {subject} is this: {state}.",
    "For {subject}, the immediate situation is now clear: {state}.",
    "One verified detail frames my question about {subject}: {state}.",
    "I need help thinking about {subject}; the known state is: {state}.",
    "This is the latest confirmed position for {subject}: {state}.",
    "My question about {subject} begins with one concrete fact: {state}.",
    "The decision around {subject} now rests on this update: {state}.",
    "Please take this as the starting point for {subject}: {state}.",
    "There is one reliable anchor for discussing {subject}: {state}.",
)

_FOLLOW_UPS = (
    "The relevant context is that {context}. Given that, how would you {intent} for {subject} without overclaiming?",
    "The surrounding facts are that {context}. How can I {intent} for {subject} responsibly?",
    "One context point matters here: {context}. What is a grounded way to {intent} for {subject}?",
    "The decision sits within this setting: {context}. How would you {intent} for {subject}?",
    "The available background is that {context}. What bounded approach would {intent} for {subject}?",
    "Keep this context in view: {context}. What practical step would help me {intent} for {subject}?",
    "The evidence comes from this setting: {context}. How can we {intent} for {subject} without guessing?",
    "This background limits what we know: {context}. How can I {intent} for {subject} and still check the result?",
    "The request is grounded in this fact: {context}. What should I do next to {intent} for {subject}?",
    "Use this context as the boundary: {context}. How would you {intent} for {subject}?",
    "The known setting is that {context}. What is the simplest safe way to {intent} for {subject}?",
    "Please keep this context in view: {context}. How can I {intent} for {subject} using only supported facts?",
)


def _stable_index(value: str, size: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % size


def _lower_first(value: str) -> str:
    value = value.strip()
    return value[:1].lower() + value[1:] if value else value


def _upper_first(value: str) -> str:
    value = value.strip()
    return value[:1].upper() + value[1:] if value else value


_INTENT_SUBJECT_TEMPLATES = {
    "apply the principle": "apply the principle to {subject}",
    "contrast nearby concepts": "contrast the concepts related to {subject}",
    "diagnose a misconception": "diagnose a misconception about {subject}",
    "explain the core mechanism": "explain the core mechanism behind {subject}",
    "walk through a worked example": "walk through a worked example of {subject}",
    "isolate the failing boundary": "isolate the failing boundary in {subject}",
    "prevent recurrence": "prevent the problem from recurring in {subject}",
    "produce a minimal reproduction": "produce a minimal reproduction of {subject}",
    "recover safely": "recover safely from {subject}",
    "verify a proposed fix": "verify a proposed fix for {subject}",
    "adapt tone for the audience": "adapt the tone of {subject} for the audience",
    "draft from structured facts": "draft {subject} from structured facts",
    "restructure for action": "restructure {subject} for action",
    "revise for clarity": "revise {subject} for clarity",
    "summarize faithfully": "summarize {subject} faithfully",
    "allocate limited resources": "allocate limited resources for {subject}",
    "compare against hard criteria": "compare {subject} against hard criteria",
    "define viable options": "define viable options for {subject}",
    "design a fallback": "design a fallback for {subject}",
    "sequence the work": "sequence the work for {subject}",
    "acknowledge the experience": "acknowledge the experience behind {subject}",
    "choose a gentle next step": "choose a gentle next step for {subject}",
    "clarify the immediate need": "clarify the immediate need in {subject}",
    "prepare a grounded conversation": "prepare a grounded conversation about {subject}",
    "reflect on meaning and progress": "reflect on the meaning and progress of {subject}",
    "clarify the safe scope": "clarify the safe scope of {subject}",
    "identify an escalation threshold": "identify an escalation threshold for {subject}",
    "offer a safe alternative": "offer a safe alternative for {subject}",
    "preserve privacy and control": "preserve privacy and control around {subject}",
    "set a clear safety boundary": "set a clear safety boundary for {subject}",
}


def _intent(payload: dict[str, str], family: str) -> str:
    return payload[_INTENT_FIELD[family]].rstrip(".")


def _surface_assignments(
    scenarios: list[dict[str, Any]], variants_per_scenario: int
) -> dict[tuple[str, int], dict[str, int]]:
    """Balance surface choices exactly while preserving deterministic output."""
    items = [
        (scenario, variant)
        for scenario in scenarios
        for variant in range(variants_per_scenario)
    ]
    assignments: dict[tuple[str, int], dict[str, int]] = {
        (row["scenario_id"], variant): {} for row, variant in items
    }
    expected_families = set(_INTENT_FIELD)
    registries = (FAMILY_OPENINGS, FAMILY_ACTION_FRAMES, FAMILY_CONSTRAINT_FRAMES)
    if any(set(registry) != expected_families for registry in registries):
        raise RuntimeError("post-training family language registries are inconsistent")
    family_groups: dict[str, list[tuple[dict[str, Any], int]]] = defaultdict(list)
    for row, variant in items:
        family_groups[row["family"]].append((row, variant))
    dimensions = (
        ("opening", FAMILY_OPENINGS),
        ("action_frame", FAMILY_ACTION_FRAMES),
        ("constraint_frame", FAMILY_CONSTRAINT_FRAMES),
    )
    for family, values in family_groups.items():
        for dimension, registry in dimensions:
            ordered = sorted(
                values,
                key=lambda item: hashlib.sha256(
                    f"{dimension}-balance:{item[0]['scenario_id']}:{item[1]}".encode()
                ).digest(),
            )
            for position, (row, variant) in enumerate(ordered):
                assignments[(row["scenario_id"], variant)][dimension] = (
                    position % len(registry[family])
                )
        order_values = sorted(
            values,
            key=lambda item: hashlib.sha256(
                f"order-balance:{item[0]['scenario_id']}:{item[1]}".encode()
            ).digest(),
        )
        for position, (row, variant) in enumerate(order_values):
            assignments[(row["scenario_id"], variant)]["order"] = (
                position % len(RESPONSE_ORDERS)
            )

    fallback_groups: dict[str, list[tuple[dict[str, Any], int]]] = defaultdict(list)
    for row, variant in items:
        fallback_groups[row["fallback"].rstrip(".")].append((row, variant))
    for fallback, values in fallback_groups.items():
        action_count = len(fallback_actions(fallback))
        ordered = sorted(
            values,
            key=lambda item: hashlib.sha256(
                f"fallback-balance:{item[0]['scenario_id']}:{item[1]}".encode()
            ).digest(),
        )
        for position, (row, variant) in enumerate(ordered):
            target = assignments[(row["scenario_id"], variant)]
            target["fallback_action"] = position % action_count
            target["fallback_frame"] = (
                position // action_count
            ) % len(FALLBACK_FRAMES)

    conclusion_order = sorted(
        items,
        key=lambda item: hashlib.sha256(
            f"conclusion-balance:{item[0]['scenario_id']}:{item[1]}".encode()
        ).digest(),
    )
    for position, (row, variant) in enumerate(conclusion_order):
        assignments[(row["scenario_id"], variant)]["conclusion_frame"] = (
            position % len(CONCLUSION_FRAMES)
        )
    return assignments


def _fallback_key(value: str) -> str:
    return hashlib.sha256(value.rstrip(".").encode()).hexdigest()[:10]


def _intent_for_subject(intent: str, subject: str) -> str:
    """Attach a subject without producing ``revise for clarity for X``."""
    if template := _INTENT_SUBJECT_TEMPLATES.get(intent):
        return template.format(subject=subject)
    parts = intent.split(maxsplit=1)
    if len(parts) == 2 and parts[1].startswith(
        ("for ", "into ", "against ", "with ", "without ", "through ")
    ):
        return f"{parts[0]} {subject} {parts[1]}"
    return f"{intent} for {subject}"


def _render_final(
    row: dict[str, Any], variant: int, surface: dict[str, int]
) -> str:
    payload = json.loads(row["semantic_payload"])
    intent = _intent(payload, row["family"])
    subject = payload["subject"]
    subject_cap = subject[:1].upper() + subject[1:]
    constraint = row["constraint"].rstrip(".")
    outcome = row["desired_outcome"].rstrip(".")
    family = row["family"]
    opening = FAMILY_OPENINGS[family][surface["opening"]]
    action_frame = FAMILY_ACTION_FRAMES[family][surface["action_frame"]].replace(
        "{intent} for {subject}", "{intent_with_subject}"
    )
    action_sentence = action_frame.format(
        intent=intent,
        intent_cap=_upper_first(intent),
        intent_with_subject=_intent_for_subject(intent, subject),
        intent_with_subject_cap=_upper_first(
            _intent_for_subject(intent, subject)
        ),
        subject=subject,
        subject_cap=subject_cap,
    )
    constraint_sentence = FAMILY_CONSTRAINT_FRAMES[family][
        surface["constraint_frame"]
    ].format(constraint=constraint)
    action = fallback_actions(row["fallback"])[surface["fallback_action"]]
    conclusion = CONCLUSION_FRAMES[surface["conclusion_frame"]].format(
        outcome=outcome,
        outcome_lower=_lower_first(outcome),
    )
    fallback = FALLBACK_FRAMES[surface["fallback_frame"]].format(action=action)
    components = {
        "opening": opening,
        "action": action_sentence,
        "constraint": constraint_sentence,
        "conclusion": conclusion,
        "fallback": fallback,
    }
    selected = " ".join(
        components[component] for component in RESPONSE_ORDERS[surface["order"]]
    )
    return correct_indefinite_articles(selected)


def _render_messages(
    row: dict[str, Any], variant: int, surface: dict[str, int]
) -> list[dict[str, str]]:
    payload = json.loads(row["semantic_payload"])
    intent = _intent(payload, row["family"])
    subject = payload["subject"]
    trigger = row["trigger"]
    if variant % 2 == 0:
        request_frame = _PROMPT_REQUESTS[
            _stable_index(f"prompt:{row['scenario_id']}:{variant}", len(_PROMPT_REQUESTS))
        ].replace("{intent} for {subject}", "{intent_with_subject}")
        request = request_frame.format(
            intent=intent,
            intent_with_subject=_intent_for_subject(intent, subject),
            subject=subject,
        )
        prompt = f"{trigger} {request}"
        return [
            {"role": "user", "content": correct_indefinite_articles(prompt)},
            {
                "role": "assistant",
                "content": _render_final(row, variant, surface),
            },
        ]

    acknowledgement = _ACKNOWLEDGEMENTS[
        _stable_index(f"ack:{row['scenario_id']}:{variant}", len(_ACKNOWLEDGEMENTS))
    ]
    chat_opening = _CHAT_OPENERS[
        _stable_index(
            f"chat-opening:{row['scenario_id']}:{variant}", len(_CHAT_OPENERS)
        )
    ].format(subject=subject, state=row["state"].rstrip("."))
    context = payload["domain_context"].rstrip(".")
    follow_up_frame = _FOLLOW_UPS[
        _stable_index(f"follow-up:{row['scenario_id']}:{variant}", len(_FOLLOW_UPS))
    ].replace("{intent} for {subject}", "{intent_with_subject}")
    follow_up = follow_up_frame.format(
        context=_lower_first(context),
        intent=intent,
        intent_with_subject=_intent_for_subject(intent, subject),
        subject=subject,
    )
    return [
        {
            "role": "user",
            "content": correct_indefinite_articles(chat_opening),
        },
        {"role": "assistant", "content": acknowledgement},
        {"role": "user", "content": correct_indefinite_articles(follow_up)},
        {"role": "assistant", "content": _render_final(row, variant, surface)},
    ]


def _render_transcript(messages: list[dict[str, str]]) -> str:
    labels = {"user": "User", "assistant": "Assistant"}
    return "\n".join(
        f"{labels[message['role']]}: {message['content']}" for message in messages
    )


def _conversation_rows(
    scenarios: list[dict[str, Any]], variants_per_scenario: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    assignments = _surface_assignments(scenarios, variants_per_scenario)
    for scenario in scenarios:
        for variant in range(variants_per_scenario):
            surface = assignments[(scenario["scenario_id"], variant)]
            messages = _render_messages(scenario, variant, surface)
            rendered = _render_transcript(messages)
            mode = "instruct" if len(messages) == 2 else "chat"
            payload = json.loads(scenario["semantic_payload"])
            fallback_key = _fallback_key(scenario["fallback"])
            fallback_action = fallback_actions(scenario["fallback"])[
                surface["fallback_action"]
            ]
            suffix = hashlib.sha256(
                f"{scenario['scenario_id']}:{variant}:{rendered}".encode()
            ).hexdigest()[:20]
            answer = {
                "scenario_id": scenario["scenario_id"],
                "family": scenario["family"],
                "domain": scenario["domain"],
                "intent": scenario["intent"],
                "risk_level": scenario["risk_level"],
                "split": scenario["split"],
                "state": scenario["state"],
                "constraint": scenario["constraint"],
                "desired_outcome": scenario["desired_outcome"],
                "fallback": scenario["fallback"],
                "subject": payload["subject"],
                "surface_intent": _intent(payload, scenario["family"]),
                "domain_context": payload["domain_context"],
                "fallback_surface": fallback_action,
                "response_contract": scenario["response_contract"],
                "variant": variant,
                "mode": mode,
                "response_opening_id": (
                    f"{scenario['family']}:opening-{surface['opening']:02d}"
                ),
                "response_action_id": (
                    f"{scenario['family']}:action-{surface['action_frame']:02d}"
                ),
                "response_constraint_id": (
                    f"{scenario['family']}:constraint-"
                    f"{surface['constraint_frame']:02d}"
                ),
                "response_order_id": f"order-{surface['order']:02d}",
                "response_structure_id": (
                    f"{scenario['family']}:action-{surface['action_frame']:02d}:"
                    f"constraint-{surface['constraint_frame']:02d}:"
                    f"order-{surface['order']:02d}"
                ),
                "response_surface_pattern_id": (
                    f"{scenario['family']}:opening-{surface['opening']:02d}:"
                    f"action-{surface['action_frame']:02d}:"
                    f"constraint-{surface['constraint_frame']:02d}:"
                    f"order-{surface['order']:02d}"
                ),
                "fallback_action_id": (
                    f"fallback-{fallback_key}:action-{surface['fallback_action']:02d}"
                ),
                "fallback_surface_id": (
                    f"fallback-{fallback_key}:action-{surface['fallback_action']:02d}:"
                    f"frame-{surface['fallback_frame']:02d}"
                ),
                "conclusion_surface_id": (
                    f"conclusion-{surface['conclusion_frame']:02d}"
                ),
                "model_generated_dialogue": False,
            }
            rows.append(
                {
                    "example_id": f"post-training:{suffix}",
                    "task": scenario["family"],
                    "mode": mode,
                    "difficulty": (
                        "hard" if scenario["risk_level"] == "high" else "medium"
                    ),
                    "dataset_id": DATASET_ID,
                    "domain": scenario["domain"],
                    "language": "en",
                    "split": scenario["split"],
                    "messages": messages,
                    "prompt": messages[0]["content"],
                    "response": messages[-1]["content"],
                    "rendered_text": rendered,
                    "source_keys": [scenario["scenario_id"]],
                    "evidence": [],
                    "answer_json": json.dumps(answer, sort_keys=True),
                    "source": DATASET_SOURCE,
                    "source_urls": [],
                    "license": DATASET_LICENSE,
                    "version": "1.0.0",
                }
            )
    return sorted(rows, key=lambda row: row["example_id"])


def _tokens(value: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD.finditer(value)]


def _p95(values: list[int]) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * 0.95))
    return float(ordered[index])


def _length_statistics(values: list[str]) -> dict[str, float | int]:
    lengths = [len(_tokens(value)) for value in values]
    return {
        "items": len(lengths),
        "minimum_tokens": min(lengths),
        "mean_tokens": round(statistics.fmean(lengths), 3),
        "median_tokens": round(statistics.median(lengths), 3),
        "p95_tokens": _p95(lengths),
        "maximum_tokens": max(lengths),
    }


def _mattr(tokens: list[str], window: int = 100) -> float:
    """Return moving-average type-token ratio over fixed lexical windows."""
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    counts = Counter(tokens[:window])
    total = len(counts) / window
    windows = 1
    for index in range(window, len(tokens)):
        outgoing = tokens[index - window]
        counts[outgoing] -= 1
        if counts[outgoing] == 0:
            del counts[outgoing]
        counts[tokens[index]] += 1
        total += len(counts) / window
        windows += 1
    return total / windows


def _ngram_statistics(messages: list[str], size: int) -> dict[str, Any]:
    counts: Counter[tuple[str, ...]] = Counter()
    message_counts: Counter[tuple[str, ...]] = Counter()
    for message in messages:
        tokens = _tokens(message)
        grams = [
            tuple(tokens[index : index + size])
            for index in range(len(tokens) - size + 1)
        ]
        counts.update(grams)
        message_counts.update(set(grams))
    windows = sum(counts.values())
    distinct = len(counts)
    singleton_ngrams = sum(value == 1 for value in counts.values())
    most_common = counts.most_common(10)
    return {
        "window_tokens": size,
        "windows": windows,
        "distinct_ngrams": distinct,
        "distinct_ngram_ratio": round(distinct / windows, 6),
        "singleton_ngrams": singleton_ngrams,
        "singleton_distinct_ratio": round(singleton_ngrams / distinct, 6),
        "singleton_window_ratio": round(singleton_ngrams / windows, 6),
        "maximum_occurrences": max(counts.values(), default=0),
        "maximum_message_coverage": round(
            max(message_counts.values(), default=0) / len(messages), 6
        )
        if messages
        else 0.0,
        "top_repeated_ngrams": [
            {
                "text": " ".join(gram),
                "occurrences": occurrences,
                "message_count": message_counts[gram],
                "message_rate": round(message_counts[gram] / len(messages), 6)
                if messages
                else 0.0,
            }
            for gram, occurrences in most_common
        ],
    }


def _text_statistics(values: list[str]) -> dict[str, Any]:
    tokens = [token for value in values for token in _tokens(value)]
    return {
        "items": len(values),
        "exact_unique_items": len(set(values)),
        "exact_uniqueness_ratio": round(len(set(values)) / len(values), 6)
        if values
        else 0.0,
        "length": _length_statistics(values),
        "lexical": {
            "word_occurrences": len(tokens),
            "observed_vocabulary": len(set(tokens)),
            "raw_type_token_ratio": round(len(set(tokens)) / len(tokens), 6)
            if tokens
            else 0.0,
            "mattr_100": round(_mattr(tokens, 100), 6),
        },
        "four_grams": _ngram_statistics(values, 4),
        "eight_grams": _ngram_statistics(values, 8),
    }


def _formulation_statistics(
    identifiers: list[str], *, semantic_values: list[str] | None = None
) -> dict[str, Any]:
    counts = Counter(identifiers)
    maximum = max(counts.values(), default=0)
    share = maximum / len(identifiers) if identifiers else 0.0
    if share >= _MAX_SURFACE_FORMULATION_SHARE:
        raise ValueError(
            "surface formulation reaches the five-percent ceiling: "
            f"{share:.3%}"
        )
    result: dict[str, Any] = {
        "formulations": len(counts),
        "maximum_examples_per_formulation": maximum,
        "maximum_formulation_share": round(share, 6),
        "strict_share_limit": _MAX_SURFACE_FORMULATION_SHARE,
        "formulation_counts": dict(sorted(counts.items())),
    }
    if semantic_values is not None:
        result["semantic_value_counts"] = dict(
            sorted(Counter(semantic_values).items())
        )
        result["semantic_values_are_not_surface_formulations"] = True
    return result


_MASKED_RESPONSE_FIELDS = (
    ("subject", "subject"),
    ("surface_intent", "intent"),
    ("state", "state"),
    ("constraint", "constraint"),
    ("desired_outcome", "desired_outcome"),
    ("fallback", "fallback"),
    ("fallback_surface", "fallback_surface"),
    ("domain_context", "domain_context"),
)


def _masked_response(response: str, answer: dict[str, Any]) -> str:
    replacements = [
        (str(answer[source]).strip().rstrip("."), f"<{target}>")
        for source, target in _MASKED_RESPONSE_FIELDS
        if str(answer.get(source, "")).strip().rstrip(".")
    ]
    masked = response
    for value, placeholder in sorted(replacements, key=lambda item: -len(item[0])):
        masked = re.sub(re.escape(value), placeholder, masked, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", masked).strip().lower()


def _masked_diversity(
    responses: list[str], answers: list[dict[str, Any]]
) -> dict[str, Any]:
    skeletons = [
        _masked_response(response, answer)
        for response, answer in zip(responses, answers, strict=True)
    ]
    counts = Counter(skeletons)
    maximum = max(counts.values(), default=0)
    maximum_share = maximum / len(skeletons) if skeletons else 0.0
    if maximum_share >= _MAX_SURFACE_FORMULATION_SHARE:
        raise ValueError(
            "masked response skeleton reaches the five-percent ceiling: "
            f"{maximum_share:.3%}"
        )
    return {
        "masked_fields": [target for _, target in _MASKED_RESPONSE_FIELDS],
        "skeletons": len(skeletons),
        "distinct_skeletons": len(counts),
        "exact_skeleton_uniqueness_ratio": round(
            len(counts) / len(skeletons), 6
        )
        if skeletons
        else 0.0,
        "maximum_examples_per_skeleton": maximum,
        "maximum_skeleton_share": round(maximum_share, 6),
        "strict_share_limit": _MAX_SURFACE_FORMULATION_SHARE,
        "four_gram_stats": _ngram_statistics(skeletons, 4),
        "eight_gram_stats": _ngram_statistics(skeletons, 8),
        "top_repeated_skeletons": [
            {"text": text, "examples": count, "share": round(count / len(skeletons), 6)}
            for text, count in counts.most_common(10)
        ],
    }


def _audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rendered = [row["rendered_text"] for row in rows]
    responses = [row["response"] for row in rows]
    messages = [message["content"] for row in rows for message in row["messages"]]
    user_prompts = [
        message["content"]
        for row in rows
        for message in row["messages"]
        if message["role"] == "user"
    ]
    assistant_messages = [
        message["content"]
        for row in rows
        for message in row["messages"]
        if message["role"] == "assistant"
    ]
    assistant_meta_hits = [
        {"phrase": phrase, "text": message}
        for message in assistant_messages
        for phrase in _FORBIDDEN_ASSISTANT_META_PHRASES
        if phrase in message.lower()
    ]
    user_meta_hits = [
        {"phrase": phrase, "text": message}
        for message in user_prompts
        for phrase in _FORBIDDEN_USER_META_PHRASES
        if phrase in message.lower()
    ]
    if assistant_meta_hits:
        raise ValueError(
            "assistant text describes how to answer instead of answering directly: "
            f"{assistant_meta_hits[0]}"
        )
    if user_meta_hits:
        raise ValueError(
            "user text asks for meta-response construction instead of the task: "
            f"{user_meta_hits[0]}"
        )
    train_cards: set[str] = set()
    validation_cards: set[str] = set()
    split_groups: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    exact_anchor_rows = 0
    answers: list[dict[str, Any]] = []
    first_user_messages: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        answer = json.loads(row["answer_json"])
        answers.append(answer)
        first_user_messages[answer["scenario_id"]][answer["mode"]] = next(
            message["content"]
            for message in row["messages"]
            if message["role"] == "user"
        )
        target = validation_cards if row["split"] == "validation" else train_cards
        target.add(answer["scenario_id"])
        split_groups[row["split"]].add(
            (answer["family"], answer["domain"], answer["intent"])
        )
        normalized = row["rendered_text"].lower()
        if all(
            normalized.count(answer[field].rstrip(".").lower()) == 1
            for field in ("state", "constraint")
        ):
            exact_anchor_rows += 1
    card_overlap = train_cards & validation_cards
    group_overlap = split_groups["train"] & split_groups["validation"]
    if card_overlap:
        raise ValueError("source scenarios leak across post-training splits")
    if group_overlap:
        raise ValueError("family/domain/intent groups leak across post-training splits")
    if len(set(rendered)) != len(rendered):
        raise ValueError("duplicate post-training conversations")
    if any(row["license"] != DATASET_LICENSE for row in rows):
        raise ValueError("post-training license mismatch")
    if exact_anchor_rows != len(rows):
        raise ValueError("state and constraint must each appear exactly once")

    paired_prompt_surfaces = [
        modes
        for modes in first_user_messages.values()
        if "instruct" in modes and "chat" in modes
    ]
    exact_first_user_matches = sum(
        modes["instruct"] == modes["chat"] for modes in paired_prompt_surfaces
    )
    chat_opener_prefix_matches = sum(
        modes["instruct"].startswith(modes["chat"])
        for modes in paired_prompt_surfaces
    )
    if exact_first_user_matches or chat_opener_prefix_matches:
        raise ValueError(
            "paired chat and instruct prompts must use independent surface openings"
        )

    unique_final_response_ratio = len(set(responses)) / len(responses)
    if unique_final_response_ratio < 0.95:
        raise ValueError(
            "post-training final responses are not sufficiently individualized: "
            f"{unique_final_response_ratio:.3f}"
        )

    surface_patterns = Counter(
        answer["response_surface_pattern_id"] for answer in answers
    )
    structures = Counter(answer["response_structure_id"] for answer in answers)
    openings = Counter(answer["response_opening_id"] for answer in answers)
    actions = Counter(answer["response_action_id"] for answer in answers)
    constraints = Counter(answer["response_constraint_id"] for answer in answers)
    orders = Counter(answer["response_order_id"] for answer in answers)
    final_response_stats = _text_statistics(responses)
    masked_response_diversity = _masked_diversity(responses, answers)
    maximum_final_phrase_share = masked_response_diversity["eight_gram_stats"][
        "maximum_message_coverage"
    ]
    if maximum_final_phrase_share >= _MAX_SURFACE_FORMULATION_SHARE:
        raise ValueError(
            "a masked final-response eight-token phrase reaches the "
            "five-percent ceiling: "
            f"{maximum_final_phrase_share:.3%}"
        )
    all_message_tokens = [token for message in messages for token in _tokens(message)]
    vocabulary = set(all_message_tokens)
    family_metrics: dict[str, dict[str, Any]] = {}
    for family in sorted({row["task"] for row in rows}):
        family_rows = [row for row in rows if row["task"] == family]
        family_answers = [
            json.loads(row["answer_json"])["scenario_id"] for row in family_rows
        ]
        family_responses = [row["response"] for row in family_rows]
        family_metrics[family] = {
            "examples": len(family_rows),
            "source_scenarios": len(set(family_answers)),
            "unique_final_response_ratio": round(
                len(set(family_responses)) / len(family_responses), 6
            ),
            "mean_response_tokens": _length_statistics(family_responses)[
                "mean_tokens"
            ],
        }
    family_source_scenarios: dict[str, set[str]] = defaultdict(set)
    for answer in answers:
        family_source_scenarios[answer["family"]].add(answer["scenario_id"])

    return {
        "rows": len(rows),
        "source_scenarios": len(train_cards | validation_cards),
        "split_holdout_units": ["scenario_id", "family+domain+intent"],
        "source_scenario_split_overlap": len(card_overlap),
        "semantic_group_split_overlap": len(group_overlap),
        "train_semantic_groups": len(split_groups["train"]),
        "validation_semantic_groups": len(split_groups["validation"]),
        "split_example_counts": dict(
            sorted(Counter(row["split"] for row in rows).items())
        ),
        "family_example_counts": dict(
            sorted(Counter(row["task"] for row in rows).items())
        ),
        "family_source_scenario_counts": dict(
            sorted(
                (family, len(scenarios))
                for family, scenarios in family_source_scenarios.items()
            )
        ),
        "mode_example_counts": dict(
            sorted(Counter(row["mode"] for row in rows).items())
        ),
        "exact_conversation_uniqueness_ratio": len(set(rendered)) / len(rendered),
        "exact_final_response_uniqueness_ratio": unique_final_response_ratio,
        "duplicate_final_response_rows": len(responses) - len(set(responses)),
        "model_generated_dialogue_rows": sum(
            bool(answer["model_generated_dialogue"]) for answer in answers
        ),
        "single_state_and_constraint_ratio": exact_anchor_rows / len(rows),
        "paired_prompt_surface_stats": {
            "paired_scenarios": len(paired_prompt_surfaces),
            "exact_first_user_message_matches": exact_first_user_matches,
            "chat_opener_is_instruct_prefix": chat_opener_prefix_matches,
            "shared_semantics": [
                "scenario_id",
                "state",
                "constraint",
                "desired_outcome",
            ],
        },
        "surface_pattern_stats": {
            "possible_opening_structure_pairs": (
                sum(
                    len(FAMILY_OPENINGS[family])
                    * len(FAMILY_ACTION_FRAMES[family])
                    * len(FAMILY_CONSTRAINT_FRAMES[family])
                    * len(RESPONSE_ORDERS)
                    for family in sorted(_INTENT_FIELD)
                )
            ),
            "observed_opening_structure_pairs": len(surface_patterns),
            "maximum_examples_per_pair": max(surface_patterns.values()),
            "maximum_pair_share": round(
                max(surface_patterns.values()) / len(rows), 6
            ),
            "opening_counts": dict(sorted(openings.items())),
            "structure_counts": dict(sorted(structures.items())),
            "action_counts": dict(sorted(actions.items())),
            "constraint_counts": dict(sorted(constraints.items())),
            "order_counts": dict(sorted(orders.items())),
        },
        "body_surface_stats": {
            "openings": _formulation_statistics(
                [answer["response_opening_id"] for answer in answers]
            ),
            "actions": _formulation_statistics(
                [answer["response_action_id"] for answer in answers]
            ),
            "constraints": _formulation_statistics(
                [answer["response_constraint_id"] for answer in answers]
            ),
            "orders": {
                "patterns": len(orders),
                "counts": dict(sorted(orders.items())),
            },
            "masked_final_eight_token_phrase_ceiling": {
                "maximum_message_coverage": maximum_final_phrase_share,
                "strict_share_limit": _MAX_SURFACE_FORMULATION_SHARE,
                "note": (
                    "Source subjects, intents, states, constraints, outcomes and "
                    "fallback semantics are masked before measuring prose templates."
                ),
            },
        },
        "fallback_surface_stats": _formulation_statistics(
            [answer["fallback_action_id"] for answer in answers],
            semantic_values=[answer["fallback"] for answer in answers],
        ),
        "conclusion_surface_stats": _formulation_statistics(
            [answer["conclusion_surface_id"] for answer in answers]
        ),
        "masked_response_diversity": masked_response_diversity,
        "role_text_stats": {
            "user_prompts": _text_statistics(user_prompts),
            "assistant_messages": _text_statistics(assistant_messages),
            "final_responses": final_response_stats,
        },
        "natural_language_gate": {
            "assistant_meta_instruction_hits": len(assistant_meta_hits),
            "user_meta_request_hits": len(user_meta_hits),
            "forbidden_assistant_phrases": list(
                _FORBIDDEN_ASSISTANT_META_PHRASES
            ),
            "forbidden_user_phrases": list(_FORBIDDEN_USER_META_PHRASES),
        },
        "message_length_stats": _length_statistics(messages),
        "response_length_stats": _length_statistics(responses),
        "lexical_stats": {
            "word_occurrences": len(all_message_tokens),
            "observed_vocabulary": len(vocabulary),
            "raw_type_token_ratio": round(
                len(vocabulary) / len(all_message_tokens), 6
            ),
            "mattr_100": round(_mattr(all_message_tokens, 100), 6),
        },
        "four_gram_stats": _ngram_statistics(messages, 4),
        "eight_gram_stats": _ngram_statistics(messages, 8),
        "family_metrics": family_metrics,
    }


def _review_sample(
    rows: list[dict[str, Any]], *, review_scenarios: int, seed: int
) -> list[dict[str, str]]:
    families = sorted({row["task"] for row in rows})
    if review_scenarios < len(families) or review_scenarios % len(families):
        raise ValueError(
            "review_scenarios must be a positive multiple of family count"
        )
    quota = review_scenarios // len(families)
    scenario_rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    scenario_answers: dict[str, dict[str, Any]] = {}
    grouped: dict[
        str, dict[tuple[str, str], dict[str, list[str]]]
    ] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        answer = json.loads(row["answer_json"])
        scenario_id = answer["scenario_id"]
        scenario_rows[scenario_id][row["mode"]].append(row)
        scenario_answers[scenario_id] = answer
    for scenario_id, modes in scenario_rows.items():
        answer = scenario_answers[scenario_id]
        if set(modes) != {"chat", "instruct"}:
            raise ValueError(
                f"review scenario {scenario_id} does not provide both chat and instruct"
            )
        grouped[answer["family"]][
            (answer["risk_level"], answer["split"])
        ][answer["domain"]].append(scenario_id)

    selected: list[dict[str, str]] = []
    for family in families:
        domain_strata = grouped[family]
        strata: dict[tuple[str, str], list[str]] = {}
        for stratum, domains in domain_strata.items():
            for values in domains.values():
                values.sort(
                    key=lambda scenario_id: hashlib.sha256(
                        f"review:{seed}:{scenario_id}".encode()
                    ).digest()
                )
            domain_order = sorted(domains)
            strata[stratum] = [
                domains[domain][position]
                for position in range(max(len(values) for values in domains.values()))
                for domain in domain_order
                if position < len(domains[domain])
            ]
        ordered_strata = sorted(strata)
        positions = Counter()
        family_scenarios: list[str] = []
        while len(family_scenarios) < quota:
            made_progress = False
            for stratum in ordered_strata:
                position = positions[stratum]
                if position < len(strata[stratum]):
                    family_scenarios.append(strata[stratum][position])
                    positions[stratum] += 1
                    made_progress = True
                    if len(family_scenarios) == quota:
                        break
            if not made_progress:
                raise ValueError(f"insufficient review candidates for {family}")
        for scenario_id in family_scenarios:
            answer = scenario_answers[scenario_id]
            for mode in ("instruct", "chat"):
                candidates = sorted(
                    scenario_rows[scenario_id][mode],
                    key=lambda row: hashlib.sha256(
                        f"review-mode:{seed}:{row['example_id']}".encode()
                    ).digest(),
                )
                row = candidates[0]
                selected.append(
                    {
                        "review_unit_id": scenario_id,
                        "example_id": row["example_id"],
                        "scenario_id": scenario_id,
                        "mode": mode,
                        "family": row["task"],
                        "domain": row["domain"],
                        "risk_level": answer["risk_level"],
                        "split": row["split"],
                        "prompt": row["prompt"],
                        "response": row["response"],
                        "review_status": "pending",
                        "semantic_accuracy": "",
                        "constraint_following": "",
                        "language_quality": "",
                        "individualization": "",
                        "safety": "",
                        "reviewer": "",
                        "reviewed_at_utc": "",
                        "reviewer_notes": "",
                    }
                )
    return selected


def build_post_training_corpus(
    scenarios_path: Path,
    output_root: Path,
    *,
    variants_per_scenario: int = 2,
    review_scenarios: int = 70,
    seed: int = 42,
) -> dict[str, Any]:
    if variants_per_scenario < 1:
        raise ValueError("variants_per_scenario must be positive")
    scenarios = pq.read_table(scenarios_path).to_pylist()
    rows = _conversation_rows(scenarios, variants_per_scenario)
    audit = _audit(rows)
    review = _review_sample(
        rows, review_scenarios=review_scenarios, seed=seed
    )

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    conversations_path = temporary / "conversations.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=INSTRUCTION_SCHEMA),
        conversations_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    audit_path = temporary / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    review_path = temporary / "human_review.csv"
    with review_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    manifest = {
        "format": "complexity-post-training-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "license": DATASET_LICENSE,
        "source": DATASET_SOURCE,
        "input": {"path": str(scenarios_path), "sha256": file_sha256(scenarios_path)},
        "variants_per_scenario": variants_per_scenario,
        "audit": audit,
        "human_review": {
            "rows": len(review),
            "source_scenarios": len({row["scenario_id"] for row in review}),
            "sample_fraction_of_source_scenarios": round(
                len({row["scenario_id"] for row in review})
                / audit["source_scenarios"],
                6,
            ),
            "modes_per_source_scenario": ["instruct", "chat"],
            "status": "pending",
            "strata": ["family", "risk_level", "split", "domain"],
            "sampling_scope": (
                "deterministic stratified quality-control sample; not a simple "
                "random estimate of corpus-wide defect prevalence"
            ),
            "required_before_training": True,
        },
        "training_ready": False,
        "release_ready": False,
    }
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return {
        **manifest,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in (
                output_root / "conversations.parquet",
                output_root / "audit.json",
                output_root / "human_review.csv",
                output_root / "manifest.json",
            )
        },
    }


def audit_human_review(review_path: Path) -> dict[str, Any]:
    """Evaluate a completed stratified review sheet without changing artifacts."""
    with review_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("human review sheet is empty")
    required = {
        "review_unit_id",
        "example_id",
        "scenario_id",
        "mode",
        "family",
        "domain",
        "risk_level",
        "split",
        "review_status",
        *REVIEW_GRADES,
        "reviewer",
        "reviewed_at_utc",
        "reviewer_notes",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"human review sheet is missing columns: {sorted(missing)}")
    statuses = [row["review_status"].strip().lower() for row in rows]
    invalid_statuses = sorted(set(statuses) - _REVIEW_STATUSES)
    if invalid_statuses:
        raise ValueError(f"invalid review statuses: {invalid_statuses}")
    for grade in REVIEW_GRADES:
        values = {row[grade].strip().lower() for row in rows}
        invalid_grades = sorted(values - _REVIEW_GRADE_VALUES)
        if invalid_grades:
            raise ValueError(f"invalid {grade} grades: {invalid_grades}")
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("human review contains duplicate example_id values")

    scenario_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["review_unit_id"] != row["scenario_id"]:
            raise ValueError("review_unit_id must equal scenario_id")
        scenario_rows[row["scenario_id"]].append(row)
    incomplete_modes = {
        scenario_id: sorted(row["mode"] for row in values)
        for scenario_id, values in scenario_rows.items()
        if Counter(row["mode"] for row in values)
        != Counter({"instruct": 1, "chat": 1})
    }
    if incomplete_modes:
        raise ValueError(
            "each reviewed scenario must contain one instruct and one chat row: "
            f"{incomplete_modes}"
        )

    status_counts = Counter(statuses)
    grade_counts = {
        grade: dict(
            sorted(Counter(row[grade].strip().lower() for row in rows).items())
        )
        for grade in REVIEW_GRADES
    }
    approved = all(status == "approved" for status in statuses)
    grades_pass = all(
        row[grade].strip().lower() == "pass"
        for row in rows
        for grade in REVIEW_GRADES
    )
    provenance_complete = all(status != "pending" for status in statuses)
    for row in rows:
        status = row["review_status"].strip().lower()
        if status == "pending":
            continue
        reviewer = row["reviewer"].strip()
        reviewed_at = row["reviewed_at_utc"].strip()
        if not reviewer or not reviewed_at:
            provenance_complete = False
            continue
        try:
            timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError:
            provenance_complete = False
            continue
        if timestamp.tzinfo is None:
            provenance_complete = False

    def scenario_axis_counts(field: str) -> dict[str, int]:
        return dict(
            sorted(
                Counter(
                    values[0][field]
                    for values in scenario_rows.values()
                ).items()
            )
        )

    def row_failed(row: dict[str, str]) -> bool:
        return row["review_status"].strip().lower() == "rejected" or any(
            row[grade].strip().lower() == "fail" for grade in REVIEW_GRADES
        )

    failed_rows = [row for row in rows if row_failed(row)]
    failed_scenarios = {
        row["scenario_id"] for row in failed_rows
    }
    ready = approved and grades_pass and provenance_complete
    zero_failure_bound = None
    if ready:
        sample_size = len(scenario_rows)
        zero_failure_bound = {
            "confidence": 0.95,
            "scenario_sample_size": sample_size,
            "upper_defect_rate_if_iid_random": round(
                1 - math.pow(0.05, 1 / sample_size), 6
            ),
            "caveat": (
                "descriptive sensitivity bound only; this review is stratified "
                "rather than a simple iid random sample"
            ),
        }
    return {
        "rows": len(rows),
        "source_scenarios": len(scenario_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "grade_counts": grade_counts,
        "coverage": {
            "family_source_scenarios": scenario_axis_counts("family"),
            "risk_source_scenarios": scenario_axis_counts("risk_level"),
            "split_source_scenarios": scenario_axis_counts("split"),
            "domain_source_scenarios": scenario_axis_counts("domain"),
            "mode_rows": dict(sorted(Counter(row["mode"] for row in rows).items())),
        },
        "failed_rows": len(failed_rows),
        "failed_scenarios": len(failed_scenarios),
        "failure_counts": {
            "family_rows": dict(
                sorted(Counter(row["family"] for row in failed_rows).items())
            ),
            "risk_rows": dict(
                sorted(Counter(row["risk_level"] for row in failed_rows).items())
            ),
            "split_rows": dict(
                sorted(Counter(row["split"] for row in failed_rows).items())
            ),
            "mode_rows": dict(
                sorted(Counter(row["mode"] for row in failed_rows).items())
            ),
        },
        "reviewer_counts": dict(
            sorted(
                Counter(
                    row["reviewer"].strip()
                    for row in rows
                    if row["reviewer"].strip()
                ).items()
            )
        ),
        "review_provenance_complete": provenance_complete,
        "zero_failure_bound": zero_failure_bound,
        "training_ready": ready,
    }
