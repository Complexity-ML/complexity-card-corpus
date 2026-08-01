from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .build import file_sha256
from .chat_template import (
    CHAT_TEMPLATE_ID,
    chat_template_contract,
    render_system_prefix,
    render_user_turn,
)
from .english_morphology import correct_indefinite_articles
from .tokenize import directory_sha256, load_encoding
from .training_cards import TrainingCards, deal_training_cards


INSTRUCTION_SCHEMA = pa.schema(
    [
        ("example_id", pa.string()),
        ("task", pa.string()),
        ("mode", pa.string()),
        ("difficulty", pa.string()),
        ("dataset_id", pa.string()),
        ("domain", pa.string()),
        ("language", pa.string()),
        ("split", pa.string()),
        (
            "messages",
            pa.list_(
                pa.struct(
                    [
                        ("role", pa.string()),
                        ("content", pa.string()),
                    ]
                )
            ),
        ),
        ("prompt", pa.string()),
        ("response", pa.string()),
        ("rendered_text", pa.string()),
        ("source_keys", pa.list_(pa.string())),
        ("evidence", pa.list_(pa.string())),
        ("answer_json", pa.string()),
        ("source", pa.string()),
        ("source_urls", pa.list_(pa.string())),
        ("license", pa.string()),
        ("version", pa.string()),
    ]
)

PROJECTED_SFT_SCHEMA = pa.schema(
    [
        ("example_id", pa.string()),
        ("task", pa.string()),
        ("mode", pa.string()),
        ("difficulty", pa.string()),
        ("domain", pa.string()),
        ("language", pa.string()),
        ("split", pa.string()),
        ("prompt", pa.string()),
        ("response", pa.string()),
        ("structure_signature", pa.string()),
        ("source_representation", pa.string()),
        ("source", pa.string()),
        ("license", pa.string()),
        ("version", pa.string()),
    ]
)

TOKEN_DTYPE = np.dtype("<u4")
LABEL_DTYPE = np.dtype("<i4")
IGNORE_INDEX = -100

_CARD_SECTION = re.compile(
    r"(?m)^(SITUATION|DATA|RULE|GOAL) CARD\s*$"
)
_HAND_PREFIX = re.compile(
    r"^(?:For\s+hand|Hand)\s+[A-Za-z0-9]+\s*(?:—|:|-)\s*",
    re.IGNORECASE,
)


def _stable_index(value: str, size: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % size


def _pick(value: str, choices: tuple[str, ...]) -> str:
    return choices[_stable_index(value, len(choices))]


def _merged_user_content(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        message["content"].strip()
        for message in messages
        if message["role"] == "user"
    )


def _card_sections(messages: list[dict[str, str]]) -> dict[str, str] | None:
    """Extract authored cards without exposing their storage labels."""

    merged = _merged_user_content(messages)
    parts = _CARD_SECTION.split(merged)
    sections = {
        parts[index].lower(): parts[index + 1].strip()
        for index in range(1, len(parts) - 1, 2)
    }
    required = {"situation", "data", "rule", "goal"}
    if not sections:
        return None
    if set(sections) != required:
        raise ValueError(
            "SFT card hand must contain situation, data, rule, and goal"
        )
    return sections


def _render_natural_instruction(
    messages: list[dict[str, str]],
    example_id: str,
    cards: TrainingCards | None = None,
) -> str:
    sections = _card_sections(messages)
    if sections is None:
        return _merged_user_content(messages)
    cards = cards or deal_training_cards(
        task="unknown",
        mode="chat" if len(messages) > 2 else "instruct",
        example_id=example_id,
    )
    goal = sections["goal"]
    lower_goal = goal[:1].lower() + goal[1:]
    values = {**sections, "lower_goal": lower_goal}
    full = {
        "direct": "{goal}\n\nContext: {situation}\n\nUse these facts: {data}\n\nRequirement: {rule}",
        "polite": "Could you help with this request? {goal}\n\n{situation}\n\nThe available information is: {data}\n\nPlease keep this limit: {rule}",
        "compact": "{goal}\n\n{data}\n\nKeep to this condition: {rule}",
        "context_first": "{situation}\n\nGiven this context, {lower_goal}\n\nRelevant information: {data}\n\n{rule}",
        "conversational": "I need help with this: {lower_goal}\n\nHere is what happened: {situation}\n\nWhat we know: {data}\n\nPlease keep in mind that {rule}",
        "follow_up": "Following up on the context below, {lower_goal}\n\n{situation}\n\nUse this information: {data}\n\nThe remaining condition is: {rule}",
        "plain": "{goal}\n\n{situation}\n\n{data}\n\n{rule}",
    }
    rendered = full[cards.surface].format(**values)
    if cards.dialogue_state == "correction":
        rendered = (
            f"One detail in the earlier request conflicts with the record. "
            f"{sections['situation']}\n\n"
            f"{goal}\n\nUse only this recorded information: {sections['data']}\n\n"
            f"Keep this condition: {sections['rule']} Do not assume the conflict "
            "is resolved unless the information says so."
        )
    elif cards.dialogue_state == "constraint_update" and cards.surface != "compact":
        rendered = (
            f"There is one additional constraint: {sections['rule']}\n\n"
            f"{goal}\n\n{sections['situation']}\n\n"
            f"Use this information: {sections['data']}"
        )
    elif cards.dialogue_state == "clarification_resolved" and cards.surface != "compact":
        rendered = (
            f"The scope is now clear. {goal}\n\n{sections['situation']}\n\n"
            f"Relevant information: {sections['data']}\n\n"
            f"Keep this requirement: {sections['rule']}"
        )

    if cards.context_density == "minimal" and cards.uncertainty == "answerable":
        rendered = f"{goal}\n\n{sections['data']}"
    elif cards.context_density == "focused":
        rendered = f"{goal}\n\n{sections['data']}\n\n{sections['rule']}"
    if cards.noise == "secondary_detail":
        notes = (
            "The record identifier is included only for traceability.",
            "The order in which the facts were copied does not determine the answer.",
            "Formatting differences in the source do not change the stated values.",
            "An administrative label in the record is not part of the requested result.",
        )
        rendered += "\n\n" + notes[_stable_index(f"noise-note:{example_id}", len(notes))]
    return correct_indefinite_articles(rendered)


def _final_assistant_target(messages: list[dict[str, str]]) -> str:
    response = next(
        message["content"].strip()
        for message in reversed(messages)
        if message["role"] == "assistant"
    )
    return _HAND_PREFIX.sub("", response, count=1)


def _labelled_fields(text: str, labels: tuple[str, ...]) -> dict[str, str]:
    """Read authored completion fields without retaining their storage labels."""

    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(label) for label in labels) + r"):\s*"
    )
    matches = list(pattern.finditer(text))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields[match.group(1)] = text[match.end() : end].strip().rstrip(" .")
    return fields


def _sentence(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    value = value[:1].upper() + value[1:]
    if value[-1] not in ".!?":
        value += "."
    return value


def _inline_sentence(value: str) -> str:
    """Make a complete clause safe to insert after an inline lead-in."""

    value = _sentence(value)
    initial = re.match(r"[A-Za-z]+", value)
    if initial is not None:
        word = initial.group(0)
        if not (len(word) > 1 and word.isupper()):
            value = value[:1].lower() + value[1:]
    return value


def _naturalize_assistant_target(
    messages: list[dict[str, str]],
    *,
    task: str,
    cards: TrainingCards,
    example_id: str,
) -> str:
    """Project card contracts into varied, direct assistant prose for SFT.

    The authored corpus keeps explicit completion labels because they make the
    source auditable. The model-facing projection deliberately removes those
    labels so the model learns the answer rather than a single house format.
    """

    response = _final_assistant_target(messages)
    variant = _stable_index(
        f"assistant-target:{example_id}:{cards.surface}:{cards.style}", 8
    )
    if task == "explanation_learning":
        fields = _labelled_fields(response, ("Core idea", "Example", "Check"))
        if set(fields) == {"Core idea", "Example", "Check"}:
            idea = re.sub(
                r"^(?:in plain terms,\s*|the key distinction is that\s+)",
                "",
                fields["Core idea"],
                flags=re.IGNORECASE,
            )
            idea = _sentence(idea)
            example = _sentence(fields["Example"])
            check = _sentence(fields["Check"])
            templates = (
                "{idea} For example, {example} {check}",
                "In simple terms, {inline_idea} For example, {inline_example} {check}",
                "{idea} You can see this in practice: {example} To check your understanding, {inline_check}",
                "The key point is that {inline_idea} For instance, {inline_example} {check}",
            )
            return templates[variant % len(templates)].format(
                idea=idea,
                inline_idea=_inline_sentence(idea),
                example=(
                    _inline_sentence(example) if variant == 0 else example
                ),
                inline_example=_inline_sentence(example),
                check=check,
                inline_check=_inline_sentence(check),
            )
    elif task == "reasoning_verification":
        fields = _labelled_fields(response, ("Equation", "Total", "Check"))
        if set(fields) == {"Equation", "Total", "Check"}:
            check = re.sub(
                r"^(?:independently,\s*|inspect the supplied values, then note that\s*|use a second view of the values;\s*)",
                "",
                fields["Check"],
                flags=re.IGNORECASE,
            )
            templates = (
                "{equation}, so the result is {total}. As an independent check, {check}.",
                "The result is {total}: {equation}. This is consistent because {check}.",
                "Using the supplied values gives {equation}. Therefore, {total}. To verify it, {check}.",
                "{equation}. That gives {total}; checking from the other direction, {check}.",
            )
            return templates[variant % len(templates)].format(
                equation=fields["Equation"],
                total=fields["Total"],
                check=check,
            )
    elif task == "summarization_synthesis":
        fields = _labelled_fields(response, ("Decision", "Action", "Open point"))
        if set(fields) == {"Decision", "Action", "Open point"}:
            open_point = _sentence(fields["Open point"])
            templates = (
                "The decision is to {decision}. {action}. {open_point}",
                "They decided to {decision}. Next, {action}. {open_point}",
                "In summary, the decision is to {decision}; {action}. {open_point}",
                "The recorded decision is to {decision}, and {inline_action} {open_point}",
            )
            return templates[variant % len(templates)].format(
                decision=fields["Decision"],
                action=fields["Action"],
                inline_action=_inline_sentence(fields["Action"]),
                open_point=open_point,
            )
    elif task == "grounded_qa":
        direct = re.sub(
            r"^(?:Based on Source [A-Za-z0-9]+:|Source [A-Za-z0-9]+ supports this answer:|According to Source [A-Za-z0-9]+:)\s*",
            "",
            response,
        )
        direct = re.sub(
            r"^The documented answer is:\s*",
            "",
            direct,
        )
        direct = re.sub(
            r"\s+This is limited to Source [A-Za-z0-9]+\.?$",
            "",
            direct,
        )
        return direct
    elif task == "critique_revision":
        fields = _labelled_fields(response, ("Weakness", "Revision"))
        if set(fields) == {"Weakness", "Revision"}:
            weakness_text = re.sub(
                r",?\s*which makes the original difficult to verify\.?$",
                "",
                fields["Weakness"],
                flags=re.IGNORECASE,
            )
            weakness_text = re.sub(
                r"\s+The revision must stay within the recorded facts\.?$",
                "",
                weakness_text,
                flags=re.IGNORECASE,
            )
            weakness = _sentence(weakness_text)
            revision = _sentence(fields["Revision"])
            templates = (
                "{revision} This fixes the main problem because {inline_weakness}",
                "{revision} The draft previously failed because {inline_weakness}",
                "{revision}",
                "{revision} This avoids the unsupported part of the original because {inline_weakness}",
            )
            return templates[variant % len(templates)].format(
                weakness=weakness,
                inline_weakness=_inline_sentence(weakness),
                revision=revision,
            )
    elif task == "safety_uncertainty":
        match = re.fullmatch(
            r"Immediate action:\s*(.*?)\s+Boundary:\s*(.*?)\s+(Escalate\b.*)",
            response,
        )
        if match is not None:
            action = _sentence(match.group(1))
            boundary = _sentence(match.group(2))
            escalation = _sentence(match.group(3))
            templates = (
                "{action} {boundary} {escalation}",
                "First, {inline_action} {boundary} Next, {inline_escalation}",
                "The safest immediate step is clear. {action} {boundary} Then {inline_escalation}",
                "{action} {boundary} {escalation}",
            )
            return templates[variant % len(templates)].format(
                action=action,
                inline_action=_inline_sentence(action),
                boundary=boundary,
                inline_boundary=_inline_sentence(boundary),
                escalation=escalation,
                inline_escalation=_inline_sentence(escalation),
            )
    elif task == "practical_action":
        fields = _labelled_fields(response, ("Next step", "Owner", "Timing"))
        if set(fields) == {"Next step", "Owner", "Timing"}:
            step = _sentence(fields["Next step"])
            owner = _sentence(fields["Owner"])
            timing = _sentence(fields["Timing"])
            if timing.lower().startswith("before "):
                timing = "Complete this " + _inline_sentence(timing)
            templates = (
                "{step} {owner} {timing}",
                "First, {step} {timing} {owner}",
                "{timing} Before committing, {inline_step} {owner}",
                "The safest workable move is clear. {step} {owner} {timing}",
            )
            return templates[variant % len(templates)].format(
                step=step,
                inline_step=_inline_sentence(step),
                owner=owner,
                timing=timing,
            )
    elif task == "context_clarification":
        fields = _labelled_fields(response, ("Understood",))
        if fields:
            return _sentence(fields["Understood"])
    elif task == "brainstorming_creativity":
        direct = re.sub(
            r"\s+(?:Each description states.*|The three retained ideas remain feasible.*|The alternatives emphasize.*)$",
            "",
            response,
            flags=re.IGNORECASE,
        )
        return direct.strip()
    elif task == "writing_transformation":
        direct = re.sub(
            r"^(?:Support reply|Project update|Internal note|Public notice|Short brief)\s+[A-Z0-9]+:\s*",
            "",
            response,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"^(?:Meeting|Status|Update)\s+[A-Z0-9]+\s*[—:-]\s*",
            "",
            direct,
            flags=re.IGNORECASE,
        )
        fields = _labelled_fields(direct, ("Decision", "Action", "Open item"))
        if set(fields) == {"Decision", "Action", "Open item"}:
            return " ".join(_sentence(fields[name]) for name in fields)
        direct = re.sub(
            r"\bRemaining work:\s*([A-Za-z])",
            lambda match: match.group(1).upper(),
            direct,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"\bBlocker:\s*([A-Za-z])",
            lambda match: match.group(1).upper(),
            direct,
            flags=re.IGNORECASE,
        )
        return direct
    elif task in {
        "conversation_empathy",
        "extraction_classification",
    }:
        return response
    elif task == "planning_comparison":
        match = re.fullmatch(
            r"(.*?)\s+Sequence:\s*(.*?)\s+Fallback trigger:\s*(.*)",
            response,
            flags=re.DOTALL,
        )
        if match is not None:
            choice = _sentence(match.group(1))
            sequence = _sentence(match.group(2))
            fallback = _sentence(match.group(3))
            templates = (
                "{choice} Then {inline_sequence} If that path fails, {inline_fallback}",
                "{choice} {sequence} {fallback}",
                "{choice} {sequence} {fallback}",
                "{sequence} On those constraints, {inline_choice} If needed, {inline_fallback}",
            )
            return templates[variant % len(templates)].format(
                choice=choice,
                inline_choice=_inline_sentence(choice),
                sequence=sequence,
                inline_sequence=_inline_sentence(sequence),
                fallback=fallback,
                inline_fallback=_inline_sentence(fallback),
            )
        return response
    elif task == "troubleshooting":
        direct = re.sub(
            r"\bDirect check:\s*(?:confirm that\s*)?",
            "Confirm that ",
            response,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"\bRegression check:\s*(?:repeat\s*)?",
            "Afterward, repeat ",
            direct,
            flags=re.IGNORECASE,
        )
        steps = [
            match.group(1).strip()
            for match in re.finditer(
                r"(?:^|\s)\d+\.\s*(.*?)(?=\s+\d+\.\s|$)",
                direct,
                flags=re.DOTALL,
            )
        ]
        if len(steps) >= 3:
            if variant % 4 == 1:
                return "First, " + " Next, ".join(
                    _inline_sentence(step) for step in steps
                )
            if variant % 4 == 2:
                return "\n".join(f"- {_sentence(step)}" for step in steps)
            if variant % 4 == 3:
                return " ".join(_sentence(step) for step in steps)
        return direct
    return re.sub(
        r"^(?:Next step|Owner|Timing|Core idea|Example|Check|Decision|Action|Open point|Weakness|Revision|Immediate action|Boundary):\s*",
        "",
        response,
        flags=re.IGNORECASE,
    )


_STRUCTURE_SLOT = re.compile(
    r"\b(?:[A-Z0-9]{5,}|[A-Z]+\d+[A-Z0-9]*|(?i:day)\s+\d+|\d{1,2}:\d{2}|\$\d+(?:\.\d+)?|\d+(?:\.\d+)?)\b",
)


def _normalized_structure(text: str) -> str:
    """Normalize volatile slots while retaining syntax and answer shape."""

    normalized = _STRUCTURE_SLOT.sub("<slot>", text)
    normalized = re.sub(r"[\"'“”][^\"'“”]{1,80}[\"'“”]", "<quoted>", normalized)
    normalized = re.sub(r"(?m)^\s*\d+[.)]\s*", "<item> ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _deduplicate_structural_rows(
    rows: list[dict[str, Any]],
    *,
    target_key: str = "_projected_target",
    max_per_structure: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bound deterministic examples per task and normalized answer shape."""

    if max_per_structure < 1:
        raise ValueError("max_per_structure must be positive")

    kept: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    retained: Counter[tuple[str, str]] = Counter()
    for row in sorted(rows, key=lambda item: item["example_id"]):
        signature = _normalized_structure(row[target_key])
        key = (row["task"], signature)
        counts[key] += 1
        if retained[key] >= max_per_structure:
            continue
        retained[key] += 1
        copy = dict(row)
        copy["_structure_signature"] = signature
        kept.append(copy)
    return kept, {
        "input_examples": len(rows),
        "kept_examples": len(kept),
        "dropped_structural_duplicates": len(rows) - len(kept),
        "distinct_task_structures": len(counts),
        "maximum_retained_per_structure": max_per_structure,
        "maximum_examples_per_structure_before_dedup": max(counts.values(), default=0),
    }


def _project_sft_exchange(
    messages: list[dict[str, str]],
    *,
    example_id: str,
    task: str,
    answer_json: str,
) -> tuple[str, str, TrainingCards]:
    try:
        metadata = json.loads(answer_json) if answer_json else {}
    except json.JSONDecodeError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    cards = deal_training_cards(
        task=task,
        mode="chat" if len(messages) > 2 else "instruct",
        example_id=example_id,
        metadata=metadata,
    )
    prompt = _render_natural_instruction(messages, example_id, cards)
    if metadata.get("evaluation_source") == "separately_authored":
        target = _final_assistant_target(messages)
    else:
        target = _naturalize_assistant_target(
            messages,
            task=task,
            cards=cards,
            example_id=example_id,
        )
    return prompt, correct_indefinite_articles(target), cards


def load_heldout_evaluation(path: Path) -> list[dict[str, Any]]:
    """Load independently authored evaluation exchanges into the common schema."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples", [])
    if not examples:
        raise ValueError("held-out evaluation contains no examples")
    rows: list[dict[str, Any]] = []
    for item in examples:
        messages = [
            {"role": "user", "content": item["prompt"].strip()},
            {"role": "assistant", "content": item["response"].strip()},
        ]
        rows.append(
            {
                "example_id": f"heldout:{item['id']}",
                "task": item["task"],
                "mode": "instruct",
                "difficulty": item.get("difficulty", "medium"),
                "dataset_id": payload["dataset_id"],
                "domain": item["domain"],
                "language": "en",
                "split": "validation",
                "messages": messages,
                "prompt": messages[0]["content"],
                "response": messages[1]["content"],
                "rendered_text": _render_messages(messages),
                "source_keys": [item["id"]],
                "evidence": item.get("evidence", []),
                "answer_json": json.dumps(
                    {
                        "evaluation_source": "separately_authored",
                        "use_verbatim_target": True,
                    },
                    sort_keys=True,
                ),
                "source": payload["source"],
                "source_urls": [],
                "license": payload["license"],
                "version": payload["version"],
            }
        )
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("held-out evaluation contains duplicate ids")
    return rows


_FORBIDDEN_SFT_TARGET_PHRASES = (
    "hand ",
    "next step:",
    "owner:",
    "timing:",
    "core idea:",
    "example:",
    "check:",
    "decision:",
    "action:",
    "open point:",
    "open item:",
    "weakness:",
    "revision:",
    "immediate action:",
    "boundary:",
    "sequence:",
    "fallback trigger:",
    "revised text:",
    "assigned action:",
    "remaining work:",
    "blocker:",
    "a concrete example is this:",
    "consider this example:",
    "keep this limit in mind:",
    "each description states",
    "remain feasible under the stated limits",
    "the response should",
    "the final review should",
    "treat the task as complete",
)

_GENERALIST_POST_TRAINING_TASKS = {
    "practical_action",
    "explanation_learning",
    "troubleshooting",
    "writing_transformation",
    "planning_comparison",
    "conversation_empathy",
    "safety_uncertainty",
    "grounded_qa",
    "summarization_synthesis",
    "extraction_classification",
    "reasoning_verification",
    "critique_revision",
    "brainstorming_creativity",
    "context_clarification",
}


def _audit_sft_projection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hits: list[dict[str, str]] = []
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        target = row["_projected_target"]
        lowered = target.lower()
        for phrase in _FORBIDDEN_SFT_TARGET_PHRASES:
            if phrase in lowered:
                hits.append(
                    {
                        "example_id": row["example_id"],
                        "task": row["task"],
                        "phrase": phrase,
                    }
                )
        by_task[row["task"]][_normalized_structure(target)] += 1
    if hits:
        raise ValueError(f"model-facing answer contains a control rubric: {hits[0]}")
    task_stats = {
        task: {
            "examples": sum(counts.values()),
            "distinct_normalized_structures": len(counts),
            "maximum_structure_share": round(
                max(counts.values()) / sum(counts.values()), 6
            ),
        }
        for task, counts in sorted(by_task.items())
    }
    underspecified = [
        task
        for task, stats in task_stats.items()
        if task in _GENERALIST_POST_TRAINING_TASKS
        and stats["examples"] > 1
        and stats["distinct_normalized_structures"] < 2
    ]
    if underspecified:
        raise ValueError(
            "model-facing family has only one normalized structure: "
            f"{underspecified}"
        )
    return {
        "examples": len(rows),
        "exact_answer_uniqueness_ratio": round(
            len({row["_projected_target"] for row in rows}) / len(rows), 6
        ),
        "control_rubric_hits": 0,
        "tasks": task_stats,
    }


def _example_id(
    task: str,
    dataset_id: str,
    source_keys: Iterable[str],
    messages: list[dict[str, str]],
) -> str:
    material = "|".join(
        (task, dataset_id, *source_keys, _render_messages(messages))
    )
    suffix = hashlib.sha256(material.encode()).hexdigest()[:16]
    return f"atlas-instruct:{task}:{suffix}"


def _render_messages(messages: list[dict[str, str]]) -> str:
    labels = {"user": "User", "assistant": "Assistant"}
    return "\n".join(
        f"{labels[message['role']]}: {message['content']}" for message in messages
    )


def _row(
    *,
    task: str,
    difficulty: str,
    card: dict[str, Any],
    messages: list[dict[str, str]],
    source_keys: list[str],
    evidence: list[str],
    answer_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not messages or messages[0]["role"] != "user":
        raise ValueError("instruction conversations must begin with a user message")
    if messages[-1]["role"] != "assistant":
        raise ValueError("instruction conversations must end with an assistant message")
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected or not message["content"].strip():
            raise ValueError("instruction roles must alternate and contain text")
    return {
        "example_id": _example_id(task, card["dataset_id"], source_keys, messages),
        "task": task,
        "mode": "instruct" if len(messages) == 2 else "chat",
        "difficulty": difficulty,
        "dataset_id": card["dataset_id"],
        "domain": card["domain"],
        "language": card["language"],
        "split": card["split"],
        "messages": messages,
        "prompt": messages[0]["content"],
        "response": messages[-1]["content"],
        "rendered_text": _render_messages(messages),
        "source_keys": source_keys,
        "evidence": evidence,
        "answer_json": (
            json.dumps(answer_json, sort_keys=True, ensure_ascii=False)
            if answer_json is not None
            else ""
        ),
        "source": card["source"],
        "source_urls": card["source_urls"],
        "license": card["license"],
        "version": card["version"],
    }


def _entity_rows(
    card: dict[str, Any],
    *,
    max_attributes_per_card: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    name = card["name"]
    key = card["key"]
    task_key = f"{card['dataset_id']}:{key}"
    question = _pick(
        task_key,
        (
            f"What is {name} according to the atlas?",
            f"Give a concise, grounded description of {name}.",
            f"Summarize the atlas entry for {name} without adding new facts.",
        ),
    )
    rows.append(
        _row(
            task="entity_summary",
            difficulty="easy",
            card=card,
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": card["description"]},
            ],
            source_keys=[key],
            evidence=[card["description"]],
        )
    )

    attributes = json.loads(card["attributes_json"])
    selected_attributes = sorted(
        attributes.items(),
        key=lambda item: hashlib.sha256(
            f"{task_key}:{item[0]}".encode()
        ).digest(),
    )[:max_attributes_per_card]
    for attribute, value in sorted(selected_attributes):
        readable = attribute.replace("_", " ")
        question = _pick(
            f"{task_key}:{attribute}",
            (
                f"What is the recorded {readable} of {name}?",
                f"In the atlas, which value is listed for {name}'s {readable}?",
                f"Answer from the card only: what is {name}'s {readable}?",
            ),
        )
        answer = f"The recorded {readable} of {name} is {json.dumps(value, ensure_ascii=False)}."
        rows.append(
            _row(
                task="attribute_query",
                difficulty="easy",
                card=card,
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                source_keys=[key],
                evidence=[f"{attribute}={json.dumps(value, ensure_ascii=False)}"],
            )
        )

    if card["facts"]:
        fact = card["facts"][_stable_index(task_key, len(card["facts"]))]
        rows.append(
            _row(
                task="recorded_fact",
                difficulty="easy",
                card=card,
                messages=[
                    {
                        "role": "user",
                        "content": f"State one documented fact about {name}.",
                    },
                    {"role": "assistant", "content": fact},
                ],
                source_keys=[key],
                evidence=[fact],
            )
        )

    answer_object = {
        "key": key,
        "kind": card["kind"],
        "name": name,
        "summary": card["summary"],
        "attributes": attributes,
    }
    answer_text = json.dumps(answer_object, sort_keys=True, ensure_ascii=False)
    rows.append(
        _row(
            task="structured_extraction",
            difficulty="medium",
            card=card,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Return the atlas record for {name} as JSON with exactly "
                        "these fields: key, kind, name, summary, attributes."
                    ),
                },
                {"role": "assistant", "content": answer_text},
            ],
            source_keys=[key],
            evidence=[card["summary"], card["attributes_json"]],
            answer_json=answer_object,
        )
    )

    if attributes:
        attribute = sorted(attributes)[_stable_index(f"followup:{task_key}", len(attributes))]
        value = attributes[attribute]
        readable = attribute.replace("_", " ")
        rows.append(
            _row(
                task="grounded_followup",
                difficulty="medium",
                card=card,
                messages=[
                    {"role": "user", "content": f"Briefly identify {name}."},
                    {"role": "assistant", "content": card["summary"]},
                    {"role": "user", "content": f"What is its recorded {readable}?"},
                    {
                        "role": "assistant",
                        "content": f"Its recorded {readable} is {json.dumps(value, ensure_ascii=False)}.",
                    },
                ],
                source_keys=[key],
                evidence=[card["summary"], f"{attribute}={json.dumps(value, ensure_ascii=False)}"],
            )
        )
    return rows


def build_instruction_dataset(
    corpus_root: Path,
    output_root: Path,
    *,
    max_attributes_per_card: int = 2,
    max_relations_per_card: int = 2,
    max_paths_per_card: int = 1,
) -> dict[str, Any]:
    if (
        max_attributes_per_card < 0
        or max_relations_per_card < 0
        or max_paths_per_card < 0
    ):
        raise ValueError("per-card limits cannot be negative")
    cards = pq.read_table(corpus_root / "cards.parquet").to_pylist()
    relations = pq.read_table(corpus_root / "relations.parquet").to_pylist()
    documents = pq.read_table(corpus_root / "documents.parquet").to_pylist()
    card_index = {(row["dataset_id"], row["key"]): row for row in cards}

    rows: list[dict[str, Any]] = []
    for card in cards:
        rows.extend(
            _entity_rows(
                card,
                max_attributes_per_card=max_attributes_per_card,
            )
        )

    relations_by_card: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for relation in relations:
        relations_by_card[(relation["dataset_id"], relation["from_key"])].append(relation)
    for source_ref, source_relations in sorted(relations_by_card.items()):
        source_card = card_index[source_ref]
        for relation in sorted(
            source_relations,
            key=lambda item: (item["relation"], item["to_dataset_id"], item["to_key"]),
        )[:max_relations_per_card]:
            target_ref = (relation["to_dataset_id"], relation["to_key"])
            target_card = card_index[target_ref]
            relation_name = relation["relation"].replace("_", " ")
            answer = relation["detail"] or (
                f"{source_card['name']} {relation_name} {target_card['name']}."
            )
            rows.append(
                _row(
                    task="direct_relation",
                    difficulty="medium",
                    card=source_card,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"How does the atlas connect {source_card['name']} "
                                f"to {target_card['name']}?"
                            ),
                        },
                        {"role": "assistant", "content": answer},
                    ],
                    source_keys=[source_card["key"], target_card["key"]],
                    evidence=[answer],
                    answer_json={
                        "from": source_card["key"],
                        "relation": relation["relation"],
                        "to": target_card["key"],
                    },
                )
            )

    cards_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        cards_by_group[(card["dataset_id"], card["kind"])].append(card)
    for group_cards in cards_by_group.values():
        ordered = sorted(group_cards, key=lambda item: item["key"])
        for left, right in zip(ordered[::2], ordered[1::2]):
            answer = (
                f"{left['name']}: {left['summary']}\n"
                f"{right['name']}: {right['summary']}"
            )
            rows.append(
                _row(
                    task="compare_records",
                    difficulty="medium",
                    card=left,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Compare the recorded atlas summaries of {left['name']} "
                                f"and {right['name']}. Do not infer beyond the cards."
                            ),
                        },
                        {"role": "assistant", "content": answer},
                    ],
                    source_keys=[left["key"], right["key"]],
                    evidence=[left["summary"], right["summary"]],
                )
            )

    path_counts: Counter[tuple[str, str]] = Counter()
    for document in sorted(documents, key=lambda item: item["document_id"]):
        if document["template"] != "path" or len(document["source_keys"]) < 2:
            continue
        start_ref = (document["dataset_id"], document["source_keys"][0])
        if path_counts[start_ref] >= max_paths_per_card:
            continue
        start_card = card_index[start_ref]
        end_card = card_index[(document["dataset_id"], document["source_keys"][-1])]
        answer = document["text"].split("\n\n", 1)[-1].strip()
        rows.append(
            _row(
                task="multi_hop_path",
                difficulty="hard",
                card=start_card,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Using only recorded links, trace a path from {start_card['name']} "
                            f"to {end_card['name']}."
                        ),
                    },
                    {"role": "assistant", "content": answer},
                ],
                source_keys=document["source_keys"],
                evidence=[answer],
            )
        )
        path_counts[start_ref] += 1

    rows.sort(key=lambda row: row["example_id"])
    example_ids = [row["example_id"] for row in rows]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("instruction example IDs are not unique")
    source_splits = {
        (card["dataset_id"], card["key"]): card["split"] for card in cards
    }
    for row in rows:
        if any(
            source_splits[(row["dataset_id"], key)] != row["split"]
            for key in row["source_keys"]
        ):
            raise ValueError(f"split leakage in {row['example_id']}")

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    table = pa.Table.from_pylist(rows, schema=INSTRUCTION_SCHEMA)
    parquet_path = temporary / "instructions.parquet"
    pq.write_table(
        table,
        parquet_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    counts = {
        "examples": len(rows),
        "examples_by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "examples_by_task": dict(sorted(Counter(row["task"] for row in rows).items())),
        "examples_by_mode": dict(sorted(Counter(row["mode"] for row in rows).items())),
        "source_cards": len(cards),
        "source_relations": len(relations),
    }
    manifest = {
        "format": "complexity-atlas-instruct-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY-NC 4.0",
        "counts": counts,
        "generation": {
            "method": "deterministic templates over authored linked cards",
            "max_attributes_per_card": max_attributes_per_card,
            "max_relations_per_card": max_relations_per_card,
            "max_paths_per_card": max_paths_per_card,
            "model_generated": False,
        },
        "source_corpus": {
            "path": str(corpus_root.resolve()),
            "cards_sha256": file_sha256(corpus_root / "cards.parquet"),
            "relations_sha256": file_sha256(corpus_root / "relations.parquet"),
            "documents_sha256": file_sha256(corpus_root / "documents.parquet"),
        },
        "files": {
            "instructions.parquet": {
                "bytes": parquet_path.stat().st_size,
                "sha256": file_sha256(parquet_path),
            }
        },
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return manifest


def _encode_messages(
    messages: list[dict[str, str]],
    example_id: str,
    task: str,
    answer_json: str,
    encoding,
    eos_id: int,
    chat_template: dict[str, Any],
    projection: tuple[str, str, TrainingCards] | None = None,
) -> tuple[list[int], list[int], TrainingCards]:
    """Project a card conversation into one direct SFT exchange.

    The authored parquet keeps its complete card hand and two- or four-message
    conversation. For model training, the cards are rendered as a natural
    instruction and only the final assistant answer is supervised. Intermediate
    acknowledgements and hand identifiers are deliberately omitted.
    """

    if projection is None:
        projection = _project_sft_exchange(
            messages,
            example_id=example_id,
            task=task,
            answer_json=answer_json,
        )
    user_content, final_assistant, cards = projection
    full_ids: list[int] = []
    target_labels: list[int] = []
    system_tokens = encoding.encode(
        render_system_prefix(chat_template),
        disallowed_special=(),
    )
    full_ids.extend(system_tokens)
    target_labels.extend([IGNORE_INDEX] * len(system_tokens))
    user_tokens = encoding.encode(
        render_user_turn(user_content, chat_template),
        disallowed_special=(),
    )
    full_ids.extend(user_tokens)
    target_labels.extend([IGNORE_INDEX] * len(user_tokens))
    prefix = encoding.encode(
        chat_template["assistant_prefix"],
        disallowed_special=(),
    )
    response = encoding.encode(final_assistant, disallowed_special=())
    full_ids.extend(prefix)
    target_labels.extend([IGNORE_INDEX] * len(prefix))
    full_ids.extend(response)
    target_labels.extend(response)
    full_ids.append(eos_id)
    target_labels.append(eos_id)
    # Causal alignment: logits at position t predict token t+1. Supervision is
    # active only when that next token belongs to an assistant response.
    return full_ids[:-1], target_labels[1:], cards


def tokenize_instruction_dataset(
    instructions_path: Path,
    tokenizer_root: Path,
    output_root: Path,
    heldout_evaluation_path: Path | None = None,
) -> dict[str, Any]:
    encoding, tokenizer_config = load_encoding(tokenizer_root)
    eos_token = tokenizer_config.get("eos_token", "<|endoftext|>")
    try:
        eos_id = encoding.encode_single_token(eos_token)
    except KeyError:
        eos_id = encoding.eot_token
    chat_template = chat_template_contract()
    chat_template["eos_token"] = eos_token
    source_rows = sorted(
        pq.read_table(instructions_path).to_pylist(),
        key=lambda row: row["example_id"],
    )
    evaluation_sha256: str | None = None
    if heldout_evaluation_path is not None:
        source_rows = [row for row in source_rows if row["split"] == "train"]
        source_rows.extend(load_heldout_evaluation(heldout_evaluation_path))
        evaluation_sha256 = file_sha256(heldout_evaluation_path)

    projected_rows: list[dict[str, Any]] = []
    for row in source_rows:
        prompt, target, cards = _project_sft_exchange(
            row["messages"],
            example_id=row["example_id"],
            task=row["task"],
            answer_json=row["answer_json"],
        )
        projected_rows.append(
            {
                **row,
                "_projected_prompt": prompt,
                "_projected_target": target,
                "_conditioning_cards": cards,
            }
        )
    projection_audit = _audit_sft_projection(projected_rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in projected_rows:
        partition = {"train": "train", "validation": "eval", "test": "test"}[row["split"]]
        grouped[partition].append(row)

    deduplication: dict[str, Any] = {}
    for partition, partition_rows in list(grouped.items()):
        grouped[partition], deduplication[partition] = _deduplicate_structural_rows(
            partition_rows,
            max_per_structure=8 if partition == "train" else 1,
        )

    train_structures = {
        (row["task"], row["_structure_signature"])
        for row in grouped.get("train", [])
    }
    eval_structures = {
        (row["task"], row["_structure_signature"])
        for row in grouped.get("eval", [])
    }
    overlap = train_structures & eval_structures
    if heldout_evaluation_path is not None and overlap:
        sample = next(iter(overlap))
        raise ValueError(
            "held-out evaluation shares a normalized answer structure with training: "
            f"{sample}"
        )

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    chat_template_path = temporary / "chat_template.json"
    chat_template_path.write_text(
        json.dumps(chat_template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    projected_records = [
        {
            "example_id": row["example_id"],
            "task": row["task"],
            "mode": row["mode"],
            "difficulty": row["difficulty"],
            "domain": row["domain"],
            "language": row["language"],
            "split": "validation" if partition == "eval" else partition,
            "prompt": row["_projected_prompt"],
            "response": row["_projected_target"],
            "structure_signature": row["_structure_signature"],
            "source_representation": (
                "card_hand" if _card_sections(row["messages"]) is not None else "conversation"
            ),
            "source": row["source"],
            "license": row["license"],
            "version": row["version"],
        }
        for partition, partition_rows in sorted(grouped.items())
        for row in partition_rows
    ]
    projected_path = temporary / "projected.parquet"
    pq.write_table(
        pa.Table.from_pylist(projected_records, schema=PROJECTED_SFT_SCHEMA),
        projected_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )

    manifests: dict[str, Any] = {}
    for partition, partition_rows in sorted(grouped.items()):
        root = temporary / partition
        root.mkdir()
        inputs_path = root / "input_ids.bin"
        labels_path = root / "labels.bin"
        examples_path = root / "examples.jsonl"
        offset = 0
        supervised_tokens = 0
        conditioning_counts: dict[str, Counter[str]] = defaultdict(Counter)
        with inputs_path.open("wb") as inputs_handle, labels_path.open("wb") as labels_handle, examples_path.open("w", encoding="utf-8") as examples_handle:
            for row in partition_rows:
                has_card_hand = _card_sections(row["messages"]) is not None
                input_ids, labels, conditioning_cards = _encode_messages(
                    row["messages"],
                    row["example_id"],
                    row["task"],
                    row["answer_json"],
                    encoding,
                    eos_id,
                    chat_template,
                    projection=(
                        row["_projected_prompt"],
                        row["_projected_target"],
                        row["_conditioning_cards"],
                    ),
                )
                np.asarray(input_ids, dtype=TOKEN_DTYPE).tofile(inputs_handle)
                np.asarray(labels, dtype=LABEL_DTYPE).tofile(labels_handle)
                for card_name, value in conditioning_cards.as_dict().items():
                    conditioning_counts[card_name][value] += 1
                supervised = sum(label != IGNORE_INDEX for label in labels)
                examples_handle.write(
                    json.dumps(
                        {
                            "example_id": row["example_id"],
                            "hand_id": row["example_id"],
                            "source_representation": (
                                "card_hand" if has_card_hand else "conversation"
                            ),
                            "training_representation": "natural_instruction",
                            "conditioning_cards": conditioning_cards.as_dict(),
                            "cards": (
                                ["situation", "data", "rule", "goal"]
                                if has_card_hand
                                else []
                            ),
                            "task": row["task"],
                            "structure_signature": row["_structure_signature"],
                            "offset": offset,
                            "num_tokens": len(input_ids),
                            "supervised_tokens": supervised,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                offset += len(input_ids)
                supervised_tokens += supervised
        metadata = {
            "format": "complexity-sft-token-shard-v1",
            "chat_template_id": CHAT_TEMPLATE_ID,
            "chat_template_sha256": file_sha256(chat_template_path),
            "partition": partition,
            "examples": len(partition_rows),
            "num_tokens": offset,
            "supervised_tokens": supervised_tokens,
            "ignore_index": IGNORE_INDEX,
            "input_dtype": TOKEN_DTYPE.str,
            "label_dtype": LABEL_DTYPE.str,
            "vocab_size": encoding.n_vocab,
            "eos_token_id": eos_id,
            "tokenizer": tokenizer_config["encoding_name"],
            "tokenizer_sha256": directory_sha256(tokenizer_root),
            "source_sha256": file_sha256(instructions_path),
            "evaluation_source_sha256": evaluation_sha256,
            "input_ids_sha256": file_sha256(inputs_path),
            "labels_sha256": file_sha256(labels_path),
            "examples_sha256": file_sha256(examples_path),
            "conditioning_card_counts": {
                name: dict(sorted(counts.items()))
                for name, counts in sorted(conditioning_counts.items())
            },
        }
        (root / "sft.idx.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        manifests[partition] = metadata
    manifest = {
        "format": "complexity-atlas-instruct-tokenized-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": tokenizer_config["encoding_name"],
        "chat_template_id": CHAT_TEMPLATE_ID,
        "chat_template_sha256": file_sha256(chat_template_path),
        "serialization": (
            "System:\\n<system>\\n\\nUser:\\n<natural instruction rendered "
            "from card attributes>\\n\\nAssistant:\\n<final answer without hand id><eos>"
        ),
        "training_projection": chat_template["training_projection"],
        "projection_audit": projection_audit,
        "structural_deduplication": deduplication,
        "train_eval_structure_overlap": len(overlap),
        "heldout_evaluation": (
            {
                "path": str(heldout_evaluation_path),
                "sha256": evaluation_sha256,
                "method": "separately_authored",
            }
            if heldout_evaluation_path is not None
            else None
        ),
        "projected_parquet": {
            "path": projected_path.name,
            "examples": len(projected_records),
            "bytes": projected_path.stat().st_size,
            "sha256": file_sha256(projected_path),
            "splits": dict(
                sorted(Counter(row["split"] for row in projected_records).items())
            ),
        },
        "partitions": manifests,
        "total_examples": sum(item["examples"] for item in manifests.values()),
        "total_tokens": sum(item["num_tokens"] for item in manifests.values()),
        "total_supervised_tokens": sum(item["supervised_tokens"] for item in manifests.values()),
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return manifest
