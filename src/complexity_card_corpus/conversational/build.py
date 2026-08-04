from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..build import file_sha256
from ..sft.schema import INSTRUCTION_SCHEMA


_DATASET_ID = "complexity-casual-conversation-v1"
_WORD = re.compile(r"[a-z0-9']+")
_PLACEHOLDER = re.compile(r"\{[^{}]+\}")
_REQUIRED_SUBCARDS = {
    "opening",
    "acknowledgement",
    "question",
    "detail",
    "reply",
    "follow_up_question",
    "shift",
    "closing",
}
_STAGES = (
    "user_opening",
    "assistant_entry",
    "user_follow_up",
    "assistant_follow_up",
    "user_shift",
    "assistant_closing_bridge",
    "assistant_closing",
)


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _stable_index(value: str, size: int) -> int:
    if size < 1:
        raise ValueError("cannot select from an empty deck")
    return int.from_bytes(_digest(value)[:8], "big") % size


def _lower_first(value: str) -> str:
    value = value.strip()
    if value.startswith(("I ", "I'm ", "I've ", "I'd ")):
        return value
    return value[:1].lower() + value[1:] if value else value


def _sentence(value: str) -> str:
    value = " ".join(value.strip().split())
    if not value:
        return value
    return value if value[-1] in ".?!" else f"{value}."


def _render(template: str, values: dict[str, str]) -> str:
    expanded = {
        **values,
        **{f"{key}_lower": _lower_first(value) for key, value in values.items()},
    }
    return _sentence(template.format(**expanded))


def _join(*parts: str) -> str:
    return " ".join(_sentence(part) for part in parts if part.strip()).strip()


def _split(source_pair_id: str, validation_percent: int) -> str:
    return (
        "validation"
        if _stable_index(f"split:{source_pair_id}", 100) < validation_percent
        else "train"
    )


def _validate_registry(registry: dict[str, Any]) -> None:
    topics = registry.get("topic_cards", [])
    contexts = registry.get("context_cards", [])
    decks = registry.get("surface_decks", {})
    if len(topics) < 20 or len(contexts) < 20:
        raise ValueError("casual conversation needs at least 20 topic and context cards")
    if set(decks) != set(_STAGES):
        raise ValueError("surface deck stages do not match the conversation contract")
    if any(len(decks[stage]) < 8 for stage in _STAGES):
        raise ValueError("every conversational surface deck needs eight subcards")
    topic_ids = [card["topic_id"] for card in topics]
    context_ids = [card["context_id"] for card in contexts]
    if len(set(topic_ids)) != len(topic_ids) or len(set(context_ids)) != len(context_ids):
        raise ValueError("conversation card IDs must be unique")
    for card in topics:
        if not _REQUIRED_SUBCARDS <= set(card.get("subcards", {})):
            raise ValueError(f"topic {card['topic_id']} has an incomplete subcard deck")
    for stage, templates in decks.items():
        if len(set(templates)) != len(templates):
            raise ValueError(f"duplicate surface subcard in {stage}")


def _messages(
    *,
    topic: dict[str, Any],
    context: dict[str, Any],
    decks: dict[str, list[str]],
    variant: int,
    pair_index: int,
) -> tuple[list[dict[str, str]], dict[str, int], int]:
    subcards = topic["subcards"]
    values = {
        **subcards,
        "context_opening": context["opening_addition"],
        "context_detail": context["detail_addition"],
        "context_shift": context["shift_addition"],
        "context_closing": context["closing_addition"],
    }
    pair_key = f"{topic['topic_id']}:{context['context_id']}"
    # Walk the full Cartesian surface space without replacement inside each
    # semantic topic/context pair. Response-bearing stages are the least
    # significant digits, so every retained variant receives a distinct final
    # target long before the larger surface space can cycle.
    digit_order = (
        "assistant_closing",
        "assistant_closing_bridge",
        "assistant_follow_up",
        "user_opening",
        "assistant_entry",
        "user_follow_up",
        "user_shift",
    )
    surface_capacity = 1
    for stage in digit_order:
        surface_capacity *= len(decks[stage])
    ordinal = (
        variant
        + _stable_index(f"surface-offset:{pair_key}:{pair_index}", surface_capacity)
    ) % surface_capacity
    indexes: dict[str, int] = {}
    for stage in digit_order:
        indexes[stage] = ordinal % len(decks[stage])
        ordinal //= len(decks[stage])
    turn_count = 4 if _stable_index(f"turns:{topic['topic_id']}:{context['context_id']}:{variant}", 5) < 2 else 6
    opening = _render(decks["user_opening"][indexes["user_opening"]], values)
    entry = _render(decks["assistant_entry"][indexes["assistant_entry"]], values)
    follow_up = _render(
        decks["user_follow_up"][indexes["user_follow_up"]], values
    )
    assistant_reply = _render(
        decks["assistant_follow_up"][indexes["assistant_follow_up"]], values
    )
    closing = _join(
        _render(
            decks["assistant_closing_bridge"][indexes["assistant_closing_bridge"]],
            values,
        ),
        _render(decks["assistant_closing"][indexes["assistant_closing"]], values),
    )
    if turn_count == 4:
        messages = [
            {"role": "user", "content": opening},
            {"role": "assistant", "content": entry},
            {"role": "user", "content": follow_up},
            {"role": "assistant", "content": _join(assistant_reply, closing)},
        ]
    else:
        assistant_reply = _join(assistant_reply, subcards["follow_up_question"])
        shift = _render(decks["user_shift"][indexes["user_shift"]], values)
        messages = [
            {"role": "user", "content": opening},
            {"role": "assistant", "content": entry},
            {"role": "user", "content": follow_up},
            {"role": "assistant", "content": assistant_reply},
            {"role": "user", "content": shift},
            {"role": "assistant", "content": closing},
        ]
    return messages, indexes, turn_count


def _audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rendered = [row["rendered_text"] for row in rows]
    responses = [row["response"] for row in rows]
    if len(set(rendered)) != len(rows):
        raise ValueError("duplicate casual conversation")
    if len(set(responses)) != len(rows):
        raise ValueError("duplicate casual final response")
    surface_hands: Counter[str] = Counter()
    response_structures: Counter[str] = Counter()
    train_pairs: set[str] = set()
    validation_pairs: set[str] = set()
    source_pairs: Counter[str] = Counter()
    four_grams: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        metadata = json.loads(row["answer_json"])
        source_pairs[metadata["source_pair_id"]] += 1
        surface_hands[metadata["surface_hand"]] += 1
        response_structures[metadata["response_structure"]] += 1
        target = validation_pairs if row["split"] == "validation" else train_pairs
        target.add(metadata["source_pair_id"])
        for message in row["messages"]:
            text = message["content"]
            if _PLACEHOLDER.search(text):
                raise ValueError("unrendered conversational placeholder")
            words = _WORD.findall(text.lower())
            four_grams.update(
                tuple(words[index : index + 4])
                for index in range(max(0, len(words) - 3))
            )
    largest_surface = max(surface_hands.values()) / len(rows)
    largest_response = max(response_structures.values()) / len(rows)
    if largest_surface > 0.05 or largest_response > 0.05:
        raise ValueError("conversation structure exceeds the five-percent ceiling")
    max_four_gram = max(four_grams.values(), default=0)
    message_count = sum(len(row["messages"]) for row in rows)
    if max_four_gram / message_count > 0.05:
        raise ValueError("a four-word conversational phrase exceeds five percent")
    overlap = train_pairs & validation_pairs
    if overlap:
        raise ValueError("conversation source pairs leak across splits")
    return {
        "rows": len(rows),
        "unique_conversation_ratio": len(set(rendered)) / len(rows),
        "unique_final_response_ratio": len(set(responses)) / len(rows),
        "largest_surface_hand_share": largest_surface,
        "largest_response_structure_share": largest_response,
        "largest_four_gram_share_per_message": max_four_gram / message_count,
        "source_pair_split_overlap": len(overlap),
        "largest_source_pair_variant_count": max(source_pairs.values(), default=0),
        "turn_counts": dict(sorted(Counter(len(row["messages"]) for row in rows).items())),
    }


def build_casual_conversation_surface(
    registry_path: Path,
    output_root: Path,
    *,
    examples: int | None = None,
    seed: int = 42,
    validation_percent: int = 5,
) -> dict[str, Any]:
    """Build an additive casual-dialogue supplement from linked original cards."""

    if not 1 <= validation_percent <= 25:
        raise ValueError("validation percent must be between 1 and 25")
    registry = json.loads(registry_path.read_text())
    _validate_registry(registry)
    topics = registry["topic_cards"]
    contexts = registry["context_cards"]
    decks = registry["surface_decks"]
    pairs = [
        (topic, context)
        for topic in topics
        for context in contexts
    ]
    pairs.sort(key=lambda pair: _digest(f"{seed}:{pair[0]['topic_id']}:{pair[1]['context_id']}"))
    if examples is None:
        examples = len(pairs)
    if examples < 100:
        raise ValueError("casual conversation build needs at least 100 examples")
    capacity = len(pairs)
    for stage in _STAGES:
        capacity *= len(decks[stage])
    if examples > capacity:
        raise ValueError(f"requested {examples} rows exceeds card capacity {capacity}")

    rows: list[dict[str, Any]] = []
    for ordinal in range(examples):
        pair_index = ordinal % len(pairs)
        variant = ordinal // len(pairs)
        topic, context = pairs[pair_index]
        source_pair_id = f"{topic['topic_id']}:context:{context['context_id']}"
        messages, indexes, turn_count = _messages(
            topic=topic,
            context=context,
            decks=decks,
            variant=variant,
            pair_index=pair_index,
        )
        rendered = "\n".join(
            f"{message['role'].title()}: {message['content']}" for message in messages
        )
        example_id = f"casual:{hashlib.sha256(rendered.encode()).hexdigest()[:24]}"
        surface_hand = "|".join(f"{stage}={indexes[stage]}" for stage in _STAGES)
        response_structure = "|".join(
            (
                f"turns={turn_count}",
                f"entry={indexes['assistant_entry']}",
                f"follow={indexes['assistant_follow_up']}",
                f"bridge={indexes['assistant_closing_bridge']}",
                f"closing={indexes['assistant_closing']}",
            )
        )
        rows.append(
            {
                "example_id": example_id,
                "task": "casual_conversation",
                "mode": "chat",
                "difficulty": "easy" if turn_count == 4 else "medium",
                "dataset_id": _DATASET_ID,
                "domain": topic["domain"],
                "language": "en",
                "split": _split(source_pair_id, validation_percent),
                "messages": messages,
                "prompt": messages[0]["content"],
                "response": messages[-1]["content"],
                "rendered_text": rendered,
                "source_keys": [topic["topic_id"], context["context_id"]],
                "evidence": [],
                "answer_json": json.dumps(
                    {
                        "conversation_kind": "casual",
                        "source_pair_id": source_pair_id,
                        "topic_card_id": topic["topic_id"],
                        "context_card_id": context["context_id"],
                        "surface_hand": surface_hand,
                        "response_structure": response_structure,
                        "deck_topology": {
                            "stages": list(_STAGES),
                            "links": [
                                f"{_STAGES[index]}->{_STAGES[index + 1]}"
                                for index in range(len(_STAGES) - 1)
                            ],
                        },
                    },
                    sort_keys=True,
                ),
                "source": registry["source"],
                "source_urls": [],
                "license": registry["license"],
                "version": registry["version"],
            }
        )
    rows.sort(key=lambda row: row["example_id"])
    audit = _audit(rows)

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    output_path = temporary / "conversations.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=INSTRUCTION_SCHEMA),
        output_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
        row_group_size=5_000,
        write_page_index=True,
    )
    audit_path = temporary / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = {
        "format": "casual-conversation-surface-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "additive GPT-2-style natural conversation supplement",
        "seed": seed,
        "counts": {
            "examples": len(rows),
            "by_task": dict(sorted(Counter(row["task"] for row in rows).items())),
            "by_domain": dict(sorted(Counter(row["domain"] for row in rows).items())),
            "by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        },
        "capacity": capacity,
        "audit": audit,
        "inputs": {
            "registry": {"path": str(registry_path), "sha256": file_sha256(registry_path)}
        },
        "files": {},
    }
    manifest["files"] = {
        "conversations.parquet": {
            "sha256": file_sha256(output_path),
            "bytes": output_path.stat().st_size,
        },
        "audit.json": {
            "sha256": file_sha256(audit_path),
            "bytes": audit_path.stat().st_size,
        },
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.rename(output_root)
    return manifest
