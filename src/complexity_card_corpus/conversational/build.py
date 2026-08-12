from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..build import file_sha256
from ..conversation_quality import audit_casual_conversation_quality
from ..variable_by import casual_variable_by
from ..variable_by.reservoirs import CASUAL_ARC_CARDS, CASUAL_INTENT_CARDS


_DATASET_ID = "complexity-casual-conversation-v18"
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
    "assistant_closing",
)


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _stable_index(value: str, size: int) -> int:
    if size < 1:
        raise ValueError("cannot select from an empty deck")
    return int.from_bytes(_digest(value)[:8], "big") % size


def _sentence(value: str) -> str:
    value = " ".join(value.strip().split())
    if not value:
        return value
    return value if value[-1] in ".?!" else f"{value}."


def _join(*parts: str) -> str:
    return " ".join(_sentence(part) for part in parts if part.strip()).strip()


def _semantic_unit_id(
    topic: dict[str, Any],
    context: dict[str, Any],
    intent: dict[str, Any],
    arc: dict[str, Any],
) -> str:
    return ":".join(
        (
            topic["topic_id"],
            context["context_id"],
            intent["intent_id"],
            arc["arc_id"],
        )
    )


def _validation_semantic_units(
    units: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
    ],
    validation_percent: int,
) -> set[str]:
    """Select an exact deterministic percentage of unique semantic units."""

    count = max(1, round(len(units) * validation_percent / 100))
    ranked = sorted(
        (
            _semantic_unit_id(topic, context, intent, arc)
            for topic, context, intent, arc in units
        ),
        key=lambda unit_id: _digest(f"validation:{unit_id}"),
    )
    return set(ranked[:count])


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
    if len(CASUAL_INTENT_CARDS) < 8 or len(CASUAL_ARC_CARDS) < 9:
        raise ValueError("casual semantic reservoirs lack intent or arc capacity")
    for card in CASUAL_INTENT_CARDS:
        if not {"intent_id", "user_opening", "assistant_entry", "closing_focus"} <= set(card):
            raise ValueError(f"incomplete casual intent card: {card.get('intent_id')}")
    for card in CASUAL_ARC_CARDS:
        if not {"arc_id", "user_follow_up", "user_shift", "closing_lens"} <= set(card):
            raise ValueError(f"incomplete casual arc card: {card.get('arc_id')}")


def _messages(
    *,
    topic: dict[str, Any],
    context: dict[str, Any],
    intent: dict[str, Any],
    arc: dict[str, Any],
    decks: dict[str, list[str]],
    unit_index: int,
) -> tuple[list[dict[str, str]], dict[str, int], int, tuple[str, ...]]:
    semantic_unit_id = _semantic_unit_id(topic, context, intent, arc)
    matrix = casual_variable_by(topic, context, intent, arc, decks)
    deal_seed = f"casual:{semantic_unit_id}:{unit_index}"
    dealt = matrix.deal(deal_seed)
    indexes = {
        stage: _stable_index(
            f"{deal_seed}:surface:{stage}",
            len(matrix.cards("surface", stage)),
        )
        for stage in _STAGES
    }
    turn_count = 4 if _stable_index(f"turns:{semantic_unit_id}", 5) < 2 else 6
    opening = _sentence(dealt["surface"]["user_opening"])
    entry = _sentence(dealt["surface"]["assistant_entry"])
    follow_up = _sentence(dealt["surface"]["user_follow_up"])
    assistant_reply = _sentence(dealt["surface"]["assistant_follow_up"])
    closing_depth = _stable_index(f"closing-depth:{semantic_unit_id}", 3)
    indexes["closing_depth"] = closing_depth
    if closing_depth == 0:
        closing = _sentence(dealt["semantic"]["closing_compact"])
    elif closing_depth == 1:
        closing = _sentence(dealt["semantic"]["closing_balanced"])
    else:
        closing = _sentence(dealt["surface"]["assistant_closing"])
    if turn_count == 4:
        messages = [
            {"role": "user", "content": opening},
            {"role": "assistant", "content": entry},
            {"role": "user", "content": follow_up},
            {"role": "assistant", "content": closing},
        ]
    else:
        assistant_reply = _join(
            assistant_reply,
            dealt["topic"]["follow_up_question"],
        )
        shift = _sentence(dealt["surface"]["user_shift"])
        messages = [
            {"role": "user", "content": opening},
            {"role": "assistant", "content": entry},
            {"role": "user", "content": follow_up},
            {"role": "assistant", "content": assistant_reply},
            {"role": "user", "content": shift},
            {"role": "assistant", "content": closing},
        ]
    if len(_sentences(messages[-1]["content"])) > 3:
        raise ValueError("casual final response exceeds three sentences")
    return messages, indexes, turn_count, matrix.field_names()


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text.strip())
        if part.strip()
    )


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
    conversation_quality = audit_casual_conversation_quality(rows)
    if not conversation_quality["passed"]:
        raise ValueError(
            "casual conversation quality gate failed: "
            + "; ".join(conversation_quality["violations"])
        )
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
        "conversation_quality": conversation_quality,
    }


def render_casual_conversation_rows(
    registry_path: Path,
    *,
    examples: int | None = None,
    seed: int = 42,
    validation_percent: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Render the native casual family without creating a separate artifact."""

    if not 1 <= validation_percent <= 25:
        raise ValueError("validation percent must be between 1 and 25")
    registry = json.loads(registry_path.read_text())
    _validate_registry(registry)
    topics = registry["topic_cards"]
    contexts = registry["context_cards"]
    decks = registry["surface_decks"]
    units = [
        (topic, context, intent, arc)
        for topic in topics
        for context in contexts
        for intent in CASUAL_INTENT_CARDS
        for arc in CASUAL_ARC_CARDS
    ]
    units.sort(
        key=lambda unit: _digest(
            f"{seed}:{_semantic_unit_id(*unit)}"
        )
    )
    if examples is None:
        examples = len(units)
    if examples < 100:
        raise ValueError("casual conversation build needs at least 100 examples")
    capacity = len(units)
    if examples > capacity:
        raise ValueError(f"requested {examples} rows exceeds card capacity {capacity}")
    selected_units = units[:examples]
    validation_source_pairs = _validation_semantic_units(
        selected_units,
        validation_percent,
    )

    rows: list[dict[str, Any]] = []
    for unit_index, (topic, context, intent, arc) in enumerate(selected_units):
        source_pair_id = _semantic_unit_id(topic, context, intent, arc)
        messages, indexes, turn_count, variable_by_fields = _messages(
            topic=topic,
            context=context,
            intent=intent,
            arc=arc,
            decks=decks,
            unit_index=unit_index,
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
                f"closing={indexes['assistant_closing']}",
                f"closing_depth={indexes['closing_depth']}",
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
                "split": (
                    "validation" if source_pair_id in validation_source_pairs else "train"
                ),
                "messages": messages,
                "prompt": messages[0]["content"],
                "response": messages[-1]["content"],
                "rendered_text": rendered,
                "source_keys": [
                    topic["topic_id"],
                    context["context_id"],
                    intent["intent_id"],
                    arc["arc_id"],
                ],
                "evidence": [],
                "answer_json": json.dumps(
                    {
                        "conversation_kind": "casual",
                        "source_pair_id": source_pair_id,
                        "topic_card_id": topic["topic_id"],
                        "context_card_id": context["context_id"],
                        "intent_card_id": intent["intent_id"],
                        "arc_card_id": arc["arc_id"],
                        "surface_hand": surface_hand,
                        "response_structure": response_structure,
                        "deck_topology": {
                            "stages": list(_STAGES),
                            "links": [
                                f"{_STAGES[index]}->{_STAGES[index + 1]}"
                                for index in range(len(_STAGES) - 1)
                            ],
                            "variable_by": list(variable_by_fields),
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

    content_digest = hashlib.sha256()
    for row in rows:
        content_digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        )
        content_digest.update(b"\n")
    summary = {
        "format": "casual-conversation-family-v18",
        "purpose": "native casual_conversation training family",
        "seed": seed,
        "counts": {
            "examples": len(rows),
            "by_task": dict(sorted(Counter(row["task"] for row in rows).items())),
            "by_domain": dict(sorted(Counter(row["domain"] for row in rows).items())),
            "by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        },
        "capacity": capacity,
        "semantic_capacity": {
            "topics": len(topics),
            "contexts": len(contexts),
            "intents": len(CASUAL_INTENT_CARDS),
            "arcs": len(CASUAL_ARC_CARDS),
            "product": capacity,
        },
        "audit": audit,
        "registry": {
            "path": str(registry_path),
            "sha256": file_sha256(registry_path),
        },
        "content_sha256": content_digest.hexdigest(),
    }
    return rows, summary
