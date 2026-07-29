from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from .build import file_sha256
from .models import AlignmentCard, ChatMessage

REPO_ID = "OpenAssistant/oasst1"
REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
LICENSE = "Apache-2.0"
FILES = {
    "train": "data/train-00000-of-00001-b42a775f407cee45.parquet",
    "validation": "data/validation-00000-of-00001-134b8fd0c89408b6.parquet",
}
SOURCE_URL = "https://huggingface.co/datasets/OpenAssistant/oasst1"

MESSAGE_SCHEMA = pa.list_(
    pa.struct(
        [
            ("role", pa.string()),
            ("content", pa.string()),
            ("source_message_id", pa.string()),
        ]
    )
)
ALIGNMENT_SCHEMA = pa.schema(
    [
        ("example_id", pa.string()),
        ("mode", pa.string()),
        ("split", pa.string()),
        ("language", pa.string()),
        ("messages", MESSAGE_SCHEMA),
        ("instruction", pa.string()),
        ("response", pa.string()),
        ("rendered_text", pa.string()),
        ("quality_score", pa.float32()),
        ("source_dataset", pa.string()),
        ("source_revision", pa.string()),
        ("source_tree_id", pa.string()),
        ("source_message_ids", pa.list_(pa.string())),
        ("source_url", pa.string()),
        ("license", pa.string()),
    ]
)


@dataclass(frozen=True)
class Node:
    message_id: str
    parent_id: str | None
    tree_id: str
    role: str
    text: str
    rank: int | None
    review_count: int
    quality: float
    helpfulness: float
    labels: dict[str, float]


def _label_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    names = value.get("name")
    values = value.get("value")
    if names is None or values is None:
        return {}
    return {
        str(name): float(score)
        for name, score in zip(names, values)
        if score is not None and not (isinstance(score, float) and math.isnan(score))
    }


def _rank(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(value)


def _eligible(row: dict[str, Any], *, max_characters: int) -> tuple[bool, str]:
    text = str(row.get("text") or "").strip()
    if str(row.get("lang")) != "en":
        return False, "language"
    if str(row.get("tree_state")) != "ready_for_export":
        return False, "tree_state"
    if bool(row.get("deleted")):
        return False, "deleted"
    if bool(row.get("synthetic")):
        return False, "synthetic"
    if row.get("review_result") is not True and not bool(row.get("review_result")):
        return False, "review"
    if len(text) < 3 or len(text) > max_characters:
        return False, "length"
    if row.get("role") not in {"prompter", "assistant"}:
        return False, "role"

    detoxify = row.get("detoxify") or {}
    if float(detoxify.get("toxicity") or 0.0) > 0.5:
        return False, "toxicity"

    labels = _label_map(row.get("labels"))
    for label in (
        "spam",
        "lang_mismatch",
        "pii",
        "not_appropriate",
        "hate_speech",
        "sexual_content",
    ):
        if labels.get(label, 0.0) > 0.5:
            return False, label
    if row.get("role") == "assistant":
        if labels.get("fails_task", 0.0) > 0.5:
            return False, "fails_task"
        if labels.get("quality", 0.5) < 0.5:
            return False, "low_quality"
        if labels.get("helpfulness", labels.get("quality", 0.5)) < 0.5:
            return False, "low_helpfulness"
    return True, "accepted"


def _node(row: dict[str, Any]) -> Node:
    labels = _label_map(row.get("labels"))
    parent_id = row.get("parent_id")
    if parent_id is not None and isinstance(parent_id, float) and math.isnan(parent_id):
        parent_id = None
    return Node(
        message_id=str(row["message_id"]),
        parent_id=str(parent_id) if parent_id else None,
        tree_id=str(row["message_tree_id"]),
        role="user" if row["role"] == "prompter" else "assistant",
        text=str(row["text"]).strip(),
        rank=_rank(row.get("rank")),
        review_count=int(row.get("review_count") or 0),
        quality=max(0.0, min(1.0, float(labels.get("quality", 0.5)))),
        helpfulness=max(
            0.0,
            min(1.0, float(labels.get("helpfulness", labels.get("quality", 0.5)))),
        ),
        labels=labels,
    )


def _assistant_sort_key(node: Node) -> tuple[int, float, float, int, str]:
    return (
        node.rank if node.rank is not None else 1_000_000,
        -node.helpfulness,
        -node.quality,
        -node.review_count,
        node.message_id,
    )


def _user_sort_key(node: Node) -> tuple[float, int, str]:
    return (-node.quality, -node.review_count, node.message_id)


def _best_assistant(
    node: Node,
    children: dict[str, list[Node]],
) -> Node | None:
    candidates = [child for child in children.get(node.message_id, []) if child.role == "assistant"]
    return min(candidates, key=_assistant_sort_key) if candidates else None


def _best_followup(
    node: Node,
    children: dict[str, list[Node]],
) -> Node | None:
    candidates = [
        child
        for child in children.get(node.message_id, [])
        if child.role == "user" and _best_assistant(child, children) is not None
    ]
    return min(candidates, key=_user_sort_key) if candidates else None


def _render(messages: list[ChatMessage]) -> str:
    labels = {"user": "User", "assistant": "Assistant"}
    return "\n\n".join(
        f"{labels[message.role]}:\n{message.content}" for message in messages
    )


def _alignment_card(
    *,
    mode: str,
    split: str,
    tree_id: str,
    nodes: list[Node],
) -> AlignmentCard:
    messages = [
        ChatMessage(
            role=node.role,
            content=node.text,
            source_message_id=node.message_id,
        )
        for node in nodes
    ]
    assistant_nodes = [node for node in nodes if node.role == "assistant"]
    quality = sum(
        (node.quality + node.helpfulness) / 2.0 for node in assistant_nodes
    ) / len(assistant_nodes)
    digest = hashlib.sha256(
        "\0".join(node.message_id for node in nodes).encode()
    ).hexdigest()[:16]
    return AlignmentCard(
        example_id=f"oasst1:{mode}:{tree_id}:{digest}",
        mode=mode,
        split=split,
        messages=messages,
        rendered_text=_render(messages),
        quality_score=quality,
        source_dataset=REPO_ID,
        source_revision=REVISION,
        source_tree_id=tree_id,
        license=LICENSE,
    )


def build_alignment_cards(
    frame: pd.DataFrame,
    *,
    split: str,
    max_characters: int = 16_000,
    max_messages: int = 10,
) -> tuple[list[AlignmentCard], Counter]:
    rejection_counts: Counter = Counter()
    accepted: dict[str, Node] = {}
    for row in frame.to_dict(orient="records"):
        eligible, reason = _eligible(row, max_characters=max_characters)
        rejection_counts[reason] += 1
        if eligible:
            node = _node(row)
            accepted[node.message_id] = node

    children: dict[str, list[Node]] = defaultdict(list)
    for node in accepted.values():
        if node.parent_id and node.parent_id in accepted:
            children[node.parent_id].append(node)

    roots = sorted(
        (
            node
            for node in accepted.values()
            if node.role == "user" and node.parent_id is None
        ),
        key=lambda node: (node.tree_id, node.message_id),
    )
    cards: list[AlignmentCard] = []
    for root in roots:
        first_assistant = _best_assistant(root, children)
        if first_assistant is None:
            rejection_counts["root_without_assistant"] += 1
            continue
        cards.append(
            _alignment_card(
                mode="instruct",
                split=split,
                tree_id=root.tree_id,
                nodes=[root, first_assistant],
            )
        )

        path = [root, first_assistant]
        current = first_assistant
        while len(path) + 2 <= max_messages:
            followup = _best_followup(current, children)
            if followup is None:
                break
            assistant = _best_assistant(followup, children)
            if assistant is None:
                break
            path.extend([followup, assistant])
            current = assistant
        if len(path) >= 4:
            cards.append(
                _alignment_card(
                    mode="chat",
                    split=split,
                    tree_id=root.tree_id,
                    nodes=path,
                )
            )

    return cards, rejection_counts


def _rows(cards: list[AlignmentCard]) -> list[dict[str, Any]]:
    rows = []
    for card in cards:
        messages = [
            message.model_dump(mode="json")
            for message in card.messages
        ]
        rows.append(
            {
                "example_id": card.example_id,
                "mode": card.mode,
                "split": card.split,
                "language": card.language,
                "messages": messages,
                "instruction": messages[0]["content"],
                "response": messages[-1]["content"],
                "rendered_text": card.rendered_text,
                "quality_score": card.quality_score,
                "source_dataset": card.source_dataset,
                "source_revision": card.source_revision,
                "source_tree_id": card.source_tree_id,
                "source_message_ids": [
                    message["source_message_id"] for message in messages
                ],
                "source_url": SOURCE_URL,
                "license": card.license,
            }
        )
    return sorted(rows, key=lambda row: row["example_id"])


def _write(rows: list[dict[str, Any]], path: Path) -> None:
    table = pa.Table.from_pylist(rows, schema=ALIGNMENT_SCHEMA)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


def download_snapshot(raw_root: Path) -> dict[str, Path]:
    paths = {}
    for split, filename in FILES.items():
        path = hf_hub_download(
            REPO_ID,
            filename,
            repo_type="dataset",
            revision=REVISION,
            local_dir=raw_root,
        )
        paths[split] = Path(path)
    return paths


def import_oasst1(
    raw_root: Path,
    output_root: Path,
    *,
    download: bool = True,
    max_characters: int = 16_000,
    max_messages: int = 10,
) -> dict[str, Any]:
    paths = (
        download_snapshot(raw_root)
        if download
        else {split: raw_root / filename for split, filename in FILES.items()}
    )
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    cards: list[AlignmentCard] = []
    rejection_by_split: dict[str, dict[str, int]] = {}
    for split, path in paths.items():
        frame = pd.read_parquet(path)
        split_cards, rejections = build_alignment_cards(
            frame,
            split=split,
            max_characters=max_characters,
            max_messages=max_messages,
        )
        cards.extend(split_cards)
        rejection_by_split[split] = dict(sorted(rejections.items()))

    output_root.mkdir(parents=True, exist_ok=True)
    all_rows = _rows(cards)
    instruct_rows = [row for row in all_rows if row["mode"] == "instruct"]
    chat_rows = [row for row in all_rows if row["mode"] == "chat"]
    paths_out = {
        "alignment": output_root / "alignment.parquet",
        "instruct": output_root / "instruct.parquet",
        "chat": output_root / "chat.parquet",
    }
    _write(all_rows, paths_out["alignment"])
    _write(instruct_rows, paths_out["instruct"])
    _write(chat_rows, paths_out["chat"])

    manifest = {
        "format": "complexity-alignment-cards-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "license": LICENSE,
            "url": SOURCE_URL,
            "files": {
                split: {
                    "path": FILES[split],
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for split, path in paths.items()
            },
        },
        "filters": {
            "language": "en",
            "review_result": True,
            "synthetic": False,
            "maximum_message_characters": max_characters,
            "maximum_chat_messages": max_messages,
            "maximum_detoxify_toxicity": 0.5,
            "minimum_assistant_quality": 0.5,
            "minimum_assistant_helpfulness": 0.5,
        },
        "counts": {
            "alignment": len(all_rows),
            "instruct": len(instruct_rows),
            "chat": len(chat_rows),
            "by_split": dict(sorted(Counter(row["split"] for row in all_rows).items())),
            "rejections": rejection_by_split,
        },
        "files": {
            name: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in paths_out.items()
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
