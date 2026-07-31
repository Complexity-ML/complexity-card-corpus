from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

from .build import file_sha256


BLUEPRINT_VERSION = "conversation-blueprint-v1"

BLUEPRINT_SCHEMA = pa.schema(
    [
        ("blueprint_id", pa.string()),
        ("source_structure_id", pa.string()),
        ("source_record_id", pa.string()),
        ("source_record_sha256", pa.string()),
        ("source_dataset", pa.string()),
        ("source_revision", pa.string()),
        ("source_license", pa.string()),
        ("source_file_sha256", pa.string()),
        ("corpus_kind", pa.string()),
        ("category", pa.string()),
        ("domain", pa.string()),
        ("emotion", pa.string()),
        ("split", pa.string()),
        ("target_turn_count", pa.int32()),
        ("target_question_turns", pa.int32()),
        ("target_speaker_pattern", pa.list_(pa.string())),
        ("target_length_pattern", pa.list_(pa.string())),
        ("dialogue_stages", pa.list_(pa.string())),
        ("response_style", pa.string()),
        ("difficulty", pa.string()),
        ("source_slot_types", pa.list_(pa.string())),
        ("surface_text_generated", pa.bool_()),
        ("blueprint_version", pa.string()),
    ]
)

_PROSE_COLUMNS = {"text", "utterance", "prompt", "response", "messages"}


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _stable_int(value: str) -> int:
    return int.from_bytes(_digest(value)[:8], "big")


def _category(row: dict[str, Any]) -> str:
    if row["corpus_kind"] == "task_oriented":
        return row["domain"]
    if row["corpus_kind"] == "empathetic_conversation":
        return row["emotion"]
    raise ValueError(f"unsupported corpus kind: {row['corpus_kind']}")


def _uniform_capacity(rows: list[dict[str, Any]]) -> int:
    groups = Counter(_category(row) for row in rows)
    if not groups:
        return 0
    return min(groups.values()) * len(groups)


def _balanced_select(
    rows: list[dict[str, Any]],
    *,
    target: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_category(row)].append(row)
    keys = sorted(groups)
    if not keys:
        return []
    base, remainder = divmod(target, len(keys))
    rotated = sorted(keys, key=lambda key: _digest(f"{seed}:category:{key}"))
    quotas = {key: base + int(key in set(rotated[:remainder])) for key in keys}
    selected: list[dict[str, Any]] = []
    for key in keys:
        candidates = sorted(
            groups[key],
            key=lambda row: _digest(f"{seed}:{row['record_id']}"),
        )
        if len(candidates) < quotas[key]:
            raise ValueError(
                f"category {key} has {len(candidates)} rows but needs {quotas[key]}"
            )
        selected.extend(candidates[: quotas[key]])
    return selected


def _target_turn_count(key: str) -> int:
    bucket = _stable_int(f"turns:{key}") % 20
    if bucket < 8:
        return 2
    if bucket < 16:
        return 4
    if bucket < 19:
        return 6
    return 8


def _task_stages(turn_count: int, has_rejection: bool) -> list[str]:
    user_stages = ("state_goal", "provide_detail", "choose_option", "confirm_choice")
    assistant_stages = (
        "acknowledge_goal",
        "ask_for_missing_detail",
        "present_bounded_options",
        "confirm_next_step",
    )
    stages = []
    for turn in range(turn_count):
        position = turn // 2
        sequence = user_stages if turn % 2 == 0 else assistant_stages
        stages.append(sequence[min(position, len(sequence) - 1)])
    if has_rejection and turn_count <= 4:
        stages[-1] = "offer_safe_alternative"
    elif has_rejection and turn_count == 6:
        stages = [
            "state_goal",
            "acknowledge_goal",
            "provide_detail",
            "offer_safe_alternative",
            "confirm_choice",
            "confirm_next_step",
        ]
    return stages


def _empathetic_stages(turn_count: int) -> list[str]:
    user_stages = ("share_situation", "expand_feeling", "reflect_on_need", "follow_up")
    assistant_stages = (
        "acknowledge_emotion",
        "invite_detail_without_assumption",
        "offer_grounded_support",
        "close_supportively",
    )
    stages = []
    for turn in range(turn_count):
        position = turn // 2
        sequence = user_stages if turn % 2 == 0 else assistant_stages
        stages.append(sequence[min(position, len(sequence) - 1)])
    return stages


def _pick_style(kind: str, key: str) -> str:
    choices = {
        "task_oriented": (
            "concise_practical",
            "clear_confirming",
            "stepwise_helpful",
        ),
        "empathetic_conversation": (
            "warm_grounded",
            "calm_supportive",
            "concise_empathetic",
        ),
    }[kind]
    return choices[_stable_int(f"style:{key}") % len(choices)]


def _split(blueprint_id: str, validation_percent: int) -> str:
    return "validation" if _stable_int(f"split:{blueprint_id}") % 100 < validation_percent else "train"


def _blueprint(row: dict[str, Any], *, seed: int, validation_percent: int) -> dict[str, Any]:
    source_id = row["record_id"]
    blueprint_id = f"conversation-blueprint:{hashlib.sha256(f'{seed}:{source_id}'.encode()).hexdigest()[:20]}"
    turn_count = _target_turn_count(blueprint_id)
    kind = row["corpus_kind"]
    if kind == "task_oriented":
        source_has_rejection = any(
            "reject" in signal for signal in row["turn_signal_sequence"]
        )
        has_rejection = (
            source_has_rejection
            and _stable_int(f"rejection-variant:{blueprint_id}") % 3 == 0
        )
        stages = _task_stages(turn_count, has_rejection)
    else:
        stages = _empathetic_stages(turn_count)
    question_turns = sum(
        "ask" in stage or "invite" in stage for stage in stages
    )
    return {
        "blueprint_id": blueprint_id,
        "source_structure_id": source_id,
        "source_record_id": row["source_record_id"],
        "source_record_sha256": row["source_record_sha256"],
        "source_dataset": row["source_dataset"],
        "source_revision": row["source_revision"],
        "source_license": row["source_license"],
        "source_file_sha256": row["source_file_sha256"],
        "corpus_kind": kind,
        "category": _category(row),
        "domain": row["domain"],
        "emotion": row["emotion"],
        "split": _split(blueprint_id, validation_percent),
        "target_turn_count": turn_count,
        "target_question_turns": question_turns,
        "target_speaker_pattern": [
            "user" if position % 2 == 0 else "assistant"
            for position in range(turn_count)
        ],
        "target_length_pattern": ["medium"] * turn_count,
        "dialogue_stages": stages,
        "response_style": _pick_style(kind, blueprint_id),
        "difficulty": "easy" if turn_count <= 4 else "medium",
        "source_slot_types": row["slot_types"],
        "surface_text_generated": False,
        "blueprint_version": BLUEPRINT_VERSION,
    }


def _counts(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any]) -> dict[str, int]:
    return dict(sorted((str(item), count) for item, count in Counter(key(row) for row in rows).items()))


def _audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if set(BLUEPRINT_SCHEMA.names) & _PROSE_COLUMNS:
        raise ValueError("blueprint schema must not contain prose columns")
    if any(row["surface_text_generated"] for row in rows):
        raise ValueError("surface text must remain disabled during blueprint construction")
    if len({row["blueprint_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate blueprint IDs")
    if len({row["source_structure_id"] for row in rows}) != len(rows):
        raise ValueError("one source structure may feed only one blueprint")
    for row in rows:
        turns = row["target_turn_count"]
        if turns not in {2, 4, 6, 8}:
            raise ValueError("unsupported target turn count")
        if not (
            len(row["target_speaker_pattern"])
            == len(row["target_length_pattern"])
            == len(row["dialogue_stages"])
            == turns
        ):
            raise ValueError("blueprint sequences must match target turn count")
        if row["target_speaker_pattern"][0] != "user" or row["target_speaker_pattern"][-1] != "assistant":
            raise ValueError("blueprints must start with user and end with assistant")

    by_kind = Counter(row["corpus_kind"] for row in rows)
    category_balance: dict[str, Any] = {}
    for kind in sorted(by_kind):
        counts = Counter(row["category"] for row in rows if row["corpus_kind"] == kind)
        category_balance[kind] = {
            "categories": len(counts),
            "minimum": min(counts.values()),
            "maximum": max(counts.values()),
            "spread": max(counts.values()) - min(counts.values()),
        }
    return {
        "prose_columns": [],
        "surface_text_rows": 0,
        "unique_blueprints": len(rows),
        "unique_source_structures": len(rows),
        "kind_balance": dict(sorted(by_kind.items())),
        "category_balance": category_balance,
        "turn_count_distribution": _counts(rows, lambda row: row["target_turn_count"]),
        "style_distribution": _counts(rows, lambda row: row["response_style"]),
        "split_distribution": _counts(rows, lambda row: row["split"]),
    }


def build_conversation_blueprints(
    mine_root: Path,
    output_root: Path,
    *,
    target_per_kind: int | None = None,
    seed: int = 42,
    validation_percent: int = 5,
) -> dict[str, Any]:
    if target_per_kind is not None and target_per_kind < 1:
        raise ValueError("target_per_kind must be positive")
    if not 1 <= validation_percent <= 25:
        raise ValueError("validation_percent must be between 1 and 25")
    mine_path = mine_root / "raw_records.parquet"
    source_rows = pq.read_table(mine_path).to_pylist()
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_kind[row["corpus_kind"]].append(row)
    required_kinds = {"task_oriented", "empathetic_conversation"}
    if set(by_kind) != required_kinds:
        raise ValueError(f"expected exactly these source kinds: {sorted(required_kinds)}")
    uniform_capacities = {kind: _uniform_capacity(rows) for kind, rows in by_kind.items()}
    target = target_per_kind or min(uniform_capacities.values())
    if any(target > capacity for capacity in uniform_capacities.values()):
        raise ValueError(
            f"target_per_kind={target} exceeds uniform capacities {uniform_capacities}"
        )

    selected: list[dict[str, Any]] = []
    for kind in sorted(required_kinds):
        selected.extend(_balanced_select(by_kind[kind], target=target, seed=seed))
    rows = [
        _blueprint(row, seed=seed, validation_percent=validation_percent)
        for row in selected
    ]
    rows.sort(key=lambda row: row["blueprint_id"])
    audit = _audit(rows)

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    blueprints_path = temporary / "blueprints.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=BLUEPRINT_SCHEMA),
        blueprints_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    manifest = {
        "format": BLUEPRINT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "balanced conversation generation plan; no surface text",
        "generation_enabled": False,
        "seed": seed,
        "validation_percent": validation_percent,
        "source_mine": {
            "path": str(mine_path),
            "sha256": file_sha256(mine_path),
            "records": len(source_rows),
        },
        "uniform_capacities": uniform_capacities,
        "target_per_kind": target,
        "counts": {
            "blueprints": len(rows),
            "by_kind": _counts(rows, lambda row: row["corpus_kind"]),
            "by_category": _counts(rows, lambda row: f"{row['corpus_kind']}:{row['category']}"),
        },
        "audit": audit,
        "files": {
            "blueprints.parquet": {
                "bytes": blueprints_path.stat().st_size,
                "sha256": file_sha256(blueprints_path),
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
