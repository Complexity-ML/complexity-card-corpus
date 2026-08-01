from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..build import file_sha256
from ..sft import INSTRUCTION_SCHEMA
from .audit import _audit, _counts
from .common import (
    DATASET_ID,
    SURFACE_LICENSE,
    SURFACE_SOURCE,
    SURFACE_VERSION,
    _VARIANT_RADIX,
    _render_messages,
    _split,
)
from .rendering import (
    _adapt_task_blueprint,
    _adapt_task_card,
    _balanced_select,
    _empathy_message_at,
    _empathy_messages,
    _expand_scenario_cards,
    _surface_target_pattern,
    _task_message_at,
    _task_messages,
)


def build_conversation_surface(
    blueprints_root: Path,
    scenarios_path: Path,
    output_root: Path,
    *,
    examples: int = 10_000,
    seed: int = 42,
    validation_percent: int = 5,
) -> dict[str, Any]:
    if examples < 64:
        raise ValueError("examples must be at least 64")
    if not 1 <= validation_percent <= 25:
        raise ValueError("validation_percent must be between 1 and 25")
    blueprint_path = blueprints_root / "blueprints.parquet"
    blueprint_rows = pq.read_table(blueprint_path).to_pylist()
    scenarios = json.loads(scenarios_path.read_text())
    context_cards = scenarios["context_cards"]
    task_cards = _expand_scenario_cards(
        scenarios["task_scenarios"],
        context_cards["task_oriented"],
        kind="task_oriented",
    )
    empathy_cards = _expand_scenario_cards(
        scenarios["empathy_scenarios"],
        context_cards["empathetic_conversation"],
        kind="empathetic_conversation",
    )
    selected = _balanced_select(blueprint_rows, pilot_size=examples, seed=seed)

    rows: list[dict[str, Any]] = []
    used_cards: Counter[str] = Counter()
    used_rendered: set[str] = set()
    used_prompts: set[str] = set()
    used_responses: set[str] = set()
    used_messages: set[str] = set()
    for blueprint, rank in selected:
        kind = blueprint["corpus_kind"]
        category = blueprint["category"]
        if kind == "task_oriented":
            if category not in task_cards:
                raise ValueError(f"missing authored task cards for {category}")
            category_cards = task_cards[category]
            card = _adapt_task_card(
                blueprint, category_cards[rank % len(category_cards)]
            )
            surface_blueprint = _adapt_task_blueprint(blueprint, card)
            phrase_rank = rank // len(category_cards)
            messages, card_id = _task_messages(
                surface_blueprint,
                category_cards,
                rank,
                selected_card=card,
            )

            def render_position(position: int, shift: int) -> str:
                return _task_message_at(
                    surface_blueprint, card, phrase_rank, position, shift
                )

            task = "practical_dialogue"
        else:
            surface_blueprint = blueprint
            if category not in empathy_cards:
                raise ValueError(f"missing authored empathy cards for {category}")
            category_cards = empathy_cards[category]
            card = category_cards[rank % len(category_cards)]
            messages, card_id = _empathy_messages(blueprint, category_cards, rank)

            def render_position(position: int, shift: int) -> str:
                return _empathy_message_at(blueprint, card, rank, position, shift)

            task = "empathetic_dialogue"

        final_position = len(messages) - 1
        current_messages: set[str] = set()
        for position, message in enumerate(messages):
            dedicated = (
                used_prompts
                if position == 0
                else used_responses
                if position == final_position
                else set()
            )
            content = message["content"]
            collision = (
                content in used_messages
                or content in current_messages
                or content in dedicated
            )
            if collision:
                for shift in range(1, _VARIANT_RADIX + 1):
                    candidate = render_position(position, shift)
                    if (
                        candidate not in used_messages
                        and candidate not in current_messages
                        and candidate not in dedicated
                    ):
                        messages[position] = {**message, "content": candidate}
                        content = candidate
                        break
                else:
                    if position in {0, final_position}:
                        for shift in range(1, _VARIANT_RADIX + 1):
                            candidate = render_position(position, shift)
                            if (
                                candidate not in current_messages
                                and candidate not in dedicated
                            ):
                                messages[position] = {
                                    **message,
                                    "content": candidate,
                                }
                                content = candidate
                                break
                        else:
                            if position == 0:
                                raise ValueError(
                                    f"could not render a unique prompt for "
                                    f"{blueprint['blueprint_id']}"
                                )
            current_messages.add(content)

        prompt = messages[0]["content"]
        response = messages[final_position]["content"]

        rendered = _render_messages(messages)
        if rendered in used_rendered:
            raise ValueError(
                f"could not render a unique dialogue for {blueprint['blueprint_id']}"
            )
        used_rendered.add(rendered)
        used_prompts.add(prompt)
        used_responses.add(response)
        used_messages.update(message["content"] for message in messages)
        used_cards[card_id] += 1
        example_id = (
            f"conversation:{hashlib.sha256(rendered.encode()).hexdigest()[:20]}"
        )
        rows.append(
            {
                "example_id": example_id,
                "task": task,
                "mode": "chat" if len(messages) > 2 else "instruct",
                "difficulty": blueprint["difficulty"],
                "dataset_id": DATASET_ID,
                "domain": category,
                "language": "en",
                "split": _split(card_id, validation_percent),
                "messages": messages,
                "prompt": messages[0]["content"],
                "response": messages[-1]["content"],
                "rendered_text": rendered,
                "source_keys": [blueprint["blueprint_id"], card_id],
                "evidence": [],
                "answer_json": json.dumps(
                    {
                        "blueprint_id": blueprint["blueprint_id"],
                        "scenario_card_id": card_id,
                        "dialogue_stages": surface_blueprint["dialogue_stages"],
                        "response_style": blueprint["response_style"],
                        "target_length_pattern": _surface_target_pattern(
                            surface_blueprint, card, kind=kind
                        ),
                        "target_question_turns": surface_blueprint[
                            "target_question_turns"
                        ],
                    },
                    sort_keys=True,
                ),
                "source": SURFACE_SOURCE,
                "source_urls": [],
                "license": SURFACE_LICENSE,
                "version": "1.0.0",
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
    )
    audit_path = temporary / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = {
        "format": SURFACE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "human-authored general conversation surface dataset",
        "surface_text": {
            "authorship": "Complexity original scenario cards and phrase libraries",
            "model_generated": False,
            "source_utterances_accessed": False,
            "license": SURFACE_LICENSE,
        },
        "seed": seed,
        "examples": examples,
        "validation_percent": validation_percent,
        "inputs": {
            "blueprints": {
                "path": str(blueprint_path),
                "sha256": file_sha256(blueprint_path),
            },
            "scenarios": {
                "path": str(scenarios_path),
                "sha256": file_sha256(scenarios_path),
            },
        },
        "counts": {
            "examples": len(rows),
            "by_task": _counts(rows, lambda row: row["task"]),
            "by_domain": _counts(rows, lambda row: row["domain"]),
            "by_mode": _counts(rows, lambda row: row["mode"]),
            "scenario_cards_used": len(used_cards),
        },
        "scenario_usage": dict(sorted(used_cards.items())),
        "audit": audit,
        "files": {
            "conversations.parquet": {
                "bytes": output_path.stat().st_size,
                "sha256": file_sha256(output_path),
            },
            "audit.json": {
                "bytes": audit_path.stat().st_size,
                "sha256": file_sha256(audit_path),
            },
        },
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return manifest


def build_conversation_surface_pilot(
    blueprints_root: Path,
    scenarios_path: Path,
    output_root: Path,
    *,
    pilot_size: int = 512,
    seed: int = 42,
    validation_percent: int = 5,
) -> dict[str, Any]:
    """Backward-compatible entry point for earlier 512-row pilot commands."""
    return build_conversation_surface(
        blueprints_root,
        scenarios_path,
        output_root,
        examples=pilot_size,
        seed=seed,
        validation_percent=validation_percent,
    )
