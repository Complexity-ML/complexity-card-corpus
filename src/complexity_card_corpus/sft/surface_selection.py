from __future__ import annotations

import hashlib
import os
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

from ..training_cards import TrainingCards
from .language import _render_messages


SURFACE_HAND_CANDIDATES_PER_EXAMPLE = 32

CardDealer = Callable[[dict[str, Any], str], TrainingCards]
Projector = Callable[[dict[str, Any], str], tuple[list[dict[str, str]], TrainingCards]]


def select_balanced_sft_surfaces(
    rows: list[dict[str, Any]],
    *,
    dealer: CardDealer,
    projector: Projector,
    candidates_per_example: int = SURFACE_HAND_CANDIDATES_PER_EXAMPLE,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Balance existing response-card hands before rendering each example.

    Candidate dealing is cheap: only the current card combinations are hashed.
    The selected key is rendered once, so this avoids multiplying full prompt
    and response generation. No semantic card or surface axis is introduced.
    """

    if candidates_per_example < 1:
        raise ValueError("surface hand candidate count must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] == "train":
            by_task[row["task"]].append(row)
        else:
            passthrough.append(row)

    arguments = [
        (task, task_rows, dealer, projector, candidates_per_example)
        for task, task_rows in sorted(by_task.items())
    ]
    if workers > 1 and len(arguments) > 1:
        worker_count = min(workers, len(arguments), os.cpu_count() or 1)

        def collect(executor: ProcessPoolExecutor | ThreadPoolExecutor):
            return list(executor.map(_select_task_surfaces, arguments))

        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                results = collect(executor)
        except (NotImplementedError, OSError, PermissionError):
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = collect(executor)
        execution = "parallel_by_task"
    else:
        worker_count = 1
        results = [_select_task_surfaces(argument) for argument in arguments]
        execution = "serial_by_task"

    selected = [row for task_rows, _audit in results for row in task_rows]
    task_audit = {
        task: stats
        for _task_rows, audit in results
        for task, stats in audit.items()
    }

    for row in passthrough:
        messages, cards = projector(row, row["example_id"])
        selected.append(
            {
                **row,
                "_projected_messages": messages,
                "_projected_prompt": _render_messages(messages[:-1]),
                "_projected_target": messages[-1]["content"],
                "_conditioning_cards": cards,
            }
        )
    selected.sort(key=lambda row: row["example_id"])
    return selected, {
        "method": "least_used_existing_response_card_hand",
        "execution": execution,
        "workers": worker_count,
        "candidates_per_training_example": candidates_per_example,
        "full_renders_per_example": 1,
        "new_card_axes": 0,
        "tasks": task_audit,
    }


def _select_task_surfaces(
    arguments: tuple[str, list[dict[str, Any]], CardDealer, Projector, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task, task_rows, dealer, projector, candidates_per_example = arguments
    selected: list[dict[str, Any]] = []
    hand_counts: Counter[str] = Counter()
    candidate_hands: set[str] = set()
    for row in sorted(
        task_rows,
        key=lambda item: hashlib.sha256(
            f"surface-row:{task}:{item['example_id']}".encode()
        ).digest(),
    ):
        options: list[tuple[int, bytes, str]] = []
        for candidate_index in range(candidates_per_example):
            selection_key = f"{row['example_id']}:surface-candidate:{candidate_index}"
            cards = dealer(row, selection_key)
            hand = cards.response_structure_signature
            candidate_hands.add(hand)
            options.append(
                (
                    hand_counts[hand],
                    hashlib.sha256(
                        f"surface-choice:{selection_key}".encode()
                    ).digest(),
                    selection_key,
                )
            )
        _, _, selection_key = min(options)
        messages, cards = projector(row, selection_key)
        hand_counts[cards.response_structure_signature] += 1
        selected.append(
            {
                **row,
                "_projected_messages": messages,
                "_projected_prompt": _render_messages(messages[:-1]),
                "_projected_target": messages[-1]["content"],
                "_conditioning_cards": cards,
            }
        )
    return selected, {
        task: {
            "examples": len(task_rows),
            "candidate_count": candidates_per_example,
            "candidate_response_card_hands": len(candidate_hands),
            "selected_response_card_hands": len(hand_counts),
            "maximum_selected_hand_share": round(
                max(hand_counts.values(), default=0) / len(task_rows), 6
            ),
        }
    }
