from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..english_morphology import correct_indefinite_articles
from ..tasks import TaskHand, deal_task_hand
from .constants import (
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_SOURCE,
    _ACKNOWLEDGEMENTS,
    _CHAT_OPENINGS,
    _INSTRUCT_OPENINGS,
    _INTENT_FIELD,
    _INTENT_SUBJECT_TEMPLATES,
    _WORD,
)


def _stable_index(value: str, size: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % size


def _intent(payload: dict[str, str], family: str) -> str:
    return payload[_INTENT_FIELD[family]].rstrip(".")


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


def _render_card_prompt(
    row: dict[str, Any], hand: TaskHand, *, include_situation: bool
) -> str:
    cards: list[str] = []
    if include_situation:
        situation_title = hand.situation_title or row["title"]
        situation = hand.situation or row["trigger"]
        cards.append(f"SITUATION CARD\n{situation_title}\n{situation}")
    cards.extend(
        (
            f"DATA CARD\n{hand.data}",
            f"RULE CARD\n{hand.rule or row['constraint']}",
            f"GOAL CARD\n{hand.goal}",
        )
    )
    return "\n\n".join(cards)


def _render_messages(
    row: dict[str, Any], variant: int, hand: TaskHand | None = None
) -> list[dict[str, str]]:
    hand = hand or deal_task_hand(row, variant)
    if variant % 2 == 0:
        opening = _INSTRUCT_OPENINGS[
            _stable_index(
                f"instruct-opening:{row['scenario_id']}:{variant}",
                len(_INSTRUCT_OPENINGS),
            )
        ]
        return [
            {
                "role": "user",
                "content": correct_indefinite_articles(
                    opening
                    + "\n\n"
                    + _render_card_prompt(row, hand, include_situation=True)
                ),
            },
            {
                "role": "assistant",
                "content": hand.answer,
            },
        ]

    acknowledgement = _ACKNOWLEDGEMENTS[
        _stable_index(f"ack:{row['scenario_id']}:{variant}", len(_ACKNOWLEDGEMENTS))
    ]
    opening = _CHAT_OPENINGS[
        _stable_index(
            f"chat-opening:{row['scenario_id']}:{variant}", len(_CHAT_OPENINGS)
        )
    ]
    situation_title = hand.situation_title or row["title"]
    situation = hand.situation or row["trigger"]
    chat_opening = (
        f"{opening}\n\n"
        f"SITUATION CARD\n{situation_title}\n{situation}"
        f"\n\nDATA CARD\n{hand.data}"
    )
    follow_up = f"RULE CARD\n{hand.rule or row['constraint']}\n\nGOAL CARD\n{hand.goal}"
    return [
        {
            "role": "user",
            "content": correct_indefinite_articles(chat_opening),
        },
        {"role": "assistant", "content": acknowledgement},
        {"role": "user", "content": correct_indefinite_articles(follow_up)},
        {"role": "assistant", "content": hand.answer},
    ]


def _render_transcript(messages: list[dict[str, str]]) -> str:
    labels = {"user": "User", "assistant": "Assistant"}
    return "\n".join(
        f"{labels[message['role']]}: {message['content']}" for message in messages
    )


def _conversation_rows(
    scenarios: list[dict[str, Any]],
    variants_per_scenario: int,
    vocabulary_placements: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if vocabulary_placements:
        scenarios = _apply_vocabulary_placements(scenarios, vocabulary_placements)
    for scenario in scenarios:
        for variant in range(variants_per_scenario):
            hand = deal_task_hand(scenario, variant)
            messages = _render_messages(scenario, variant, hand)
            rendered = _render_transcript(messages)
            mode = "instruct" if len(messages) == 2 else "chat"
            payload = json.loads(scenario["semantic_payload"])
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
                "state": hand.situation or scenario["state"],
                "source_state": scenario["state"],
                "constraint": hand.rule or scenario["constraint"],
                "source_constraint": scenario["constraint"],
                "desired_outcome": scenario["desired_outcome"],
                "fallback": scenario["fallback"],
                "subject": payload["subject"],
                "surface_intent": _intent(payload, scenario["family"]),
                "domain_context": payload["domain_context"],
                "fallback_surface": scenario["fallback"],
                "response_contract": scenario["response_contract"],
                "variant": variant,
                "mode": mode,
                "card_hand": {
                    "cards": ["situation", "data", "rule", "goal"],
                    "completion_contract": list(hand.contract),
                },
                "model_generated_dialogue": False,
                "lexical_focus": scenario.get("lexical_focus", ""),
                "lexical_assignment_method": scenario.get(
                    "lexical_assignment_method", ""
                ),
            }
            rows.append(
                {
                    "example_id": f"post-training:{suffix}",
                    "task": scenario["family"],
                    "mode": mode,
                    "difficulty": (
                        "hard"
                        if scenario["risk_level"] in {"high", "critical"}
                        else (
                            "easy"
                            if variant % 4 in {0, 1}
                            else ("hard" if variant % 4 == 3 else "medium")
                        )
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
    deduplicated: list[dict[str, Any]] = []
    seen_transcripts: set[str] = set()
    seen_responses: set[str] = set()
    ranked_rows = sorted(
        rows,
        key=lambda item: (
            not bool(json.loads(item["answer_json"])["lexical_focus"]),
            item["example_id"],
        ),
    )
    for row in ranked_rows:
        transcript = row["rendered_text"]
        response = row["response"]
        if transcript in seen_transcripts or response in seen_responses:
            continue
        seen_transcripts.add(transcript)
        seen_responses.add(response)
        deduplicated.append(row)
    return sorted(deduplicated, key=lambda item: item["example_id"])


def _balance_conversation_families(
    rows: list[dict[str, Any]], *, max_examples_per_family: int = 5_000
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cap dominant families after exact response deduplication."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["task"]].append(row)
    before = dict(sorted((task, len(items)) for task, items in buckets.items()))
    kept: list[dict[str, Any]] = []
    for task, items in sorted(buckets.items()):
        ranked = sorted(
            items,
            key=lambda item: (
                not bool(json.loads(item["answer_json"])["lexical_focus"]),
                hashlib.sha256(
                    f"post-training-balance:{task}:{item['example_id']}".encode()
                ).digest(),
            ),
        )
        kept.extend(ranked[:max_examples_per_family])
    kept.sort(key=lambda item: item["example_id"])
    after = dict(sorted(Counter(row["task"] for row in kept).items()))
    return kept, {
        "before": before,
        "after": after,
        "maximum_examples_per_family": max_examples_per_family,
        "dropped": len(rows) - len(kept),
    }


def _load_vocabulary_placements(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("vocabulary placement contains no rows")
    tokens = [row["token"] for row in rows]
    if len(tokens) != len(set(tokens)):
        raise ValueError("vocabulary placement contains duplicate tokens")
    if any(
        _WORD.fullmatch(token) is None or token != token.lower() for token in tokens
    ):
        raise ValueError("vocabulary placement tokens must be normalized words")
    if any(row["family"] not in _INTENT_FIELD for row in rows):
        raise ValueError("vocabulary placement contains an unknown family")
    if any(not row.get("domain") for row in rows):
        raise ValueError("vocabulary placement must include a target domain")
    if any(row["surface_policy"] != "grounded_quoted_term" for row in rows):
        raise ValueError("vocabulary placement must use grounded quoted terms")
    return rows


def _apply_vocabulary_placements(
    scenarios: list[dict[str, Any]], placements: list[dict[str, str]]
) -> list[dict[str, Any]]:
    scenarios_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    placements_by_cell: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for scenario in scenarios:
        scenarios_by_cell[(scenario["family"], scenario["domain"])].append(scenario)
    for placement in placements:
        placements_by_cell[(placement["family"], placement["domain"])].append(placement)

    assigned: dict[str, dict[str, str]] = {}
    cell_offsets: Counter[tuple[str, str]] = Counter()
    overflow: list[dict[str, str]] = []
    for cell, cell_placements in placements_by_cell.items():
        cell_scenarios = sorted(
            scenarios_by_cell[cell],
            key=lambda row: hashlib.sha256(
                f"vocabulary-scenario:{row['scenario_id']}".encode()
            ).digest(),
        )
        ordered_placements = sorted(
            cell_placements,
            key=lambda row: hashlib.sha256(
                f"vocabulary-token:{row['token']}".encode()
            ).digest(),
        )
        primary_count = min(len(cell_scenarios), len(ordered_placements))
        for scenario, placement in zip(
            cell_scenarios[:primary_count], ordered_placements[:primary_count]
        ):
            assigned[scenario["scenario_id"]] = placement
        cell_offsets[cell] = primary_count
        overflow.extend(ordered_placements[primary_count:])

    # Rebalancing the scenario registry must not silently drop vocabulary.
    # Keep every statistically selected cell when it has capacity, then move
    # only the overflow to a documented alternative context. If all recorded
    # alternatives are full, stay inside the same task family and choose its
    # least-filled domain deterministically.
    for placement in sorted(
        overflow,
        key=lambda row: hashlib.sha256(
            f"vocabulary-overflow:{row['token']}".encode()
        ).digest(),
    ):
        source_cell = (placement["family"], placement["domain"])
        alternatives: list[tuple[int, float, tuple[str, str]]] = []
        try:
            usages = json.loads(placement.get("statistical_usages_json", "[]"))
        except json.JSONDecodeError:
            usages = []
        for usage in usages:
            cell = (str(usage.get("family", "")), str(usage.get("domain", "")))
            if (
                cell != source_cell
                and cell in scenarios_by_cell
                and cell_offsets[cell] < len(scenarios_by_cell[cell])
            ):
                alternatives.append(
                    (
                        int(usage.get("rank", 10_000)),
                        -float(usage.get("score", 0.0)),
                        cell,
                    )
                )

        if alternatives:
            target_cell = min(alternatives)[2]
            fallback_kind = "statistical_alternative"
        else:
            family_cells = [
                cell
                for cell, cell_scenarios in scenarios_by_cell.items()
                if cell[0] == placement["family"]
                and cell_offsets[cell] < len(cell_scenarios)
            ]
            if not family_cells:
                raise ValueError(
                    "vocabulary placement has no compatible scenario capacity "
                    f"for {placement['token']!r} in {placement['family']!r}"
                )
            target_cell = min(
                family_cells,
                key=lambda cell: (
                    cell_offsets[cell] / len(scenarios_by_cell[cell]),
                    hashlib.sha256(
                        f"vocabulary-family-fallback:{placement['token']}:{cell}".encode()
                    ).digest(),
                ),
            )
            fallback_kind = "family_capacity_fallback"

        target_scenarios = sorted(
            scenarios_by_cell[target_cell],
            key=lambda row: hashlib.sha256(
                f"vocabulary-scenario:{row['scenario_id']}".encode()
            ).digest(),
        )
        scenario = target_scenarios[cell_offsets[target_cell]]
        cell_offsets[target_cell] += 1
        reassigned = dict(placement)
        reassigned["assignment_method"] = (
            f"{placement['assignment_method']}:{fallback_kind}"
        )
        assigned[scenario["scenario_id"]] = reassigned

    result: list[dict[str, Any]] = []
    for scenario in scenarios:
        enriched = dict(scenario)
        if placement := assigned.get(scenario["scenario_id"]):
            enriched["lexical_focus"] = placement["token"]
            enriched["lexical_assignment_method"] = placement["assignment_method"]
        result.append(enriched)
    return result
