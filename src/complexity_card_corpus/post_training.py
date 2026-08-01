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
from .task_cards import TaskHand, deal_task_hand


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
_MAX_FAMILY_SKELETON_SHARE = 0.20
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
    "grounded_qa": "question_goal",
    "summarization_synthesis": "summary_goal",
    "extraction_classification": "extraction_goal",
    "reasoning_verification": "reasoning_goal",
    "critique_revision": "critique_goal",
    "brainstorming_creativity": "ideation_goal",
    "context_clarification": "clarification_goal",
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

_INSTRUCT_OPENINGS = (
    "Please solve this hand.",
    "Work through the following card hand.",
    "Use these cards to produce the requested result.",
    "Resolve this case from the supplied cards.",
    "Complete the task described by this hand.",
    "Apply the rule and goal to the data below.",
    "Handle this card hand using only its stated facts.",
    "Produce a bounded answer for the following case.",
)

_CHAT_OPENINGS = (
    "I want to work through this hand.",
    "Can we resolve this case together?",
    "I need help with the following card hand.",
    "Let's work from these situation and data cards.",
    "Could you help me reason through this case?",
    "I have a bounded task to work through.",
    "Let's start with the situation and known facts.",
    "I would like to handle this case carefully.",
)


def _stable_index(value: str, size: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % size


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
    "answer the direct question": "answer the direct question about {subject}",
    "locate supporting evidence": "locate supporting evidence for {subject}",
    "compare two claims": "compare two claims about {subject}",
    "draw a cautious inference": "draw a cautious inference about {subject}",
    "identify what remains unknown": "identify what remains unknown about {subject}",
    "summarize the essentials": "summarize the essentials of {subject}",
    "synthesize related points": "synthesize the related points in {subject}",
    "extract decisions and actions": "extract decisions and actions from {subject}",
    "organize the chronology": "organize the chronology of {subject}",
    "adapt the summary for its audience": "adapt the summary of {subject} for its audience",
    "extract the requested fields": "extract the requested fields from {subject}",
    "normalize the recorded values": "normalize the recorded values in {subject}",
    "classify the record": "classify the record for {subject}",
    "identify missing required fields": "identify missing required fields in {subject}",
    "convert the record into a clear structure": "convert {subject} into a clear structure",
    "calculate the requested result": "calculate the requested result for {subject}",
    "compare the available quantities": "compare the available quantities for {subject}",
    "test whether the constraint is satisfied": "test whether the constraint for {subject} is satisfied",
    "explain the reasoning step by step": "explain the reasoning for {subject} step by step",
    "verify the proposed result": "verify the proposed result for {subject}",
    "identify the most important weakness": "identify the most important weakness in {subject}",
    "revise the weak section": "revise the weak section of {subject}",
    "check internal consistency": "check the internal consistency of {subject}",
    "strengthen the evidence connection": "strengthen the evidence connection in {subject}",
    "prioritize the necessary fixes": "prioritize the necessary fixes for {subject}",
    "generate several distinct options": "generate several distinct options for {subject}",
    "diversify the current ideas": "diversify the current ideas for {subject}",
    "filter ideas against the criteria": "filter ideas for {subject} against the criteria",
    "combine compatible concepts": "combine compatible concepts for {subject}",
    "develop one promising idea": "develop one promising idea for {subject}",
    "ask one decisive clarifying question": "ask one decisive clarifying question about {subject}",
    "restate the understood request": "restate the understood request for {subject}",
    "resolve the ambiguous reference": "resolve the ambiguous reference in {subject}",
    "separate facts from assumptions": "separate facts from assumptions about {subject}",
    "propose a bounded interpretation": "propose a bounded interpretation of {subject}",
}


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
    scenario_id = str(answer.get("scenario_id", ""))
    codes = {scenario_id.split(":")[-1][:6] if scenario_id else ""}
    prefix = re.match(r"^(?:for\s+)?hand\s+([0-9a-f]{6})\s*[:—-]", masked, re.I)
    if prefix:
        codes.add(prefix.group(1))
    for code in sorted(filter(None, codes), key=len, reverse=True):
        masked = re.sub(
            rf"\b{re.escape(code)}(?:-[ab])?\b",
            "<id>",
            masked,
            flags=re.IGNORECASE,
        )
    masked = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "<date>", masked)
    masked = re.sub(r"\$\d+(?:\.\d+)?", "<amount>", masked)
    masked = re.sub(r"\b\d{1,2}:\d{2}\b", "<time>", masked)
    masked = re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", masked)
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
        "masked_surface_variables": [
            "scenario_code",
            "reference_suffix",
            "date",
            "amount",
            "time",
            "number",
        ],
        "skeletons": len(skeletons),
        "distinct_skeletons": len(counts),
        "exact_skeleton_uniqueness_ratio": round(len(counts) / len(skeletons), 6)
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
        modes["instruct"].startswith(modes["chat"]) for modes in paired_prompt_surfaces
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

    card_contracts: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for answer in answers:
        card_hand = answer.get("card_hand")
        if not isinstance(card_hand, dict):
            raise ValueError("post-training answer is missing its card-hand contract")
        if card_hand.get("cards") != ["situation", "data", "rule", "goal"]:
            raise ValueError("post-training card hand has an invalid card sequence")
        contract = tuple(card_hand.get("completion_contract", ()))
        if not contract:
            raise ValueError("post-training card hand has an empty completion contract")
        card_contracts[answer["family"]].add(contract)
    if set(card_contracts) != set(_INTENT_FIELD):
        raise ValueError("post-training card hands do not cover all task families")

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
        family_answer_json = [json.loads(row["answer_json"]) for row in family_rows]
        family_skeletons = [
            _masked_response(response, answer)
            for response, answer in zip(
                family_responses, family_answer_json, strict=True
            )
        ]
        family_skeleton_counts = Counter(family_skeletons)
        maximum_family_skeleton_share = max(
            family_skeleton_counts.values(), default=0
        ) / len(family_skeletons)
        if maximum_family_skeleton_share >= _MAX_FAMILY_SKELETON_SHARE:
            raise ValueError(
                f"{family} response skeleton reaches the family repetition ceiling: "
                f"{maximum_family_skeleton_share:.3%}"
            )
        family_metrics[family] = {
            "examples": len(family_rows),
            "source_scenarios": len(set(family_answers)),
            "unique_final_response_ratio": round(
                len(set(family_responses)) / len(family_responses), 6
            ),
            "mean_response_tokens": _length_statistics(family_responses)["mean_tokens"],
            "masked_response_templates": len(family_skeleton_counts),
            "maximum_masked_template_share": round(maximum_family_skeleton_share, 6),
            "strict_masked_template_share_limit": _MAX_FAMILY_SKELETON_SHARE,
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
        "card_game": {
            "cards_per_hand": ["situation", "data", "rule", "goal"],
            "families": len(card_contracts),
            "family_completion_contracts": {
                family: [list(contract) for contract in sorted(contracts)]
                for family, contracts in sorted(card_contracts.items())
            },
            "all_hands_have_non_empty_contract": True,
        },
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
        "response_repetition_gate": {
            "maximum_masked_eight_token_message_coverage": maximum_final_phrase_share,
            "strict_share_limit": _MAX_SURFACE_FORMULATION_SHARE,
            "measured_from_rendered_responses": True,
        },
        "masked_response_diversity": masked_response_diversity,
        "role_text_stats": {
            "user_prompts": _text_statistics(user_prompts),
            "assistant_messages": _text_statistics(assistant_messages),
            "final_responses": final_response_stats,
        },
        "natural_language_gate": {
            "assistant_meta_instruction_hits": len(assistant_meta_hits),
            "user_meta_request_hits": len(user_meta_hits),
            "forbidden_assistant_phrases": list(_FORBIDDEN_ASSISTANT_META_PHRASES),
            "forbidden_user_phrases": list(_FORBIDDEN_USER_META_PHRASES),
        },
        "message_length_stats": _length_statistics(messages),
        "response_length_stats": _length_statistics(responses),
        "lexical_stats": {
            "word_occurrences": len(all_message_tokens),
            "observed_vocabulary": len(vocabulary),
            "raw_type_token_ratio": round(len(vocabulary) / len(all_message_tokens), 6),
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
        raise ValueError("review_scenarios must be a positive multiple of family count")
    quota = review_scenarios // len(families)
    scenario_rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    scenario_answers: dict[str, dict[str, Any]] = {}
    grouped: dict[str, dict[tuple[str, str], dict[str, list[str]]]] = defaultdict(
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
            continue
        grouped[answer["family"]][(answer["risk_level"], answer["split"])][
            answer["domain"]
        ].append(scenario_id)

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
                        "transcript": row["rendered_text"],
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
    variants_per_scenario: int = 4,
    review_scenarios: int = 140,
    seed: int = 42,
    vocabulary_placement_path: Path | None = None,
) -> dict[str, Any]:
    if variants_per_scenario < 1:
        raise ValueError("variants_per_scenario must be positive")
    scenarios = pq.read_table(scenarios_path).to_pylist()
    placements = (
        _load_vocabulary_placements(vocabulary_placement_path)
        if vocabulary_placement_path is not None
        else []
    )
    rows = _conversation_rows(
        scenarios,
        variants_per_scenario,
        vocabulary_placements=placements,
    )
    rows, family_balance = _balance_conversation_families(rows)
    audit = _audit(rows)
    audit["family_balance"] = family_balance
    observed_lexical_focus = {
        json.loads(row["answer_json"])["lexical_focus"]
        for row in rows
        if json.loads(row["answer_json"])["lexical_focus"]
    }
    requested_lexical_focus = {row["token"] for row in placements}
    if observed_lexical_focus != requested_lexical_focus:
        missing = sorted(requested_lexical_focus - observed_lexical_focus)
        raise ValueError(f"vocabulary placement coverage is incomplete: {missing[:5]}")
    realized_placements: dict[str, tuple[str, str]] = {}
    for row in rows:
        answer = json.loads(row["answer_json"])
        if answer["lexical_focus"]:
            realized_placements[answer["lexical_focus"]] = (
                answer["lexical_assignment_method"],
                answer["family"],
            )
    lexical_methods = Counter(
        method for method, _family in realized_placements.values()
    )
    lexical_families = Counter(
        family for _method, family in realized_placements.values()
    )
    audit["vocabulary_placement"] = {
        "requested_tokens": len(requested_lexical_focus),
        "observed_tokens": len(observed_lexical_focus),
        "coverage_ratio": 1.0 if placements else None,
        "mapped_conversation_rows": sum(
            bool(json.loads(row["answer_json"])["lexical_focus"]) for row in rows
        ),
        "surfaced_conversation_rows": 0,
        "assignment_methods": dict(sorted(lexical_methods.items())),
        "family_counts": dict(sorted(lexical_families.items())),
        "surface_policy": "metadata_only" if placements else None,
        "automatic_definition_generation": False,
    }
    review = _review_sample(rows, review_scenarios=review_scenarios, seed=seed)

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
        "vocabulary_placement_input": (
            {
                "path": str(vocabulary_placement_path),
                "sha256": file_sha256(vocabulary_placement_path),
            }
            if vocabulary_placement_path is not None
            else None
        ),
        "variants_per_scenario": variants_per_scenario,
        "audit": audit,
        "human_review": {
            "rows": len(review),
            "source_scenarios": len({row["scenario_id"] for row in review}),
            "sample_fraction_of_source_scenarios": round(
                len({row["scenario_id"] for row in review}) / audit["source_scenarios"],
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
        "transcript",
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
        if Counter(row["mode"] for row in values) != Counter({"instruct": 1, "chat": 1})
    }
    if incomplete_modes:
        raise ValueError(
            "each reviewed scenario must contain one instruct and one chat row: "
            f"{incomplete_modes}"
        )

    status_counts = Counter(statuses)
    grade_counts = {
        grade: dict(sorted(Counter(row[grade].strip().lower() for row in rows).items()))
        for grade in REVIEW_GRADES
    }
    approved = all(status == "approved" for status in statuses)
    grades_pass = all(
        row[grade].strip().lower() == "pass" for row in rows for grade in REVIEW_GRADES
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
                Counter(values[0][field] for values in scenario_rows.values()).items()
            )
        )

    def row_failed(row: dict[str, str]) -> bool:
        return row["review_status"].strip().lower() == "rejected" or any(
            row[grade].strip().lower() == "fail" for grade in REVIEW_GRADES
        )

    failed_rows = [row for row in rows if row_failed(row)]
    failed_scenarios = {row["scenario_id"] for row in failed_rows}
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
                    row["reviewer"].strip() for row in rows if row["reviewer"].strip()
                ).items()
            )
        ),
        "review_provenance_complete": provenance_complete,
        "zero_failure_bound": zero_failure_bound,
        "training_ready": ready,
    }
