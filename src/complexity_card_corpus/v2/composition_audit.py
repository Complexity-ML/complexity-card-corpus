from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any, Iterable


_NON_WORD = re.compile(r"[^a-z0-9]+")


def _share(counter: Counter[Any]) -> float:
    if not counter:
        return 0.0
    return counter.most_common(1)[0][1] / max(1, sum(counter.values()))


def _signature(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in value) if isinstance(value, list) else ()


def _edges(value: Any) -> frozenset[tuple[str, str]]:
    if not isinstance(value, list):
        raise TypeError("compatibility edges must be a list")
    edges: set[tuple[str, str]] = set()
    for edge in value:
        if not isinstance(edge, list) or len(edge) != 2:
            raise TypeError("compatibility edges must contain two-item lists")
        source, target = (str(item).strip() for item in edge)
        if not source or not target:
            raise TypeError("compatibility edge names must be visible")
        edges.add((source, target))
    if not edges:
        raise TypeError("a compatibility graph cannot be empty")
    return frozenset(edges)


def _normalise_text(value: Any) -> str:
    return " ".join(_NON_WORD.sub(" ", str(value).casefold()).split())


def _contains_value(text: str, value: Any) -> bool:
    needle = _normalise_text(value)
    haystack = _normalise_text(text)
    return bool(needle) and f" {needle} " in f" {haystack} "


def _semantic_contract_is_valid(
    *,
    metadata: dict[str, Any],
    composition: dict[str, Any],
    messages: Any,
) -> tuple[bool, bool]:
    """Return (frame_valid, contextual_history_proven).

    A multi-turn row is contextual only when a declared frame fact is visible in
    the prior turns and absent from the current user turn.  Merely prepending a
    disposable exchange therefore cannot satisfy the contract.
    """

    frame = metadata.get("semantic_frame")
    if (
        not isinstance(frame, dict)
        or not isinstance(messages, list)
        or len(messages) < 2
    ):
        return False, False
    if frame.get("facts") != metadata.get("facts"):
        return False, False
    if str(frame.get("intent", "")) != str(composition.get("intent", "")):
        return False, False
    if str(frame.get("user_tone", "")) != str(composition.get("user_tone", "")):
        return False, False
    history = messages[:-2]
    if frame.get("history") != history:
        return False, False
    required = frame.get("history_required_facts")
    if not isinstance(required, list):
        return False, False
    if not history:
        return (not required), False
    if not required or len(required) != len(set(map(str, required))):
        return False, False
    facts = frame.get("facts")
    if not isinstance(facts, dict):
        return False, False
    if not all(isinstance(turn, dict) for turn in history):
        return False, False
    history_text = " ".join(str(turn.get("content", "")) for turn in history)
    current_user = messages[-2]
    if not isinstance(current_user, dict) or current_user.get("role") != "user":
        return False, False
    current_text = str(current_user.get("content", ""))
    for raw_key in required:
        key = str(raw_key)
        if key not in facts:
            return False, False
        value = facts[key]
        if not _contains_value(history_text, value) or _contains_value(
            current_text, value
        ):
            return False, False
    return True, True


def audit_v2_composition(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Prove semantic frames, compatibility graphs, and behavioral diversity."""

    task_rows: Counter[str] = Counter()
    unavailable: Counter[str] = Counter()
    semantic_failures: Counter[str] = Counter()
    history_failures: Counter[str] = Counter()
    genuine_multi_turn: Counter[str] = Counter()
    invalid_prompt_answer: Counter[str] = Counter()
    invalid_answer_thinking: Counter[str] = Counter()
    prompt_plans: dict[str, Counter[str]] = defaultdict(Counter)
    answer_plans: dict[str, Counter[str]] = defaultdict(Counter)
    thinking_plans: dict[str, Counter[str]] = defaultdict(Counter)
    prompt_functions: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    answer_functions: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    thinking_functions: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    edges: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    thinking_budgets: dict[str, Counter[str]] = defaultdict(Counter)
    multi_turn: Counter[str] = Counter()
    deck_rows: Counter[tuple[str, str]] = Counter()
    allowed_pa: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    allowed_at: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    observed_pa: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    observed_at: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    graph_signatures: dict[
        tuple[str, str], set[tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]]
    ] = defaultdict(set)

    for row in rows:
        task = str(row.get("task", "unknown"))
        task_rows[task] += 1
        messages = row.get("messages", [])
        is_multi_turn = isinstance(messages, list) and len(messages) > 2
        if is_multi_turn:
            multi_turn[task] += 1
        try:
            metadata = json.loads(str(row["source_representation"]))
            if not isinstance(metadata, dict):
                raise TypeError("source representation must decode to an object")
            composition = metadata["composition"]
            if not isinstance(composition, dict):
                raise TypeError("composition must be an object")
            prompt = str(composition["prompt_plan"])
            answer = str(composition["answer_plan"])
            thinking = str(composition["thinking_plan"])
            prompt_signature = _signature(composition["prompt_functions"])
            answer_signature = _signature(composition["answer_functions"])
            thinking_signature = _signature(composition["thinking_functions"])
            budget = str(composition["thinking_budget"])
            deck = str(composition["deck_name"])
            prompt_answer_graph = _edges(composition["allowed_prompt_answer_edges"])
            answer_thinking_graph = _edges(composition["allowed_answer_thinking_edges"])
        except (KeyError, TypeError, json.JSONDecodeError):
            unavailable[task] += 1
            continue

        frame_valid, history_proven = _semantic_contract_is_valid(
            metadata=metadata,
            composition=composition,
            messages=messages,
        )
        if not frame_valid:
            semantic_failures[task] += 1
        if is_multi_turn:
            if history_proven:
                genuine_multi_turn[task] += 1
            else:
                history_failures[task] += 1

        deck_key = (task, deck)
        deck_rows[deck_key] += 1
        pa_edge = (prompt, answer)
        at_edge = (answer, thinking)
        allowed_pa[deck_key].update(prompt_answer_graph)
        allowed_at[deck_key].update(answer_thinking_graph)
        observed_pa[deck_key].add(pa_edge)
        observed_at[deck_key].add(at_edge)
        graph_signatures[deck_key].add(
            (tuple(sorted(prompt_answer_graph)), tuple(sorted(answer_thinking_graph)))
        )
        if pa_edge not in prompt_answer_graph:
            invalid_prompt_answer[task] += 1
        if at_edge not in answer_thinking_graph:
            invalid_answer_thinking[task] += 1

        prompt_plans[task][prompt] += 1
        answer_plans[task][answer] += 1
        thinking_plans[task][thinking] += 1
        prompt_functions[task][prompt_signature] += 1
        answer_functions[task][answer_signature] += 1
        thinking_functions[task][thinking_signature] += 1
        thinking_budgets[task][budget] += 1
        edges[task][pa_edge] += 1

    tasks: dict[str, Any] = {}
    failing_tasks: list[str] = []
    for task in sorted(task_rows):
        rows_count = task_rows[task]
        failures = []
        if unavailable[task]:
            failures.append("composition_provenance_unavailable")
        if semantic_failures[task]:
            failures.append("semantic_frame_contract")
        if invalid_prompt_answer[task]:
            failures.append("invalid_prompt_answer_compatibility")
        if invalid_answer_thinking[task]:
            failures.append("invalid_answer_thinking_compatibility")
        task_decks = [key for key in deck_rows if key[0] == task]
        graph_drift_decks = sum(len(graph_signatures[key]) > 1 for key in task_decks)
        if graph_drift_decks:
            failures.append("compatibility_graph_drift")

        tested_pa = [
            key
            for key in task_decks
            if deck_rows[key] >= 5 * len(allowed_pa[key])
        ]
        tested_at = [
            key
            for key in task_decks
            if deck_rows[key] >= 5 * len(allowed_at[key])
        ]
        pa_denominator = sum(len(allowed_pa[key]) for key in tested_pa)
        at_denominator = sum(len(allowed_at[key]) for key in tested_at)
        pa_covered = sum(
            len(observed_pa[key] & allowed_pa[key]) for key in tested_pa
        )
        at_covered = sum(
            len(observed_at[key] & allowed_at[key]) for key in tested_at
        )
        pa_coverage = pa_covered / max(1, pa_denominator) if tested_pa else 1.0
        at_coverage = at_covered / max(1, at_denominator) if tested_at else 1.0
        if tested_pa and pa_coverage < 0.90:
            failures.append("insufficient_prompt_answer_graph_coverage")
        if tested_at and at_coverage < 0.90:
            failures.append("insufficient_answer_thinking_graph_coverage")

        prompt_plan_share = _share(prompt_plans[task])
        answer_plan_share = _share(answer_plans[task])
        prompt_function_share = _share(prompt_functions[task])
        answer_function_share = _share(answer_functions[task])
        edge_share = _share(edges[task])
        if len(prompt_plans[task]) >= 4 and prompt_plan_share > 0.45:
            failures.append("prompt_plan_concentration")
        if len(answer_plans[task]) >= 4 and answer_plan_share > 0.45:
            failures.append("answer_plan_concentration")
        if len(prompt_functions[task]) >= 4 and prompt_function_share > 0.45:
            failures.append("prompt_function_concentration")
        if len(answer_functions[task]) >= 4 and answer_function_share > 0.45:
            failures.append("answer_function_concentration")
        if len(edges[task]) >= 8 and edge_share > 0.25:
            failures.append("prompt_answer_plan_coupling")
        multi_turn_share = multi_turn[task] / max(1, rows_count)
        if task == "casual_conversation" and multi_turn_share < 0.20:
            failures.append("insufficient_contextual_multi_turn")
        if history_failures[task]:
            failures.append("fake_or_unproven_multi_turn")
        if task == "casual_conversation" and any(
            budget != "none" for budget in thinking_budgets[task]
        ):
            failures.append("unnecessary_casual_thinking")
        if task == "reasoning_verification" and set(
            thinking_budgets[task]
        ) != {"verification"}:
            failures.append("reasoning_budget_mismatch")
        if failures:
            failing_tasks.append(task)
        tasks[task] = {
            "rows": rows_count,
            "unavailable_rows": unavailable[task],
            "semantic_frame_contract_failure_rows": semantic_failures[task],
            "invalid_prompt_answer_edge_rows": invalid_prompt_answer[task],
            "invalid_answer_thinking_edge_rows": invalid_answer_thinking[task],
            "compatibility_graph_drift_decks": graph_drift_decks,
            "distinct_prompt_plans": len(prompt_plans[task]),
            "top_prompt_plan_share": round(prompt_plan_share, 6),
            "distinct_answer_plans": len(answer_plans[task]),
            "top_answer_plan_share": round(answer_plan_share, 6),
            "distinct_thinking_plans": len(thinking_plans[task]),
            "distinct_prompt_function_signatures": len(prompt_functions[task]),
            "top_prompt_function_signature_share": round(prompt_function_share, 6),
            "distinct_answer_function_signatures": len(answer_functions[task]),
            "top_answer_function_signature_share": round(answer_function_share, 6),
            "distinct_thinking_function_signatures": len(thinking_functions[task]),
            "distinct_prompt_answer_edges": len(edges[task]),
            "top_prompt_answer_edge_share": round(edge_share, 6),
            "tested_prompt_answer_graphs": len(tested_pa),
            "allowed_prompt_answer_edges": pa_denominator,
            "prompt_answer_edge_coverage": round(pa_coverage, 6),
            "tested_answer_thinking_graphs": len(tested_at),
            "allowed_answer_thinking_edges": at_denominator,
            "answer_thinking_edge_coverage": round(at_coverage, 6),
            "multi_turn_rows": multi_turn[task],
            "genuine_multi_turn_rows": genuine_multi_turn[task],
            "history_contract_failure_rows": history_failures[task],
            "multi_turn_share": round(multi_turn_share, 6),
            "thinking_budgets": dict(sorted(thinking_budgets[task].items())),
            "failures": failures,
        }
    return {
        "format": "complexity-card-corpus-v2-composition-audit-v2",
        "passed": not failing_tasks,
        "failing_tasks": failing_tasks,
        "tasks": tasks,
        "thresholds": {
            "maximum_prompt_plan_share_when_diverse": 0.45,
            "maximum_answer_plan_share_when_diverse": 0.45,
            "maximum_function_signature_share_when_diverse": 0.45,
            "maximum_prompt_answer_edge_share_when_diverse": 0.25,
            "minimum_compatibility_edge_coverage": 0.90,
            "minimum_rows_per_declared_edge_for_coverage": 5,
            "minimum_casual_multi_turn_share": 0.20,
            "maximum_invalid_compatibility_edges": 0,
            "maximum_unproven_multi_turn_rows": 0,
        },
    }


__all__ = ("audit_v2_composition",)
