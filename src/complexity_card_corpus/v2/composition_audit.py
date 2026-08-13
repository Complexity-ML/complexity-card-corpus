from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Iterable


def _share(counter: Counter[Any]) -> float:
    return counter.most_common(1)[0][1] / max(1, sum(counter.values())) if counter else 0.0


def _signature(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in value) if isinstance(value, list) else ()


def audit_v2_composition(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Measure behavioral plans instead of treating lexical changes as diversity."""

    task_rows: Counter[str] = Counter()
    unavailable: Counter[str] = Counter()
    prompt_plans: dict[str, Counter[str]] = defaultdict(Counter)
    answer_plans: dict[str, Counter[str]] = defaultdict(Counter)
    thinking_plans: dict[str, Counter[str]] = defaultdict(Counter)
    prompt_functions: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    answer_functions: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    thinking_functions: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    edges: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    thinking_budgets: dict[str, Counter[str]] = defaultdict(Counter)
    multi_turn: Counter[str] = Counter()

    for row in rows:
        task = str(row.get("task", "unknown"))
        task_rows[task] += 1
        messages = row.get("messages", [])
        if isinstance(messages, list) and len(messages) > 2:
            multi_turn[task] += 1
        try:
            metadata = json.loads(str(row["source_representation"]))
            composition = metadata["composition"]
            prompt = str(composition["prompt_plan"])
            answer = str(composition["answer_plan"])
            thinking = str(composition["thinking_plan"])
            prompt_signature = _signature(composition["prompt_functions"])
            answer_signature = _signature(composition["answer_functions"])
            thinking_signature = _signature(composition["thinking_functions"])
            budget = str(composition["thinking_budget"])
        except (KeyError, TypeError, json.JSONDecodeError):
            unavailable[task] += 1
            continue
        prompt_plans[task][prompt] += 1
        answer_plans[task][answer] += 1
        thinking_plans[task][thinking] += 1
        prompt_functions[task][prompt_signature] += 1
        answer_functions[task][answer_signature] += 1
        thinking_functions[task][thinking_signature] += 1
        thinking_budgets[task][budget] += 1
        edges[task][(prompt, answer)] += 1

    tasks: dict[str, Any] = {}
    failing_tasks: list[str] = []
    for task in sorted(task_rows):
        rows_count = task_rows[task]
        failures = []
        if unavailable[task]:
            failures.append("composition_provenance_unavailable")
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
            "multi_turn_rows": multi_turn[task],
            "multi_turn_share": round(multi_turn_share, 6),
            "thinking_budgets": dict(sorted(thinking_budgets[task].items())),
            "failures": failures,
        }
    return {
        "format": "complexity-card-corpus-v2-composition-audit-v1",
        "passed": not failing_tasks,
        "failing_tasks": failing_tasks,
        "tasks": tasks,
        "thresholds": {
            "maximum_prompt_plan_share_when_diverse": 0.45,
            "maximum_answer_plan_share_when_diverse": 0.45,
            "maximum_function_signature_share_when_diverse": 0.45,
            "maximum_prompt_answer_edge_share_when_diverse": 0.25,
            "minimum_casual_multi_turn_share": 0.20,
        },
    }


__all__ = ("audit_v2_composition",)
