from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any, Iterable


def _field_value(nested: dict[str, dict[str, int]], field: str) -> int:
    axis, separator, sense = field.partition("[")
    if not separator or not sense.endswith("]"):
        raise ValueError(f"invalid VariableBy2D field {field!r}")
    return int(nested[axis][sense[:-1]])


def _normalized_entropy(counts: Counter[int], size: int) -> float:
    total = sum(counts.values())
    if size <= 1 or total == 0:
        return 1.0
    entropy = -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
        if count
    )
    return entropy / math.log(size)


def audit_v2_distribution(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Audit domains, VariableBy2D card entropy, and nested edge coverage."""

    task_rows = Counter()
    task_train_rows = Counter()
    task_train_domains: dict[str, Counter[str]] = defaultdict(Counter)
    task_all_domains: dict[str, Counter[str]] = defaultdict(Counter)
    auditable_rows = Counter()
    unavailable_rows = Counter()
    card_counts: dict[tuple[str, str, str, int], Counter[int]] = defaultdict(Counter)
    edge_counts: dict[
        tuple[str, str, str, str, int], Counter[tuple[int, int]]
    ] = defaultdict(Counter)

    for row in rows:
        task = str(row.get("task", "unknown"))
        task_rows[task] += 1
        domain = str(row.get("domain", "unknown"))
        task_all_domains[task][domain] += 1
        if row.get("split", "train") == "train":
            task_train_rows[task] += 1
            task_train_domains[task][domain] += 1
        try:
            metadata = json.loads(str(row.get("source_representation", "")))
            deck = str(metadata["deck_name"])
            indices = metadata["variable_indices"]
            sizes = metadata["variable_card_counts"]
            graph = metadata["dependency_graph"]
        except (KeyError, TypeError, json.JSONDecodeError):
            unavailable_rows[task] += 1
            continue
        auditable_rows[task] += 1
        flat_sizes = {
            f"{axis}[{sense}]": int(size)
            for axis, senses in sizes.items()
            for sense, size in senses.items()
        }
        for field, size in flat_sizes.items():
            card_counts[(task, deck, field, size)][_field_value(indices, field)] += 1
        for parent, dependencies in graph.items():
            for dependency in dependencies:
                product = flat_sizes[parent] * flat_sizes[dependency]
                edge_counts[(task, deck, parent, dependency, product)][
                    (
                        _field_value(indices, parent),
                        _field_value(indices, dependency),
                    )
                ] += 1

    task_metrics: dict[str, Any] = {}
    violations: list[str] = []
    for task in sorted(task_rows):
        rows_count = task_rows[task]
        domain_counts = task_train_domains[task] or task_all_domains[task]
        domain, domain_count = domain_counts.most_common(1)[0]
        domain_share = domain_count / sum(domain_counts.values())
        reservoirs = []
        reservoir_failures = 0
        for (group_task, deck, field, size), counts in sorted(card_counts.items()):
            if group_task != task or size <= 1:
                continue
            samples = sum(counts.values())
            entropy = _normalized_entropy(counts, size)
            coverage = len(counts) / size
            tested = samples >= size * 5
            passed = not tested or (coverage == 1.0 and entropy >= 0.90)
            reservoir_failures += not passed
            reservoirs.append(
                {
                    "deck": deck,
                    "field": field,
                    "cards": size,
                    "samples": samples,
                    "coverage": round(coverage, 6),
                    "normalized_entropy": round(entropy, 6),
                    "tested": tested,
                    "passed": passed,
                }
            )
        edges = []
        edge_failures = 0
        for (group_task, deck, parent, dependency, product), counts in sorted(
            edge_counts.items()
        ):
            if group_task != task or product <= 1:
                continue
            samples = sum(counts.values())
            coverage = len(counts) / product
            tested = samples >= product * 5
            passed = not tested or coverage >= 0.80
            edge_failures += not passed
            edges.append(
                {
                    "deck": deck,
                    "parent": parent,
                    "dependency": dependency,
                    "possible_edges": product,
                    "observed_edges": len(counts),
                    "coverage": round(coverage, 6),
                    "tested": tested,
                    "passed": passed,
                }
            )
        failures = []
        if unavailable_rows[task]:
            failures.append("variable_provenance_unavailable")
        if domain_share > 0.35:
            failures.append("domain_dominance")
        if reservoir_failures:
            failures.append("variable_card_entropy")
        if edge_failures:
            failures.append("subcard_edge_coverage")
        if failures:
            violations.append(task)
        task_metrics[task] = {
            "rows": rows_count,
            "train_rows": task_train_rows[task],
            "auditable_variable_rows": auditable_rows[task],
            "unavailable_variable_rows": unavailable_rows[task],
            "distinct_domains": len(domain_counts),
            "top_domain": domain,
            "top_domain_share": round(domain_share, 6),
            "reservoir_failure_count": reservoir_failures,
            "edge_failure_count": edge_failures,
            "failures": failures,
            "reservoirs": reservoirs,
            "edges": edges,
        }
    return {
        "format": "complexity-card-corpus-v2-distribution-audit-v2",
        "passed": not violations,
        "failing_tasks": violations,
        "tasks": task_metrics,
        "thresholds": {
            "maximum_domain_share": 0.35,
            "minimum_normalized_card_entropy": 0.90,
            "minimum_card_coverage": 1.0,
            "minimum_nested_edge_coverage": 0.80,
            "minimum_samples_per_card_or_edge": 5,
            "variable_coverage_scope": "all_splits",
            "domain_balance_scope": "train",
        },
    }


__all__ = ("audit_v2_distribution",)
