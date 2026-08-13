from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .behavior_audit import audit_v2_behavior
from .composition_audit import audit_v2_composition
from .distribution_audit import audit_v2_distribution
from .gates import v2_gate_progress
from .integrity_audit import audit_v2_integrity
from .length_audit import audit_v2_lengths
from .near_duplicate_audit import audit_v2_near_duplicates
from .split_audit import audit_v2_splits
from .tokenization_audit import audit_v2_tokenization


_P0_INTEGRITY = {
    "required model fields are missing",
    "assistant envelopes or response fields are inconsistent",
    "unrendered placeholders remain",
    "identical prompts have conflicting answers",
    "arithmetic validation failed",
    "deterministic response validation failed",
}
_P1_BEHAVIOR = {
    "internal_repetition",
    "prompt_copy",
    "abstract_function",
    "exact_response",
    "thinking_internal_repetition",
    "exact_thinking_signature",
    "thinking_fivegram",
    "thinking_final_overlap",
    "thinking_prompt_copy",
    "thinking_coverage",
}
_ACTIONS = {
    "required model fields are missing": "emit every required model-facing field",
    "assistant envelopes or response fields are inconsistent": (
        "repair the one-assistant response and think/final envelope"
    ),
    "unrendered placeholders remain": "repair VariableBy rendering before generation",
    "identical prompts have conflicting answers": (
        "make prompt facts sufficient to determine one answer"
    ),
    "arithmetic validation failed": "rebuild arithmetic cases from recomputed facts",
    "deterministic response validation failed": (
        "author and store a deterministic validator for every generated row"
    ),
    "punctuation or composition defects remain": (
        "repair grammatical compatibility between nested subcards"
    ),
    "variable_provenance_unavailable": (
        "migrate the family to machine-readable V2 decks, cards, indices, and edges"
    ),
    "variable_card_entropy": "rebalance VariableBy card selection",
    "subcard_edge_coverage": "expand and exercise nested subcard compatibility edges",
    "domain_dominance": "add genuinely different domains and reasoning functions",
    "internal_repetition": "remove repeated clauses inside individual answers",
    "prompt_copy": "separate prompt facts from answer phrasing",
    "abstract_function": "replace abstract rubric language with task-specific content",
    "closing_sentence": "expand conclusion functions and sentence forms",
    "exact_response": "expand answer functions beyond a small fixed response pool",
    "thinking_internal_repetition": "remove recursive or repeated thinking clauses",
    "exact_thinking_signature": "expand thinking plans, operations, and ordering",
    "thinking_fivegram": "replace dominant reasoning phrases with functional variants",
    "thinking_final_overlap": "keep reasoning work distinct from the final answer",
    "thinking_prompt_copy": "derive from prompt facts without restating the prompt",
    "thinking_coverage": "author real thinking cards for reasoning examples",
    "prompt_near_duplicates": "expand prompt intents and syntax, not only slot values",
    "final_near_duplicates": "expand answer functions and syntax, not only slot values",
    "empty final responses": "ensure every row has a non-empty final answer",
    "final responses exceed 512 words": "bound verbose answer compositions",
    "thinking traces fall outside 8-120 words": "bound thinking compositions",
    "no train examples were available for tokenization": (
        "provide train rows to the tokenizer audit"
    ),
    "chat serialization does not round-trip through tokenizer": (
        "align the tokenizer and canonical chat serializer"
    ),
    "assistant-only loss mask is misaligned": (
        "repair causal label shifting and assistant-only masking"
    ),
    "think/final markers are not preserved by tokenizer": (
        "make think/final markers stable under tokenization"
    ),
    "at least two populated splits are required": (
        "populate train and held-out splits for the family"
    ),
    "exact conversations leak across splits": (
        "assign exact conversation groups to a single split"
    ),
    "normalized template structures leak across splits": (
        "assign normalized structural groups to a single split"
    ),
    "behavioral compositions leak across splits": (
        "assign each intent × domain × prompt plan × answer plan group to one split"
    ),
    "composition_provenance_unavailable": (
        "render the family through SemanticFrame and explicit discourse plans"
    ),
    "prompt_plan_concentration": "rebalance user-side discourse plans",
    "answer_plan_concentration": "rebalance assistant-side discourse plans",
    "prompt_function_concentration": "author genuinely different request functions",
    "answer_function_concentration": "author genuinely different response functions",
    "prompt_answer_plan_coupling": "decouple prompt and answer plan selection",
    "semantic_frame_contract": "align frame facts, intent, tone, and rendered history",
    "invalid_prompt_answer_compatibility": (
        "select every prompt and answer pair through its declared compatibility graph"
    ),
    "invalid_answer_thinking_compatibility": (
        "select every answer and thinking pair through its declared compatibility graph"
    ),
    "compatibility_graph_drift": (
        "give each distinct compatibility schema a stable deck identity"
    ),
    "insufficient_prompt_answer_graph_coverage": (
        "exercise the declared prompt-to-answer graph more evenly"
    ),
    "insufficient_answer_thinking_graph_coverage": (
        "exercise the declared answer-to-thinking graph more evenly"
    ),
    "insufficient_contextual_multi_turn": "add history-dependent conversational turns",
    "fake_or_unproven_multi_turn": (
        "make prior turns supply a declared fact absent from the current user prompt"
    ),
    "unnecessary_casual_thinking": "use no-thinking plans for casual direct responses",
    "reasoning_budget_mismatch": "use verification thinking only for reasoning tasks",
}


def _actions(*failure_groups: list[str]) -> list[str]:
    return list(
        dict.fromkeys(
            _ACTIONS.get(failure, f"resolve gate failure: {failure}")
            for failures in failure_groups
            for failure in failures
        )
    )


def _priority(
    behavior_failures: list[str],
    integrity_violations: list[str],
    distribution_failures: list[str],
    composition_failures: list[str],
    near_duplicate_failures: list[str],
    length_violations: list[str],
    split_violations: list[str],
    tokenization_failures: list[str],
) -> str:
    if (
        _P0_INTEGRITY & set(integrity_violations)
        or split_violations
        or tokenization_failures
    ):
        return "P0"
    if (
        _P1_BEHAVIOR & set(behavior_failures)
        or integrity_violations
        or distribution_failures
        or composition_failures
        or near_duplicate_failures
        or length_violations
    ):
        return "P1"
    if behavior_failures:
        return "P2"
    return "PASS"


def audit_v2_family_roadmap(
    rows: Iterable[dict[str, Any]],
    *,
    tokenizer_root: str | Path | None = None,
    require_splits: bool = False,
) -> dict[str, Any]:
    """Apply V2 gates independently to every model-facing task family."""

    materialized = list(rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in materialized:
        grouped.setdefault(str(row.get("task", "unknown")), []).append(row)
    families: dict[str, Any] = {}
    for task, task_rows in sorted(grouped.items()):
        train_rows = sum(row.get("split", "train") == "train" for row in task_rows)
        behavior = audit_v2_behavior(
            task_rows,
            thresholds={
                "required_train_examples": train_rows,
                "minimum_direct_casual_examples": 0,
                "minimum_direct_casual_share": 0.0,
                "minimum_short_direct_casual_share": 0.0,
            },
        )
        integrity = audit_v2_integrity(task_rows)
        distribution = audit_v2_distribution(task_rows)
        composition = audit_v2_composition(task_rows)
        near_duplicates = audit_v2_near_duplicates(task_rows)
        lengths = audit_v2_lengths(task_rows, require_global_bands=False)
        splits = (
            audit_v2_splits(task_rows)
            if require_splits
            else {
                "format": "complexity-card-corpus-v2-split-audit-v1",
                "passed": None,
                "violations": [],
                "rows": len(task_rows),
                "split_counts": {},
                "status": "not_run",
            }
        )
        tokenization = (
            audit_v2_tokenization(task_rows, tokenizer_root)
            if tokenizer_root is not None
            else None
        )
        behavior_failures = behavior["failing_tasks"].get(task, [])
        distribution_failures = distribution["tasks"][task]["failures"]
        composition_failures = composition["tasks"][task]["failures"]
        near_duplicate_failures = near_duplicates["tasks"][task]["failures"]
        tokenization_failures = (
            tokenization["failures"] if tokenization is not None else []
        )
        families[task] = {
            "priority": _priority(
                behavior_failures,
                integrity["violations"],
                distribution_failures,
                composition_failures,
                near_duplicate_failures,
                lengths["violations"],
                splits["violations"],
                tokenization_failures,
            ),
            "rows": len(task_rows),
            "train_rows": train_rows,
            "behavior_failures": behavior_failures,
            "behavior": behavior["tasks"][task],
            "integrity_passed": integrity["passed"],
            "integrity_violations": integrity["violations"],
            "integrity_counts": {
                key: value
                for key, value in integrity.items()
                if key.endswith("_count") or key.endswith("_counts")
            },
            "examples": integrity["examples"],
            "distribution_failures": distribution_failures,
            "distribution": distribution["tasks"][task],
            "composition_failures": composition_failures,
            "composition": composition["tasks"][task],
            "near_duplicate_failures": near_duplicate_failures,
            "near_duplicates": near_duplicates["tasks"][task],
            "length_violations": lengths["violations"],
            "lengths": {
                "final_bands": lengths["final_bands"],
                "final_band_shares": lengths["final_band_shares"],
                "thinking_examples": lengths["thinking_examples"],
                "thinking_outside_contract": lengths[
                    "thinking_outside_contract"
                ],
                "thinking_min_words": lengths["thinking_min_words"],
                "thinking_max_words": lengths["thinking_max_words"],
            },
            "split_passed": splits["passed"],
            "split_violations": splits["violations"],
            "splits": splits,
            "tokenization_status": "passed" if tokenization else "not_run",
            "tokenization_failures": tokenization_failures,
            "tokenization": tokenization,
            "actions": _actions(
                integrity["violations"],
                behavior_failures,
                distribution_failures,
                composition_failures,
                near_duplicate_failures,
                lengths["violations"],
                splits["violations"],
                tokenization_failures,
            ),
        }
    gate_progress = v2_gate_progress()
    if not gate_progress["complete"]:
        for family in families.values():
            if family["priority"] == "PASS":
                family["priority"] = "PROVISIONAL"
    priority_counts = {
        priority: sum(family["priority"] == priority for family in families.values())
        for priority in ("P0", "P1", "P2", "PROVISIONAL", "PASS")
    }
    systemic_blockers = []
    blocker_counts: dict[str, int] = {}
    for family in families.values():
        failures = set(
            family["integrity_violations"]
            + family["distribution_failures"]
            + family["composition_failures"]
            + family["behavior_failures"]
            + family["near_duplicate_failures"]
            + family["length_violations"]
            + family["split_violations"]
            + family["tokenization_failures"]
        )
        for failure in failures:
            blocker_counts[failure] = blocker_counts.get(failure, 0) + 1
    for failure, count in sorted(
        blocker_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        if count >= max(2, len(families) // 2):
            systemic_blockers.append(
                {
                    "failure": failure,
                    "families": count,
                    "action": _ACTIONS.get(
                        failure, f"resolve gate failure: {failure}"
                    ),
                }
            )
    split_audit = (
        audit_v2_splits(materialized)
        if require_splits
        else {
            "format": "complexity-card-corpus-v2-split-audit-v1",
            "passed": None,
            "violations": [],
            "rows": len(materialized),
            "split_counts": {},
            "status": "not_run",
        }
    )
    return {
        "format": "complexity-card-corpus-v2-family-roadmap-v1",
        "gate_progress": gate_progress,
        "rows": len(materialized),
        "train_rows": sum(family["train_rows"] for family in families.values()),
        "families": families,
        "priority_counts": priority_counts,
        "systemic_blockers": systemic_blockers,
        "split_audit": split_audit,
        "tokenization_executed": tokenizer_root is not None,
        "split_execution_required": require_splits,
        "complete_gate_contract": gate_progress["complete"],
    }


def roadmap_markdown(roadmap: dict[str, Any]) -> str:
    lines = [
        "# Card Corpus V2 family roadmap",
        "",
        f"Rows audited: {roadmap['rows']:,}",
        (
            "Gate implementation: "
            f"{roadmap['gate_progress']['implemented_count']}/"
            f"{roadmap['gate_progress']['total_count']}"
        ),
        (
            "Tokenizer execution: "
            + ("all families" if roadmap["tokenization_executed"] else "not run")
        ),
        "Split execution: "
        + (
            "passed"
            if roadmap["split_audit"]["passed"] is True
            else "failed"
            if roadmap["split_audit"]["passed"] is False
            else "not run"
        ),
        "",
        (
            "| Priority | Family | Rows | Behavior | Integrity | Distribution | "
            "Near duplicates | Length | Splits | Tokenizer |"
        ),
        "|---|---|---:|---|---|---|---|---|---|---|",
    ]
    for task, family in sorted(
        roadmap["families"].items(),
        key=lambda item: (item[1]["priority"], item[0]),
    ):
        lines.append(
            "| {priority} | {task} | {rows:,} | {behavior} | {integrity} | "
            "{distribution} | {near_duplicates} | {length} | {splits} | "
            "{tokenizer} |".format(
                priority=family["priority"],
                task=task,
                rows=family["rows"],
                behavior=", ".join(family["behavior_failures"]) or "—",
                integrity=", ".join(family["integrity_violations"]) or "—",
                distribution=", ".join(family["distribution_failures"]) or "—",
                near_duplicates=(
                    ", ".join(family["near_duplicate_failures"]) or "—"
                ),
                length=", ".join(family["length_violations"]) or "—",
                splits=(
                    ", ".join(family["split_violations"])
                    or (
                        "passed"
                        if family["split_passed"] is True
                        else "not run"
                    )
                ),
                tokenizer=(
                    ", ".join(family["tokenization_failures"])
                    or family["tokenization_status"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Measured rates",
            "",
            (
                "| Family | Prompt copy | Internal repeat | Abstract rubric | "
                "Prompt collision | Final collision |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task, family in sorted(roadmap["families"].items()):
        behavior = family["behavior"]
        near = family["near_duplicates"]
        lines.append(
            "| {task} | {copy:.1%} | {repeat:.1%} | {abstract:.1%} | "
            "{prompt_collision:.1%} | {final_collision:.1%} |".format(
                task=task,
                copy=behavior["prompt_copy_share"],
                repeat=behavior["internal_repetition_share"],
                abstract=behavior["abstract_function_share"],
                prompt_collision=near["prompt"]["collision_share"],
                final_collision=near["final"]["collision_share"],
            )
        )
    if roadmap["systemic_blockers"]:
        lines.extend(["", "## Systemic blockers", ""])
        for blocker in roadmap["systemic_blockers"]:
            lines.append(
                f"- {blocker['failure']} ({blocker['families']} families): "
                f"{blocker['action']}."
            )
    lines.extend(["", "## Family actions", ""])
    for task, family in sorted(
        roadmap["families"].items(),
        key=lambda item: (item[1]["priority"], item[0]),
    ):
        lines.append(f"### {family['priority']} — {task}")
        lines.append("")
        for action in family["actions"]:
            lines.append(f"- {action}.")
        lines.append("")
    return "\n".join(lines) + "\n"


__all__ = (
    "audit_v2_family_roadmap",
    "roadmap_markdown",
)
