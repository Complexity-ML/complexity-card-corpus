"""Regression coverage for the families whose answers were re-grounded with
real per-scenario variables after the generic answer-development reservoir
was removed (see sft/answer_development.py).

These tests lock in two properties that a future edit could silently break:
  - responses stay below the near-duplicate threshold the training-quality
    audit enforces, even sampled across many scenarios of the same domain;
  - every f-string placeholder in an authored case actually interpolates,
    instead of leaking a literal "{variable}" into the rendered answer.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

from complexity_card_corpus.quality_audit import audit_rows_quality
from complexity_card_corpus.tasks import deal_task_hand
from complexity_card_corpus.tasks.intent_contracts import intent_contract_catalog

_DOMAINS_BY_FAMILY = {
    "grounded_qa": (
        "product_specs", "policy_excerpt", "science_passage", "historical_note",
        "project_brief", "travel_information", "technical_documentation",
        "comparison_table", "conflicting_service_reports", "public_event_notice",
        "energy_bill", "course_catalog", "maintenance_log", "environmental_report",
        "software_release_note", "contract_clause", "lab_report", "procurement_quote",
        "accessibility_statement", "equipment_manual",
    ),
    "summarization_synthesis": (
        "meeting_transcript", "research_notes", "support_thread", "project_update",
        "policy_memo", "article_excerpt", "incident_log", "learning_notes",
    ),
    "critique_revision": (
        "email_draft", "argument", "project_plan", "explanation", "instructions",
        "summary", "claim_evidence", "interface_copy", "status_update",
        "survey_report", "policy_notice", "data_caption", "release_note",
        "support_macro", "risk_assessment",
    ),
    "reasoning_verification": (
        "shopping_arithmetic", "schedule_math", "unit_conversion", "proportions",
        "table_comparison", "sequence_pattern", "logical_constraints",
        "work_allocation", "simple_probability",
    ),
    "context_clarification": (
        "ambiguous_request", "missing_reference", "conflicting_instruction",
        "unclear_pronoun", "incomplete_goal", "scope_boundary", "format_preference",
        "timeline_ambiguity", "team_request", "data_request", "travel_request",
        "purchasing_request",
    ),
    "explanation_learning": (
        "computing", "software_resilience", "data_literacy", "physical_science",
        "life_science", "mathematics", "personal_finance", "civics",
        "media_literacy", "probability", "ecology", "electrical_energy",
        "language_grammar", "computer_networks", "research_methods",
    ),
    "safety_uncertainty": (
        "privacy_security", "medical_information", "financial_decision",
        "physical_safety",
    ),
    "brainstorming_creativity": (
        "names", "lesson_activity", "event_plan", "feature_ideas",
        "writing_prompts", "low_cost_activity", "outreach", "workflow",
    ),
}

_SCENARIOS_PER_DOMAIN = 60

_TASK_RENDERERS_BY_FILE = {
    "action.py": {"_practical", "_troubleshooting", "_planning"},
    "communication.py": {"_explanation", "_writing", "_empathy", "_clarification"},
    "extraction.py": {"render_extraction", "_answer_payload"},
    "knowledge.py": {"_grounded_qa", "_summary"},
    "reasoning.py": {"_reasoning", "_critique", "_brainstorm"},
    "safety.py": {"_safety"},
}

# Snapshot ceilings for long, visible strings that contain no scenario
# interpolation. A refactor may lower these budgets, but new static prose must
# not silently raise them instead of adding a semantic variable_by dimension.
_LONG_STATIC_TEXT_LINE_BUDGET = {
    "action.py": 17,
    "communication.py": 132,
    "extraction.py": 2,
    "knowledge.py": 78,
    "reasoning.py": 25,
    "safety.py": 44,
}


def _task_row(family: str, domain: str, scenario: str) -> dict:
    intent = next(iter(intent_contract_catalog()[family]))
    return {
        "scenario_id": f"scenario:{scenario}",
        "family": family,
        "domain": domain,
        "intent": intent,
        "constraint": "Keep the action bounded.",
        "state": "",
        "semantic_payload": json.dumps(
            {
                "subject": domain.replace("_", " "),
                "domain_context": f"Context for {domain}.",
            }
        ),
    }


def _rows_for_family(family: str, domains: tuple[str, ...]) -> list[dict]:
    rows = []
    for domain in domains:
        for index in range(_SCENARIOS_PER_DOMAIN):
            digest = hashlib.sha256(f"{family}:{domain}:{index}".encode()).hexdigest()
            row = _task_row(family, domain, digest[:24])
            hand = deal_task_hand(row, index % 8)
            rows.append(
                {
                    "example_id": f"{family}:{domain}:{index}",
                    "task": family,
                    "split": "train",
                    "prompt": hand.data,
                    "response": hand.answer,
                    "source_keys": [f"{family}:{domain}:{index}"],
                }
            )
    return rows


def _long_static_text_lines(path: Path, renderers: set[str]) -> set[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    docstrings = {
        node.body[0].value
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    lines: set[int] = set()
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Constant)
            or not isinstance(node.value, str)
            or node in docstrings
            or len(node.value.split()) < 8
            or isinstance(parents.get(node), ast.JoinedStr)
        ):
            continue
        owner = parents.get(node)
        while owner is not None and not isinstance(
            owner, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            owner = parents.get(owner)
        if owner is not None and owner.name in renderers:
            lines.add(node.lineno)
    return lines


def test_long_static_task_text_budget_does_not_increase() -> None:
    tasks_root = Path(__file__).parents[1] / "src" / "complexity_card_corpus" / "tasks"
    observed = {
        filename: len(_long_static_text_lines(tasks_root / filename, renderers))
        for filename, renderers in _TASK_RENDERERS_BY_FILE.items()
    }
    excess = {
        filename: (count, _LONG_STATIC_TEXT_LINE_BUDGET[filename])
        for filename, count in observed.items()
        if count > _LONG_STATIC_TEXT_LINE_BUDGET[filename]
    }
    assert not excess, {"excess": excess, "observed": observed}
    assert sum(observed.values()) <= sum(_LONG_STATIC_TEXT_LINE_BUDGET.values()), observed


@pytest.mark.parametrize("family", sorted(_DOMAINS_BY_FAMILY))
def test_regrounded_family_passes_response_only_repetition_audit(
    family: str,
) -> None:
    rows = _rows_for_family(family, _DOMAINS_BY_FAMILY[family])
    audit = audit_rows_quality(
        rows,
        input_label=f"{family}-diversity-regression",
        sample_size=len(rows),
        workers=2,
    )
    repetition = audit["response_only_repetition"]
    assert repetition["exact_duplicate_rows"] == 0, (family, repetition)
    assert repetition["near_duplicate_ratio"] < 0.05, (family, repetition)


_UNRENDERED_PLACEHOLDER = re.compile(r"\{[^{}]+\}")


def test_no_unrendered_template_placeholders_leak_into_answers() -> None:
    """A pool entry missing its ``f`` prefix renders a literal "{name}"
    instead of interpolating the scenario value (regression: this happened
    across two families before the string reservoirs were audited by hand).
    """
    offenders = []
    for family, domains in _DOMAINS_BY_FAMILY.items():
        for domain in domains:
            for index in range(4):
                row = _task_row(family, domain, f"placeholder-check-{index}")
                hand = deal_task_hand(row, index)
                if _UNRENDERED_PLACEHOLDER.search(hand.answer):
                    offenders.append((family, domain, hand.answer))
                if "Variable:" in hand.answer:
                    offenders.append((family, domain, hand.answer))
    assert not offenders, offenders


def test_regrounded_family_scenarios_are_not_all_identical_per_domain() -> None:
    """Cheap, fast complement to the statistical audit above: two different
    scenarios in the same domain must not render the exact same answer.

    Scenario ids must vary in their first six characters: ``_code()`` (see
    tasks/core.py) derives its high-cardinality per-scenario token from
    ``scenario_id.split(":")[-1][:6]``, so a shared literal prefix like
    ``uniqueness-`` collapses every draw onto the same code and silently
    defeats this check.
    """
    for family, domains in _DOMAINS_BY_FAMILY.items():
        for domain in domains:
            answers = {
                deal_task_hand(
                    _task_row(
                        family,
                        domain,
                        hashlib.sha256(f"{domain}:{index}".encode()).hexdigest()[:24],
                    ),
                    index % 8,
                ).answer
                for index in range(30)
            }
            assert len(answers) >= 25, (family, domain, len(answers))
