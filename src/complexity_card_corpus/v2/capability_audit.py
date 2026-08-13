from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    selectors: tuple[tuple[str, frozenset[str] | None], ...]
    minimum_examples: int
    minimum_domains: int
    minimum_unique_prompt_share: float = 0.25
    minimum_unique_response_share: float = 0.10

    def matches(self, task: str, domain: str) -> bool:
        return any(
            task == selected_task
            and (selected_domains is None or domain in selected_domains)
            for selected_task, selected_domains in self.selectors
        )


DEFAULT_CAPABILITY_SPECS = (
    CapabilitySpec(
        "direct_safety",
        (
            ("safety_uncertainty", None),
            (
                "casual_conversation",
                frozenset(
                    {
                        "account_safety",
                        "chemical_safety",
                        "crisis_support",
                        "emergency_health",
                    }
                ),
            ),
        ),
        minimum_examples=3_000,
        minimum_domains=8,
    ),
    CapabilitySpec(
        "small_arithmetic",
        (
            (
                "casual_conversation",
                frozenset({"addition", "subtraction", "multiplication", "division"}),
            ),
        ),
        minimum_examples=2_000,
        minimum_domains=4,
    ),
    CapabilitySpec(
        "summarization",
        (("summarization_synthesis", None),),
        minimum_examples=4_000,
        minimum_domains=8,
    ),
    CapabilitySpec(
        "writing_transformation",
        (("writing_transformation", None),),
        minimum_examples=4_000,
        minimum_domains=8,
    ),
    CapabilitySpec(
        "multi_constraint_following",
        (
            (
                "casual_conversation",
                frozenset(
                    {
                        "bullet_constraints",
                        "length_constraints",
                        "sentence_constraints",
                        "structured_constraints",
                    }
                ),
            ),
        ),
        minimum_examples=4_000,
        minimum_domains=4,
    ),
    CapabilitySpec(
        "concept_definitions",
        (("casual_conversation", frozenset({"concept_definition"})),),
        minimum_examples=1_000,
        minimum_domains=1,
    ),
    CapabilitySpec(
        "general_facts",
        (("casual_conversation", frozenset({"general_knowledge"})),),
        minimum_examples=1_000,
        minimum_domains=1,
    ),
    CapabilitySpec(
        "reflective_conversation",
        (
            ("conversation_empathy", None),
            ("casual_conversation", frozenset({"social_reflection"})),
        ),
        minimum_examples=2_000,
        minimum_domains=8,
    ),
    CapabilitySpec(
        "neutral_greeting",
        (("casual_conversation", frozenset({"social_greeting"})),),
        minimum_examples=1_000,
        minimum_domains=1,
    ),
)


def _normalized(value: object) -> str:
    return _SPACE.sub(" ", str(value).strip().casefold())


def audit_v2_capability_coverage(
    rows: Iterable[dict[str, Any]],
    *,
    specs: tuple[CapabilitySpec, ...] = DEFAULT_CAPABILITY_SPECS,
) -> dict[str, Any]:
    """Measure whether each promoted behavior has learnable corpus support."""

    selected: dict[str, list[tuple[str, str, str]]] = {
        spec.name: [] for spec in specs
    }
    domains: dict[str, Counter[str]] = {
        spec.name: Counter() for spec in specs
    }
    for row in rows:
        if str(row.get("split", "train")) != "train":
            continue
        task = str(row.get("task", ""))
        domain = str(row.get("domain", ""))
        prompt = _normalized(row.get("prompt", ""))
        response = _normalized(row.get("final_response", row.get("response", "")))
        for spec in specs:
            if spec.matches(task, domain):
                selected[spec.name].append((prompt, response, domain))
                domains[spec.name][domain] += 1

    capabilities: dict[str, dict[str, Any]] = {}
    violations: list[str] = []
    for spec in specs:
        examples = selected[spec.name]
        count = len(examples)
        unique_prompts = len({prompt for prompt, _response, _domain in examples})
        unique_responses = len({response for _prompt, response, _domain in examples})
        prompt_share = unique_prompts / max(1, count)
        response_share = unique_responses / max(1, count)
        failures = []
        if count < spec.minimum_examples:
            failures.append("examples")
        if len(domains[spec.name]) < spec.minimum_domains:
            failures.append("domains")
        if prompt_share < spec.minimum_unique_prompt_share:
            failures.append("prompt_diversity")
        if response_share < spec.minimum_unique_response_share:
            failures.append("response_diversity")
        if failures:
            violations.append(f"{spec.name}: " + ", ".join(failures))
        capabilities[spec.name] = {
            "passed": not failures,
            "failures": failures,
            "examples": count,
            "minimum_examples": spec.minimum_examples,
            "domains": dict(sorted(domains[spec.name].items())),
            "domain_count": len(domains[spec.name]),
            "minimum_domains": spec.minimum_domains,
            "unique_prompts": unique_prompts,
            "unique_prompt_share": round(prompt_share, 6),
            "minimum_unique_prompt_share": spec.minimum_unique_prompt_share,
            "unique_responses": unique_responses,
            "unique_response_share": round(response_share, 6),
            "minimum_unique_response_share": spec.minimum_unique_response_share,
        }

    return {
        "format": "complexity-card-corpus-v2-capability-coverage-v1",
        "passed": not violations,
        "violations": violations,
        "capabilities": capabilities,
    }


__all__ = (
    "CapabilitySpec",
    "DEFAULT_CAPABILITY_SPECS",
    "audit_v2_capability_coverage",
)
