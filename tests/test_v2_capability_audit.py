from __future__ import annotations

from complexity_card_corpus.v2.capability_audit import (
    CapabilitySpec,
    audit_v2_capability_coverage,
)


def _row(index: int, *, task: str = "demo", domain: str = "alpha") -> dict:
    return {
        "split": "train",
        "task": task,
        "domain": domain,
        "prompt": f"Prompt {index}",
        "response": f"Response {index}",
        "final_response": f"Response {index}",
    }


def test_capability_coverage_requires_volume_domains_and_surface_diversity() -> None:
    spec = CapabilitySpec(
        "demo_capability",
        (("demo", None),),
        minimum_examples=4,
        minimum_domains=2,
        minimum_unique_prompt_share=0.75,
        minimum_unique_response_share=0.75,
    )
    rows = [_row(0), _row(0), _row(0)]

    audit = audit_v2_capability_coverage(rows, specs=(spec,))

    assert audit["passed"] is False
    assert audit["capabilities"]["demo_capability"]["failures"] == [
        "examples",
        "domains",
        "prompt_diversity",
        "response_diversity",
    ]


def test_capability_coverage_accepts_supported_behavior_and_ignores_eval() -> None:
    spec = CapabilitySpec(
        "demo_capability",
        (("demo", frozenset({"alpha", "beta"})),),
        minimum_examples=4,
        minimum_domains=2,
        minimum_unique_prompt_share=1.0,
        minimum_unique_response_share=1.0,
    )
    rows = [
        _row(0, domain="alpha"),
        _row(1, domain="alpha"),
        _row(2, domain="beta"),
        _row(3, domain="beta"),
        {**_row(4, domain="beta"), "split": "validation"},
    ]

    audit = audit_v2_capability_coverage(rows, specs=(spec,))

    assert audit["passed"] is True
    assert audit["capabilities"]["demo_capability"]["examples"] == 4
    assert audit["capabilities"]["demo_capability"]["domain_count"] == 2
