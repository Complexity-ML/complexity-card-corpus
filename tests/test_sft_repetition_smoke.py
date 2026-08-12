from __future__ import annotations

from pathlib import Path

import pytest

from complexity_card_corpus.posttrain.build import _parallel_conversation_rows
from complexity_card_corpus.scenarios import compile_scenarios, load_scenario_registry
from complexity_card_corpus.sft.evaluation import audit_sft_repetition_quality
from complexity_card_corpus.sft.projection import _project_sft_exchange


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/scenario-forge/scenario-forge-v1.json"
FAST_TARGET_SCENARIOS = 2_800
FAST_VARIANTS_PER_SCENARIO = 2
FAST_MAXIMUM_RESPONSE_SHARE = 0.08


@pytest.fixture(scope="module")
def fast_projection_sample() -> tuple[list[dict], dict]:
    """Render a small stratified source sample for rapid edit/audit loops.

    The release audit remains authoritative at 4.5% on up to 10k rows per
    family.  This smoke layer deliberately uses an 8% threshold so a roughly
    5.6k-row run catches structural collisions without pretending to replace
    the final population check.
    """

    registry = load_scenario_registry(REGISTRY)
    scenarios = compile_scenarios(
        registry,
        target_scenarios=FAST_TARGET_SCENARIOS,
    )
    rows = _parallel_conversation_rows(
        scenarios,
        FAST_VARIANTS_PER_SCENARIO,
        vocabulary_placements=[],
        workers=1,
    )
    projected = []
    for row in rows:
        prompt, target, cards = _project_sft_exchange(
            row["messages"],
            example_id=row["example_id"],
            task=row["task"],
            answer_json=row["answer_json"],
            reasoning_envelope_version="v18",
            reasoning_seed=f"fast-audit:{row['example_id']}",
        )
        projected.append(
            {
                **row,
                "_projected_prompt": prompt,
                "_projected_target": target,
                "_conditioning_cards": cards,
            }
        )
    return (
        projected,
        audit_sft_repetition_quality(
            projected,
            maximum_share=FAST_MAXIMUM_RESPONSE_SHARE,
            minimum_examples=200,
        ),
    )


def test_fast_projection_audits_all_fourteen_core_families(
    fast_projection_sample: tuple[list[dict], dict],
) -> None:
    _projected, fast_projection_audit = fast_projection_sample
    assert len(fast_projection_audit["tasks"]) == 14
    assert all(
        task["audited"] for task in fast_projection_audit["tasks"].values()
    )


def test_fast_projection_has_no_major_response_collision(
    fast_projection_sample: tuple[list[dict], dict],
) -> None:
    _projected, fast_projection_audit = fast_projection_sample
    failures = {}
    for family, family_audit in fast_projection_audit["tasks"].items():
        response_failures = {
            name: {
                "share": dimension["maximum_share"],
                "signature": dimension["most_common_signature"],
            }
            for name, dimension in family_audit["dimensions"].items()
            if name.startswith("response_")
            and dimension["audited"]
            and not dimension["passed"]
        }
        if response_failures:
            failures[family] = response_failures

    if failures:
        report = ["major response collisions in fast SFT projection:"]
        for family, dimensions in sorted(failures.items()):
            rendered = "; ".join(
                f"{name}={metric['share']:.2%} [{metric['signature']}]"
                for name, metric in sorted(dimensions.items())
            )
            report.append(f"- {family}: {rendered}")
        pytest.fail("\n".join(report), pytrace=False)


def test_fast_projection_has_no_internal_card_rubric_leaks(
    fast_projection_sample: tuple[list[dict], dict],
) -> None:
    projected, _audit = fast_projection_sample
    forbidden = (
        "hand ",
        "next step:",
        "owner:",
        "timing:",
        "core idea:",
        "example:",
        "check:",
        "decision:",
        "action:",
        "open point:",
        "open item:",
        "weakness:",
        "revision:",
        "immediate action:",
        "boundary:",
        "sequence:",
        "fallback trigger:",
        "revised text:",
    )
    leaks = []
    for row in projected:
        lowered = row["_projected_target"].lower()
        leaked = next((phrase for phrase in forbidden if phrase in lowered), None)
        if leaked is not None:
            leaks.append((row["example_id"], row["task"], leaked))
    assert not leaks, leaks[:20]
