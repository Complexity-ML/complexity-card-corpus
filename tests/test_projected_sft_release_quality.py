from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
import json

import pyarrow.parquet as pq
import pytest

from complexity_card_corpus.conversational import render_casual_conversation_rows


PROJECTED_SFT = Path("build/post-training-v18/projected.parquet")
CASUAL_REGISTRY = Path("data/conversation/original/casual-conversation-decks-v1.json")
MAX_NORMALIZED_SENTENCE_SHARE = 0.05


def test_projected_sft_manifest_passes_every_release_quality_gate() -> None:
    manifest_path = PROJECTED_SFT.parent / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("build the projected SFT release before running release-quality tests")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    release_quality = manifest["release_quality"]

    assert release_quality["ready"] is True
    assert all(release_quality["checks"].values())


def test_all_fourteen_core_families_have_multiscale_repetition_gates() -> None:
    manifest_path = PROJECTED_SFT.parent / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("build the unified V18 SFT release before testing repetition")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = manifest["release_quality"]["train_family_counts"]
    core_families = set(counts) - {"casual_conversation"}
    audit = manifest["model_facing_repetition_quality"]

    assert len(core_families) == 14
    assert core_families <= set(audit["tasks"])
    required_dimensions = {
        "response_exact",
        "response_structure",
        "response_opening_3",
        "response_opening_5",
        "response_opening_8",
        "response_closing_3",
        "response_closing_5",
        "response_closing_8",
        "response_sentence",
        "response_span_8",
    }
    failures = {}
    for family in sorted(core_families):
        family_audit = audit["tasks"][family]
        dimensions = family_audit["dimensions"]
        assert family_audit["audited"] is True
        if family != "extraction_classification":
            assert required_dimensions <= set(dimensions)
        if not family_audit["supervised_passed"]:
            failures[family] = {
                name: metric
                for name, metric in dimensions.items()
                if name.startswith("response_")
                and metric["audited"]
                and not metric["passed"]
            }

    assert failures == {}


def test_projected_sft_includes_audited_casual_conversation() -> None:
    manifest_path = PROJECTED_SFT.parent / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("build the projected SFT release before running release-quality tests")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality = manifest["casual_conversation_quality"]

    assert manifest["release_quality"]["train_family_counts"][
        "casual_conversation"
    ] == 28_728
    assert quality["present"] is True
    assert quality["passed"] is True
    assert quality["violations"] == []
    assert set(quality["final_sentence_counts"]) <= {"1", "2", "3"}


def test_projected_sft_preserves_original_casual_turns_verbatim() -> None:
    if not PROJECTED_SFT.exists():
        pytest.skip("build the unified V18 SFT release before testing projection")
    rendered_rows, _summary = render_casual_conversation_rows(CASUAL_REGISTRY)
    source_rows = {
        row["example_id"]: row["messages"]
        for row in rendered_rows
        if row["split"] == "train"
    }
    projected_rows = {
        row["example_id"]: row["messages"]
        for row in pq.read_table(
            PROJECTED_SFT,
            columns=["example_id", "task", "messages"],
        ).to_pylist()
        if row["task"] == "casual_conversation"
    }

    assert projected_rows == source_rows


@pytest.fixture(scope="module")
def projected_rows() -> list[dict]:
    if not PROJECTED_SFT.exists():
        pytest.skip("build the projected SFT release before running release-quality tests")
    return pq.read_table(
        PROJECTED_SFT,
        columns=["example_id", "task", "messages"],
    ).to_pylist()


def _prompt(row: dict) -> str:
    return "\n".join(
        message["content"] for message in row["messages"][:-1]
    )


def _answer(row: dict) -> str:
    return row["messages"][-1]["content"]


def _sentence_count(text: str) -> int:
    return len(
        [
            sentence
            for sentence in re.split(r"(?<=[.!?])(?:\s+|$)", text.strip())
            if sentence.strip()
        ]
    )


def _normalized_sentences(text: str) -> set[str]:
    normalized: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = sentence.lower()
        sentence = re.sub(r"\b[0-9a-f]{6,}\b", "<id>", sentence)
        sentence = re.sub(r"\$?\d+(?:[.,:/]\d+)*(?:%|°c)?", "<n>", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip(" -—•\t")
        if len(sentence) >= 25:
            normalized.add(sentence)
    return normalized


def test_projected_sft_respects_exact_sentence_contracts(
    projected_rows: list[dict],
) -> None:
    requested = [
        row
        for row in projected_rows
        if row["task"] == "critique_revision"
        and "exactly two sentences" in _prompt(row).lower()
    ]
    offenders = [
        (row["example_id"], _sentence_count(_answer(row)))
        for row in requested
        if _sentence_count(_answer(row)) != 3
    ]

    assert requested, "the release must exercise the exact-two-sentence contract"
    assert offenders == []


def test_projected_sft_does_not_invent_timing_constraints(
    projected_rows: list[dict],
) -> None:
    offenders = [
        row["example_id"]
        for row in projected_rows
        if row["task"] == "practical_action"
        and (
            match := re.search(
                r"\bday (\d+) cutoff\b",
                _answer(row),
                re.IGNORECASE,
            )
        )
        and not re.search(
            rf"\bbefore day {match.group(1)}\b",
            _prompt(row),
            re.IGNORECASE,
        )
    ]

    assert offenders == []


def test_projected_sft_keeps_safety_scenario_and_domain_coherent(
    projected_rows: list[dict],
) -> None:
    offenders = [
        row["example_id"]
        for row in projected_rows
        if row["task"] == "safety_uncertainty"
        and "animal bite" in _prompt(row).lower()
        and "chest pressure" in _prompt(row).lower()
    ]

    assert offenders == []


def test_projected_sft_contains_no_known_grammar_or_spacing_defect(
    projected_rows: list[dict],
) -> None:
    patterns = {
        "because_it_which": re.compile(r"\bbecause it which\b", re.IGNORECASE),
        "double_article": re.compile(
            r"\b(?:the|a|an) (?:the|a|an)\b",
            re.IGNORECASE,
        ),
        "lowercase_after_period": re.compile(r"\. [a-z]"),
        "missing_space_after_period": re.compile(
            r"(?<=[a-z0-9'\"])\.(?=[A-Z])"
        ),
        "double_space": re.compile(r" {2,}"),
        "capitalized_article_after_note_that": re.compile(
            r"\bnote that A (?:reverse|division|backward)\b"
        ),
        "orphan_dash_label": re.compile(r"(?:^|\n)- — [A-Z]"),
        "orphan_dash_punctuation": re.compile(r"\s[—-]\s*\."),
    }
    offenders: dict[str, list[str]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for row in projected_rows:
        answer = _answer(row)
        for name, pattern in patterns.items():
            if pattern.search(answer):
                counts[name] += 1
                if len(offenders[name]) < 5:
                    offenders[name].append(row["example_id"])

    assert counts == Counter(), {
        "counts": dict(counts),
        "examples": dict(offenders),
    }


def test_projected_brainstorming_contains_no_control_check_sentences(
    projected_rows: list[dict],
) -> None:
    control_checks = re.compile(
        r"(?:options remain bounded by the explicit brief|"
        r"each retained option satisfies the stated criteria|"
        r"option is concrete enough to test first)",
        re.IGNORECASE,
    )
    offenders = [
        row["example_id"]
        for row in projected_rows
        if row["task"] == "brainstorming_creativity"
        and control_checks.search(_answer(row))
    ]

    assert offenders == []


def test_projected_sft_contains_no_internal_card_or_debug_labels(
    projected_rows: list[dict],
) -> None:
    forbidden = re.compile(
        r"(?:\b(?:Hand|For hand)\s+[A-Z0-9]+\b|"
        r"\bCandidate task\b|"
        r"(?:^|\n)(?:Option set|Idea generation|Brief comparison|"
        r"Criteria review|Fit check|Pilot choice)\s*(?::|—)|"
        r"\b(?:event_plan|feature_ideas)\s+\d+\b)",
        re.IGNORECASE,
    )
    offenders = [
        row["example_id"]
        for row in projected_rows
        if forbidden.search(_prompt(row) + "\n" + _answer(row))
    ]

    assert offenders == []


def test_projected_sft_normalized_sentence_share_stays_below_five_percent(
    projected_rows: list[dict],
) -> None:
    row_counts: Counter[str] = Counter()
    sentence_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in projected_rows:
        task = row["task"]
        row_counts[task] += 1
        sentence_counts[task].update(_normalized_sentences(_answer(row)))

    offenders: dict[str, dict[str, object]] = {}
    for task, counts in sentence_counts.items():
        sentence, count = counts.most_common(1)[0]
        share = count / row_counts[task]
        if share > MAX_NORMALIZED_SENTENCE_SHARE:
            offenders[task] = {
                "share": round(share, 6),
                "count": count,
                "rows": row_counts[task],
                "sentence": sentence,
            }

    assert offenders == {}


def test_projected_sft_has_no_reasoning_composition_artifacts(
    projected_rows: list[dict],
) -> None:
    offenders: list[tuple[str, str]] = []
    for row in projected_rows:
        if row["task"] not in {"planning_comparison", "reasoning_verification"}:
            continue
        answer = _answer(row)
        lowered = answer.lower()
        if (
            ".." in answer
            or "confirms that a separate numerical route" in lowered
            or "establishes that a separate numerical route" in lowered
            or re.search(r"\bAn (?:after|alone|and|at|is)\b", answer)
        ):
            offenders.append((row["example_id"], answer))

    assert offenders == []
