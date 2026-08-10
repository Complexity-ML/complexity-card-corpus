from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import REVIEW_GRADES, _REVIEW_GRADE_VALUES, _REVIEW_STATUSES


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
                available = sum(len(values) for values in strata.values())
                raise ValueError(
                    "insufficient review candidates for "
                    f"{family}: found {available} complete instruct/chat "
                    f"scenarios but {quota} are required; increase "
                    "variants_per_scenario, raise max_examples_per_family, "
                    "or lower review_scenarios"
                )
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
