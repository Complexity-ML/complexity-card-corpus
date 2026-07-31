from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .build import file_sha256
from .english_morphology import correct_indefinite_articles
from .instruct import INSTRUCTION_SCHEMA


DATASET_ID = "complexity-original-post-training-v1"
DATASET_LICENSE = "CC BY-NC 4.0"
DATASET_SOURCE = "Complexity original Scenario Forge conversations"
REVIEW_GRADES = (
    "semantic_accuracy",
    "constraint_following",
    "language_quality",
    "individualization",
    "safety",
)
_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")

_INTENT_FIELD = {
    "practical_action": "requested_action",
    "explanation_learning": "learning_goal",
    "troubleshooting": "diagnostic_goal",
    "writing_transformation": "transformation",
    "planning_comparison": "planning_goal",
    "conversation_empathy": "conversational_goal",
    "safety_uncertainty": "safe_goal",
}

_OPENINGS = (
    "I would begin with the verified facts",
    "The safest useful approach is evidence first",
    "A grounded response can proceed in three bounded parts",
    "The clearest next move is to separate facts from assumptions",
    "I would keep the answer practical and reviewable",
    "This can be handled without guessing",
    "The response should start from what is currently supported",
    "A careful answer can remain concise",
    "I would preserve control while moving the task forward",
    "The useful path is narrow but actionable",
    "A responsible answer should make its basis explicit",
    "The next step should remain reversible where possible",
)

_ACKNOWLEDGEMENTS = (
    "I understand the decision you are trying to make, and I will keep the answer tied to evidence.",
    "That gives us a concrete decision point, so I will avoid filling any gap with an assumption.",
    "I can help with that by separating the immediate task from the final acceptance check.",
    "The situation is specific enough to plan a bounded response without pretending the uncertainty is resolved.",
    "I will focus on the requested result, the available evidence, and a safe fallback if support is missing.",
    "This calls for a practical answer with a visible decision rule rather than a broad recommendation.",
    "I will keep the response narrow enough to verify and useful enough to support the next decision.",
    "The key is to move from the known facts to a bounded action without hiding the remaining uncertainty.",
    "I can work from that update while keeping the final check and fallback explicit.",
    "That provides a clear starting point, and the response can preserve control throughout the decision.",
    "I will treat the evidence as the basis for action and leave unsupported details unresolved.",
    "The answer can stay grounded by naming one objective, one limit, and one acceptance check.",
)

_PROMPT_REQUESTS = (
    "Please give me a grounded response that helps me {intent} for {subject}.",
    "What practical response would help me {intent} for {subject} without guessing?",
    "Can you help me {intent} for {subject} while keeping the answer evidence-based?",
    "I need a bounded way to {intent} for {subject}; what should the response cover?",
    "Please outline how to {intent} for {subject} using only supported details.",
    "What is a reviewable approach to {intent} for {subject} in this situation?",
    "Help me {intent} for {subject} without turning uncertainty into a claim.",
    "Which response would {intent} for {subject} and still preserve user control?",
    "I want to {intent} for {subject}; how can the answer remain practical and grounded?",
    "Please propose a careful response to {intent} for {subject} from the available facts.",
    "How should an assistant {intent} for {subject} while making its reasoning inspectable?",
    "What concise answer can {intent} for {subject} and retain a safe fallback?",
)

_FOLLOW_UPS = (
    "The relevant context is that {context}. What response would {intent} for {subject} without overclaiming?",
    "The surrounding facts are that {context}. How can the response {intent} for {subject} responsibly?",
    "One context point matters here: {context}. What is a grounded way to {intent} for {subject}?",
    "The decision sits within this setting: {context}. How should the assistant {intent} for {subject}?",
    "The available background is that {context}. Which bounded response would {intent} for {subject}?",
    "Keep this context in view: {context}. What practical answer can {intent} for {subject}?",
    "The evidence comes from this setting: {context}. How can we {intent} for {subject} without guessing?",
    "This background limits the answer: {context}. What response can {intent} for {subject} and remain reviewable?",
    "The request is grounded in this fact: {context}. Which next response should {intent} for {subject}?",
    "Use this context as the boundary for reasoning: {context}. How would you {intent} for {subject}?",
    "The known setting is that {context}. What concise answer would {intent} for {subject} safely?",
    "The response must reflect this context: {context}. How can it {intent} for {subject} with visible support?",
)


def _stable_index(value: str, size: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % size


def _lower_first(value: str) -> str:
    value = value.strip()
    return value[:1].lower() + value[1:] if value else value


def _intent(payload: dict[str, str], family: str) -> str:
    return payload[_INTENT_FIELD[family]].rstrip(".")


def _render_final(row: dict[str, Any], variant: int) -> str:
    payload = json.loads(row["semantic_payload"])
    intent = _intent(payload, row["family"])
    subject = payload["subject"]
    constraint = row["constraint"].rstrip(".")
    outcome = row["desired_outcome"].rstrip(".")
    fallback = row["fallback"].rstrip(".")
    opening = _OPENINGS[
        _stable_index(f"final:{row['scenario_id']}:{variant}", len(_OPENINGS))
    ]
    structures = (
        f"{opening}. I would {intent} for {subject} while preserving this boundary: {constraint}. The answer is complete when {_lower_first(outcome)}. If the evidence cannot support that result, {_lower_first(fallback)}.",
        f"{opening}. For {subject}, the immediate objective is to {intent}. One limit remains firm: {constraint}. Success means that {_lower_first(outcome)}. If that cannot be verified, {_lower_first(fallback)}.",
        f"{opening}. The next response should {intent} for {subject}. It must respect this condition: {constraint}. The acceptance check is that {_lower_first(outcome)}. Without enough support, {_lower_first(fallback)}.",
        f"{opening}. I would {intent} for {subject} through a reviewable step. The controlling boundary is: {constraint}. A valid result establishes that {_lower_first(outcome)}. If uncertainty remains material, {_lower_first(fallback)}.",
        f"{opening}. My first step would be to {intent} for {subject}. I would treat this as a hard boundary: {constraint}. Completion should show that {_lower_first(outcome)}. Lacking that evidence, {_lower_first(fallback)}.",
        f"{opening}. For {subject}, I would use confirmed details to {intent}. The response cannot override this rule: {constraint}. Its final check should establish that {_lower_first(outcome)}. If the check fails, {_lower_first(fallback)}.",
        f"{opening}. A bounded plan can {intent} for {subject}. It must not violate this condition: {constraint}. The resulting evidence should demonstrate that {_lower_first(outcome)}. If it does not, {_lower_first(fallback)}.",
        f"{opening}. I would keep ownership visible while trying to {intent} for {subject}. This requirement remains fixed: {constraint}. We can accept the answer when {_lower_first(outcome)}. Until then, {_lower_first(fallback)}.",
        f"{opening}. The practical route is to {intent} for {subject}. Every option remains governed by this boundary: {constraint}. The selected route must show that {_lower_first(outcome)}. If none does, {_lower_first(fallback)}.",
        f"{opening}. For this case, efforts to {intent} for {subject} should follow the available evidence. The limiting rule is: {constraint}. Resolution requires proof that {_lower_first(outcome)}. If proof is missing, {_lower_first(fallback)}.",
        f"{opening}. I would make one reversible move to {intent} for {subject}. The move must preserve this condition: {constraint}. Its result is acceptable if {_lower_first(outcome)}. If that remains unclear, {_lower_first(fallback)}.",
        f"{opening}. The answer can {intent} for {subject} by naming facts, action, and verification. It must keep this boundary intact: {constraint}. Verification succeeds when {_lower_first(outcome)}. Otherwise, {_lower_first(fallback)}.",
    )
    selected = structures[
        _stable_index(f"structure:{row['scenario_id']}:{variant}", len(structures))
    ]
    return correct_indefinite_articles(selected)


def _render_messages(row: dict[str, Any], variant: int) -> list[dict[str, str]]:
    payload = json.loads(row["semantic_payload"])
    intent = _intent(payload, row["family"])
    subject = payload["subject"]
    trigger = row["trigger"]
    if variant % 2 == 0:
        request = _PROMPT_REQUESTS[
            _stable_index(f"prompt:{row['scenario_id']}:{variant}", len(_PROMPT_REQUESTS))
        ].format(intent=intent, subject=subject)
        prompt = f"{trigger} {request}"
        return [
            {"role": "user", "content": correct_indefinite_articles(prompt)},
            {"role": "assistant", "content": _render_final(row, variant)},
        ]

    acknowledgement = _ACKNOWLEDGEMENTS[
        _stable_index(f"ack:{row['scenario_id']}:{variant}", len(_ACKNOWLEDGEMENTS))
    ]
    context = payload["domain_context"].rstrip(".")
    follow_up = _FOLLOW_UPS[
        _stable_index(f"follow-up:{row['scenario_id']}:{variant}", len(_FOLLOW_UPS))
    ].format(context=_lower_first(context), intent=intent, subject=subject)
    return [
        {"role": "user", "content": trigger},
        {"role": "assistant", "content": acknowledgement},
        {"role": "user", "content": correct_indefinite_articles(follow_up)},
        {"role": "assistant", "content": _render_final(row, variant)},
    ]


def _render_transcript(messages: list[dict[str, str]]) -> str:
    labels = {"user": "User", "assistant": "Assistant"}
    return "\n".join(
        f"{labels[message['role']]}: {message['content']}" for message in messages
    )


def _conversation_rows(
    scenarios: list[dict[str, Any]], variants_per_scenario: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for variant in range(variants_per_scenario):
            messages = _render_messages(scenario, variant)
            rendered = _render_transcript(messages)
            suffix = hashlib.sha256(
                f"{scenario['scenario_id']}:{variant}:{rendered}".encode()
            ).hexdigest()[:20]
            answer = {
                "scenario_id": scenario["scenario_id"],
                "family": scenario["family"],
                "domain": scenario["domain"],
                "intent": scenario["intent"],
                "risk_level": scenario["risk_level"],
                "state": scenario["state"],
                "constraint": scenario["constraint"],
                "response_contract": scenario["response_contract"],
                "model_generated_dialogue": False,
            }
            rows.append(
                {
                    "example_id": f"post-training:{suffix}",
                    "task": scenario["family"],
                    "mode": "instruct" if len(messages) == 2 else "chat",
                    "difficulty": (
                        "hard" if scenario["risk_level"] == "high" else "medium"
                    ),
                    "dataset_id": DATASET_ID,
                    "domain": scenario["domain"],
                    "language": "en",
                    "split": scenario["split"],
                    "messages": messages,
                    "prompt": messages[0]["content"],
                    "response": messages[-1]["content"],
                    "rendered_text": rendered,
                    "source_keys": [scenario["scenario_id"]],
                    "evidence": [],
                    "answer_json": json.dumps(answer, sort_keys=True),
                    "source": DATASET_SOURCE,
                    "source_urls": [],
                    "license": DATASET_LICENSE,
                    "version": "1.0.0",
                }
            )
    return sorted(rows, key=lambda row: row["example_id"])


def _audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rendered = [row["rendered_text"] for row in rows]
    responses = [row["response"] for row in rows]
    train_cards: set[str] = set()
    validation_cards: set[str] = set()
    exact_anchor_rows = 0
    for row in rows:
        answer = json.loads(row["answer_json"])
        target = validation_cards if row["split"] == "validation" else train_cards
        target.add(answer["scenario_id"])
        normalized = row["rendered_text"].lower()
        if all(
            normalized.count(answer[field].rstrip(".").lower()) == 1
            for field in ("state", "constraint")
        ):
            exact_anchor_rows += 1
    overlap = train_cards & validation_cards
    if overlap:
        raise ValueError("source scenarios leak across post-training splits")
    if len(set(rendered)) != len(rendered):
        raise ValueError("duplicate post-training conversations")
    if any(row["license"] != DATASET_LICENSE for row in rows):
        raise ValueError("post-training license mismatch")
    if exact_anchor_rows != len(rows):
        raise ValueError("state and constraint must each appear exactly once")
    unique_final_response_ratio = len(set(responses)) / len(responses)
    if unique_final_response_ratio < 0.95:
        raise ValueError(
            "post-training final responses are not sufficiently individualized: "
            f"{unique_final_response_ratio:.3f}"
        )
    messages = [message["content"] for row in rows for message in row["messages"]]
    ngram_stats: dict[int, dict[str, float | int]] = {}
    for size in (4, 8):
        counts: Counter[tuple[str, ...]] = Counter()
        windows = 0
        for message in messages:
            tokens = [match.group(0).lower() for match in _WORD.finditer(message)]
            grams = [
                tuple(tokens[index : index + size])
                for index in range(len(tokens) - size + 1)
            ]
            counts.update(grams)
            windows += len(grams)
        ngram_stats[size] = {
            "windows": windows,
            "unique": len(counts),
            "unique_rate": round(len(counts) / windows, 6),
            "maximum_repetitions": max(counts.values(), default=0),
        }
    return {
        "rows": len(rows),
        "source_cards": len(train_cards | validation_cards),
        "split_holdout_unit": "scenario_id",
        "source_card_split_overlap": len(overlap),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "family_counts": dict(sorted(Counter(row["task"] for row in rows).items())),
        "mode_counts": dict(sorted(Counter(row["mode"] for row in rows).items())),
        "unique_rendered_ratio": len(set(rendered)) / len(rendered),
        "unique_final_response_ratio": unique_final_response_ratio,
        "model_generated_dialogue_rows": 0,
        "single_state_and_constraint_ratio": exact_anchor_rows / len(rows),
        "four_gram_stats": ngram_stats[4],
        "eight_gram_stats": ngram_stats[8],
    }


def _review_sample(
    rows: list[dict[str, Any]], *, review_rows: int, seed: int
) -> list[dict[str, str]]:
    families = sorted({row["task"] for row in rows})
    if review_rows < len(families) or review_rows % len(families):
        raise ValueError("review_rows must be a positive multiple of family count")
    quota = review_rows // len(families)
    grouped: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        answer = json.loads(row["answer_json"])
        grouped[row["task"]][(answer["risk_level"], row["split"])].append(row)

    selected: list[dict[str, str]] = []
    for family in families:
        strata = grouped[family]
        ordered_strata = sorted(strata)
        for values in strata.values():
            values.sort(
                key=lambda row: hashlib.sha256(
                    f"review:{seed}:{row['example_id']}".encode()
                ).digest()
            )
        positions = Counter()
        family_rows: list[dict[str, Any]] = []
        while len(family_rows) < quota:
            made_progress = False
            for stratum in ordered_strata:
                position = positions[stratum]
                if position < len(strata[stratum]):
                    family_rows.append(strata[stratum][position])
                    positions[stratum] += 1
                    made_progress = True
                    if len(family_rows) == quota:
                        break
            if not made_progress:
                raise ValueError(f"insufficient review candidates for {family}")
        for row in family_rows:
            answer = json.loads(row["answer_json"])
            selected.append(
                {
                    "example_id": row["example_id"],
                    "scenario_id": answer["scenario_id"],
                    "family": row["task"],
                    "domain": row["domain"],
                    "risk_level": answer["risk_level"],
                    "split": row["split"],
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "review_status": "pending",
                    "semantic_accuracy": "",
                    "constraint_following": "",
                    "language_quality": "",
                    "individualization": "",
                    "safety": "",
                    "reviewer_notes": "",
                }
            )
    return selected


def build_post_training_corpus(
    scenarios_path: Path,
    output_root: Path,
    *,
    variants_per_scenario: int = 2,
    review_rows: int = 70,
    seed: int = 42,
) -> dict[str, Any]:
    if variants_per_scenario < 1:
        raise ValueError("variants_per_scenario must be positive")
    scenarios = pq.read_table(scenarios_path).to_pylist()
    rows = _conversation_rows(scenarios, variants_per_scenario)
    audit = _audit(rows)
    review = _review_sample(rows, review_rows=review_rows, seed=seed)

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    conversations_path = temporary / "conversations.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=INSTRUCTION_SCHEMA),
        conversations_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    audit_path = temporary / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    review_path = temporary / "human_review.csv"
    with review_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    manifest = {
        "format": "complexity-post-training-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "license": DATASET_LICENSE,
        "source": DATASET_SOURCE,
        "input": {"path": str(scenarios_path), "sha256": file_sha256(scenarios_path)},
        "variants_per_scenario": variants_per_scenario,
        "audit": audit,
        "human_review": {
            "rows": len(review),
            "status": "pending",
            "strata": ["family", "risk_level", "split"],
            "required_before_training": True,
        },
        "training_ready": False,
        "release_ready": False,
    }
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return {
        **manifest,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in (
                output_root / "conversations.parquet",
                output_root / "audit.json",
                output_root / "human_review.csv",
                output_root / "manifest.json",
            )
        },
    }


def audit_human_review(review_path: Path) -> dict[str, Any]:
    """Evaluate a completed stratified review sheet without changing artifacts."""
    with review_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("human review sheet is empty")
    required = {
        "example_id",
        "scenario_id",
        "family",
        "risk_level",
        "split",
        "review_status",
        *REVIEW_GRADES,
        "reviewer_notes",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"human review sheet is missing columns: {sorted(missing)}")
    status_counts = Counter(row["review_status"].strip().lower() for row in rows)
    grade_counts = {
        grade: dict(
            sorted(Counter(row[grade].strip().lower() for row in rows).items())
        )
        for grade in REVIEW_GRADES
    }
    approved = all(row["review_status"].strip().lower() == "approved" for row in rows)
    grades_pass = all(
        row[grade].strip().lower() == "pass"
        for row in rows
        for grade in REVIEW_GRADES
    )
    return {
        "rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "grade_counts": grade_counts,
        "families": sorted({row["family"] for row in rows}),
        "risk_levels": sorted({row["risk_level"] for row in rows}),
        "splits": sorted({row["split"] for row in rows}),
        "training_ready": approved and grades_pass,
    }
