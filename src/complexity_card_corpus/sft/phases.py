from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ..build import file_sha256
from ..chat_template import CHAT_TEMPLATE_ID, chat_template_contract
from ..conversation_quality import audit_casual_conversation_quality
from ..quality_audit import audit_rows_quality
from ..tokenize import directory_sha256, load_encoding
from ..training_cards import RESPONSE_STRUCTURE_SIBLING_TASKS, TrainingCards
from .evaluation import _audit_sft_projection, audit_sft_repetition_quality
from .reasoning_envelope import (
    REASONING_ENVELOPE_VERSION,
    audit_reasoning_envelopes,
)
from .schema import IGNORE_INDEX, LABEL_DTYPE, TOKEN_DTYPE
from .tokenization import (
    TRAIN_QUALITY_MAX_MULTISCALE_REPETITION_SHARE,
    TRAIN_QUALITY_MAX_RESPONSE_CARD_HAND_SHARE,
    TRAIN_QUALITY_MIN_AUTHORED_CONVERSATION_SHARE,
    TRAIN_QUALITY_REASONING_ENVELOPE_SHARE,
    TRAIN_QUALITY_REPETITION_SAMPLE_PER_FAMILY,
    _audit_model_facing_style,
    _encode_messages,
    tokenize_instruction_dataset,
)


def project_instruction_dataset(
    instructions_path: Path,
    output_root: Path,
    *,
    heldout_evaluation_path: Path | None = None,
    supplementary_instruction_paths: list[Path] | None = None,
    casual_registry_path: Path | None = None,
    workers: int = 1,
    max_examples_per_family: int | None = None,
    max_per_structure: int | None = None,
    max_domain_share: float | None = None,
    max_response_card_hand_share: float | None = None,
    target_training_examples: int | None = None,
    target_supervised_tokens: int | None = None,
    require_casual_conversation: bool = True,
    reasoning_envelope_version: str | None = None,
) -> dict[str, Any]:
    """Run only the deterministic model-facing projection phase."""

    return tokenize_instruction_dataset(
        instructions_path,
        None,
        output_root,
        heldout_evaluation_path=heldout_evaluation_path,
        supplementary_instruction_paths=supplementary_instruction_paths,
        casual_registry_path=casual_registry_path,
        workers=workers,
        max_examples_per_family=max_examples_per_family,
        max_per_structure=max_per_structure,
        max_domain_share=max_domain_share,
        max_response_card_hand_share=max_response_card_hand_share,
        target_training_examples=target_training_examples,
        target_supervised_tokens=target_supervised_tokens,
        require_casual_conversation=require_casual_conversation,
        reasoning_envelope_version=reasoning_envelope_version,
        projection_only=True,
    )


def _load_projection(
    artifact_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = artifact_root / "projection-manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing projection manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    projected_path = artifact_root / manifest["projected_parquet"]["path"]
    if file_sha256(projected_path) != manifest["projected_parquet"]["sha256"]:
        raise ValueError("projected Parquet no longer matches projection manifest")
    return manifest, pq.read_table(projected_path).to_pylist()


def _cards_from_record(record: dict[str, Any]) -> TrainingCards:
    payload = record.get("conditioning_cards_json")
    if payload:
        return TrainingCards(**json.loads(payload))
    axes = record["response_card_hand"].split("|")
    if len(axes) != 4:
        raise ValueError(
            f"invalid response-card hand signature: {record['response_card_hand']!r}"
        )
    return TrainingCards(
        surface="projected",
        dialogue_state="projected",
        output="projected",
        evidence="projected",
        reasoning="projected",
        style="projected",
        context_density="projected",
        noise="projected",
        uncertainty="projected",
        response_order=axes[0],
        response_bridge=axes[1],
        response_layout=axes[2],
        response_opening=axes[3],
    )


def _audit_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **record,
            "_projected_messages": record["messages"],
            "_projected_prompt": record["prompt"],
            "_projected_target": record["response"],
            "_structure_signature": record["structure_signature"],
            "_conditioning_cards": _cards_from_record(record),
        }
        for record in records
    ]


def _release_quality(
    records: list[dict[str, Any]],
    projection: dict[str, Any],
    *,
    statistical_quality: dict[str, Any],
    casual_quality: dict[str, Any],
    style_quality: dict[str, Any],
    repetition_quality: dict[str, Any],
    reasoning_quality: dict[str, Any],
    train_eval_structure_overlap: int,
) -> dict[str, Any]:
    train = [row for row in records if row["split"] == "train"]
    train_count = len(train)
    family_counts = dict(sorted(Counter(row["task"] for row in train).items()))
    family_shares = {
        task: count / train_count if train_count else 0.0
        for task, count in family_counts.items()
    }
    core_counts = {
        task: count
        for task, count in family_counts.items()
        if task != "casual_conversation"
    }
    core_total = sum(core_counts.values())
    core_shares = {
        task: count / core_total if core_total else 0.0
        for task, count in core_counts.items()
    }
    difficulty_counts = dict(
        sorted(Counter(row["difficulty"] for row in train).items())
    )
    length_bands = Counter()
    for row in train:
        words = len(row["response"].split())
        if words <= 25:
            length_bands["direct_1_25"] += 1
        elif words <= 45:
            length_bands["short_26_45"] += 1
        elif words <= 80:
            length_bands["standard_46_80"] += 1
        else:
            length_bands["extended_81_plus"] += 1
    length_bands = Counter(
        {
            name: length_bands[name]
            for name in (
                "direct_1_25",
                "short_26_45",
                "standard_46_80",
                "extended_81_plus",
            )
        }
    )
    multi_turn = sum(len(row["messages"]) > 2 for row in train)
    authored = sum(row["source_representation"] == "conversation" for row in train)
    reasoning = sum(bool(row["reasoning_envelope"]) for row in train)
    distinct_structures = len(
        {(row["task"], row["structure_signature"]) for row in train}
    )
    hands: dict[str, Counter[str]] = defaultdict(Counter)
    siblings: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for row in train:
        task = row["task"]
        hand = row["response_card_hand"]
        hands[task][hand] += 1
        if task in RESPONSE_STRUCTURE_SIBLING_TASKS:
            cards = _cards_from_record(row)
            for (
                dimension,
                signature,
            ) in cards.response_structure_sibling_signatures.items():
                siblings[task][dimension][signature] += 1
    maximum_hand_share = max(
        (max(counts.values()) / sum(counts.values()) for counts in hands.values()),
        default=0.0,
    )
    maximum_sibling_share = max(
        (
            max(counts.values()) / sum(counts.values())
            for dimensions in siblings.values()
            for counts in dimensions.values()
        ),
        default=0.0,
    )
    eval_count = sum(row["split"] == "validation" for row in records)
    diagnostic_count = sum(row["split"] == "diagnostic" for row in records)
    checks = {
        "no_exact_duplicate_train_responses": len({row["response"] for row in train})
        == train_count,
        "no_exact_duplicate_train_prompts": len({row["prompt"] for row in train})
        == train_count,
        "at_least_fourteen_training_families": len(family_counts) >= 14,
        "maximum_family_share_at_most_15_percent": max(
            family_shares.values(), default=0.0
        )
        <= 0.15,
        "minimum_family_share_at_least_2_percent": min(
            core_shares.values(), default=0.0
        )
        >= 0.02,
        "has_easy_medium_and_hard_examples": set(difficulty_counts)
        == {"easy", "medium", "hard"},
        "easy_examples_are_at_least_20_percent": (
            difficulty_counts.get("easy", 0) / train_count if train_count else 0.0
        )
        >= 0.20,
        "multi_turn_share_between_10_and_30_percent": 0.10
        <= (multi_turn / train_count if train_count else 0.0)
        <= 0.30,
        "four_response_length_bands_each_at_least_5_percent": all(
            (count / train_count if train_count else 0.0) >= 0.05
            for count in length_bands.values()
        ),
        "distinct_structure_share_at_least_20_percent": (
            distinct_structures / train_count if train_count else 0.0
        )
        >= 0.20,
        "maximum_response_card_hand_share_at_most_5_percent": maximum_hand_share
        <= TRAIN_QUALITY_MAX_RESPONSE_CARD_HAND_SHARE,
        "maximum_response_card_sibling_share_at_most_5_percent": maximum_sibling_share
        <= TRAIN_QUALITY_MAX_RESPONSE_CARD_HAND_SHARE,
        "sklearn_statistical_quality_audit_passed": statistical_quality["passed"],
        "model_facing_style_repetition_audit_passed": style_quality["passed"],
        "all_fourteen_core_families_have_multiscale_repetition_metrics": len(
            core_counts
        )
        == 14
        and set(core_counts) <= set(repetition_quality["tasks"])
        and all(repetition_quality["tasks"][task]["audited"] for task in core_counts),
        "supervised_response_multiscale_repetition_audit_passed": repetition_quality[
            "supervised_passed"
        ],
        "authored_conversation_share_at_least_5_percent": (
            authored / train_count if train_count else 0.0
        )
        >= TRAIN_QUALITY_MIN_AUTHORED_CONVERSATION_SHARE,
        "no_train_eval_structure_overlap": train_eval_structure_overlap == 0,
    }
    if projection.get("heldout_evaluation") is not None:
        checks["heldout_evaluation_has_at_least_28_authored_examples"] = (
            eval_count >= 28
        )
        checks["diagnostic_companion_has_500_to_1000_examples"] = (
            500 <= diagnostic_count <= 1_000
        )
    if projection.get("require_casual_conversation", True):
        checks["casual_conversation_is_present"] = (
            family_counts.get("casual_conversation", 0) > 0
        )
        checks["casual_conversation_quality_passed"] = casual_quality["passed"]
    if projection.get("reasoning_envelope_version") == REASONING_ENVELOPE_VERSION:
        share = reasoning / train_count if train_count else 0.0
        checks["reasoning_envelope_v18_quality_passed"] = reasoning_quality["passed"]
        checks["reasoning_envelope_share_between_15_and_25_percent"] = (
            TRAIN_QUALITY_REASONING_ENVELOPE_SHARE[0]
            <= share
            <= TRAIN_QUALITY_REASONING_ENVELOPE_SHARE[1]
        )
    target_examples = projection.get("target_training_examples")
    if target_examples is not None:
        checks["training_examples_reach_requested_target"] = (
            train_count >= target_examples
        )
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "deferred_checks": (
            ["supervised_tokens_reach_requested_target"]
            if projection.get("target_supervised_tokens") is not None
            else []
        ),
        "train_family_counts": family_counts,
        "train_family_shares": {
            task: round(share, 6) for task, share in family_shares.items()
        },
        "core_train_family_shares": {
            task: round(share, 6) for task, share in core_shares.items()
        },
        "required_casual_conversation": projection.get(
            "require_casual_conversation", True
        ),
        "reasoning_envelope_version": projection.get("reasoning_envelope_version"),
        "difficulty_counts": difficulty_counts,
        "response_length_bands": dict(length_bands),
        "multi_turn_examples": multi_turn,
        "multi_turn_share": round(multi_turn / train_count if train_count else 0.0, 6),
        "authored_conversation_examples": authored,
        "authored_conversation_share": round(
            authored / train_count if train_count else 0.0, 6
        ),
        "reasoning_envelope_examples": reasoning,
        "reasoning_envelope_share": round(
            reasoning / train_count if train_count else 0.0, 6
        ),
        "distinct_train_structures": distinct_structures,
        "distinct_train_structure_share": round(
            distinct_structures / train_count if train_count else 0.0, 6
        ),
        "maximum_response_card_hand_share": round(maximum_hand_share, 6),
        "maximum_response_card_sibling_share": round(maximum_sibling_share, 6),
        "target_training_examples": target_examples,
        "target_supervised_training_tokens": projection.get("target_supervised_tokens"),
        "exact_train_response_uniqueness_ratio": round(
            len({row["response"] for row in train}) / train_count
            if train_count
            else 0.0,
            6,
        ),
        "exact_train_prompt_uniqueness_ratio": round(
            len({row["prompt"] for row in train}) / train_count if train_count else 0.0,
            6,
        ),
    }


def audit_projected_instruction_dataset(
    artifact_root: Path,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    """Run quality gates against an existing projection without rebuilding it."""

    projection, records = _load_projection(artifact_root)
    rows = _audit_rows(records)
    failures: list[dict[str, str]] = []
    try:
        projection_audit = _audit_sft_projection(rows)
    except ValueError as exc:
        projection_audit = {"passed": False, "error": str(exc)}
        failures.append({"audit": "projection", "error": str(exc)})
    reasoning_quality = audit_reasoning_envelopes(
        rows,
        enabled=projection.get("reasoning_envelope_version")
        == REASONING_ENVELOPE_VERSION,
    )
    if not reasoning_quality["passed"]:
        failures.append(
            {"audit": "reasoning_envelope", "error": "quality checks failed"}
        )
    statistical_quality = audit_rows_quality(
        rows,
        input_label="model-facing SFT projection",
        prompt_key="_projected_prompt",
        response_key="_projected_target",
        sample_size=None,
        near_duplicate_threshold=0.95,
        max_features=None,
        workers=workers,
    )
    casual_quality = audit_casual_conversation_quality(records)
    style_quality = _audit_model_facing_style(records)
    repetition_quality = audit_sft_repetition_quality(
        [row for row in rows if row["split"] == "train"],
        maximum_share=TRAIN_QUALITY_MAX_MULTISCALE_REPETITION_SHARE,
        maximum_examples_per_task=TRAIN_QUALITY_REPETITION_SAMPLE_PER_FAMILY,
    )
    train_structures = {
        (row["task"], row["structure_signature"])
        for row in records
        if row["split"] == "train"
    }
    eval_structures = {
        (row["task"], row["structure_signature"])
        for row in records
        if row["split"] == "validation"
    }
    release_quality = _release_quality(
        records,
        projection,
        statistical_quality=statistical_quality,
        casual_quality=casual_quality,
        style_quality=style_quality,
        repetition_quality=repetition_quality,
        reasoning_quality=reasoning_quality,
        train_eval_structure_overlap=len(train_structures & eval_structures),
    )
    if failures:
        release_quality["ready"] = False
    audit_manifest = {
        "format": "complexity-sft-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "audit",
        "phase_status": "passed" if not failures else "failed",
        "quality_status": "passed" if release_quality["ready"] else "failed",
        "projection_manifest_sha256": file_sha256(
            artifact_root / "projection-manifest.json"
        ),
        "projected_parquet_sha256": projection["projected_parquet"]["sha256"],
        "projection_audit": projection_audit,
        "reasoning_envelope_quality": reasoning_quality,
        "statistical_quality_audit": statistical_quality,
        "casual_conversation_quality": casual_quality,
        "model_facing_style_quality": style_quality,
        "model_facing_repetition_quality": repetition_quality,
        "train_eval_structure_overlap": len(train_structures & eval_structures),
        "failures": failures,
        "release_quality": release_quality,
    }
    (artifact_root / "audit-manifest.json").write_text(
        json.dumps(audit_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit_manifest


def tokenize_projected_instruction_dataset(
    artifact_root: Path,
    tokenizer_root: Path,
    *,
    require_audit_passed: bool = True,
) -> dict[str, Any]:
    """Tokenize an existing projection; this function deliberately runs no audit."""

    projection, records = _load_projection(artifact_root)
    audit_path = artifact_root / "audit-manifest.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("projected_parquet_sha256")
            != projection["projected_parquet"]["sha256"]
        ):
            raise ValueError("audit manifest belongs to a different projection")
    else:
        audit = {
            "quality_status": "not_run",
            "release_quality": {"ready": False, "checks": {}},
        }
    if require_audit_passed and audit["quality_status"] != "passed":
        raise ValueError(
            "projected SFT audit is not passed; run audit-projected-sft first "
            "or explicitly allow a failed audit for diagnostics"
        )

    encoding, tokenizer_config = load_encoding(tokenizer_root)
    chat_template = chat_template_contract()
    eos_token = tokenizer_config.get("eos_token", "<|endoftext|>")
    chat_template["eos_token"] = eos_token
    try:
        eos_id = encoding.encode_single_token(eos_token)
    except KeyError:
        eos_id = encoding.eot_token

    temporary = artifact_root / ".tokenization.partial"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    chat_template_path = temporary / "chat_template.json"
    chat_template_path.write_text(
        json.dumps(chat_template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        partition = "eval" if record["split"] == "validation" else record["split"]
        grouped[partition].append(record)

    partitions: dict[str, Any] = {}
    for partition, partition_rows in sorted(grouped.items()):
        root = temporary / partition
        root.mkdir()
        inputs_path = root / "input_ids.bin"
        labels_path = root / "labels.bin"
        examples_path = root / "examples.jsonl"
        offset = 0
        supervised_tokens = 0
        conditioning_counts: dict[str, Counter[str]] = defaultdict(Counter)
        with (
            inputs_path.open("wb") as inputs,
            labels_path.open("wb") as labels_handle,
            examples_path.open("w", encoding="utf-8") as examples,
        ):
            for row in partition_rows:
                cards = _cards_from_record(row)
                input_ids, labels, _ = _encode_messages(
                    [],
                    row["example_id"],
                    row["task"],
                    "{}",
                    encoding,
                    eos_id,
                    chat_template,
                    projection=(row["messages"], cards),
                )
                np.asarray(input_ids, dtype=TOKEN_DTYPE).tofile(inputs)
                np.asarray(labels, dtype=LABEL_DTYPE).tofile(labels_handle)
                for name, value in cards.as_dict().items():
                    conditioning_counts[name][value] += 1
                supervised = sum(label != IGNORE_INDEX for label in labels)
                examples.write(
                    json.dumps(
                        {
                            "example_id": row["example_id"],
                            "hand_id": row["example_id"],
                            "source_representation": row["source_representation"],
                            "training_representation": "natural_multi_turn"
                            if len(row["messages"]) > 2
                            else "natural_instruction",
                            "conditioning_cards": cards.as_dict(),
                            "response_card_hand": row["response_card_hand"],
                            "reasoning_envelope": row["reasoning_envelope"],
                            "reasoning_card_hand": row["reasoning_card_hand"],
                            "cards": ["situation", "data", "rule", "goal"]
                            if row["source_representation"] == "card_hand"
                            else [],
                            "task": row["task"],
                            "structure_signature": row["structure_signature"],
                            "offset": offset,
                            "num_tokens": len(input_ids),
                            "supervised_tokens": supervised,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                offset += len(input_ids)
                supervised_tokens += supervised
        metadata = {
            "format": "complexity-sft-token-shard-v1",
            "chat_template_id": CHAT_TEMPLATE_ID,
            "partition": partition,
            "examples": len(partition_rows),
            "num_tokens": offset,
            "supervised_tokens": supervised_tokens,
            "ignore_index": IGNORE_INDEX,
            "input_dtype": TOKEN_DTYPE.str,
            "label_dtype": LABEL_DTYPE.str,
            "vocab_size": encoding.n_vocab,
            "eos_token_id": eos_id,
            "tokenizer": tokenizer_config["encoding_name"],
            "tokenizer_sha256": directory_sha256(tokenizer_root),
            "source_sha256": projection["instruction_sources_sha256"],
            "input_ids_sha256": file_sha256(inputs_path),
            "labels_sha256": file_sha256(labels_path),
            "examples_sha256": file_sha256(examples_path),
            "conditioning_card_counts": {
                name: dict(sorted(counts.items()))
                for name, counts in sorted(conditioning_counts.items())
            },
        }
        (root / "sft.idx.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        partitions[partition] = metadata

    release_quality = dict(audit.get("release_quality", {}))
    checks = dict(release_quality.get("checks", {}))
    target_tokens = projection.get("target_supervised_tokens")
    if target_tokens is not None:
        checks["supervised_tokens_reach_requested_target"] = (
            partitions.get("train", {}).get("supervised_tokens", 0) >= target_tokens
        )
    release_quality["checks"] = checks
    release_quality["deferred_checks"] = []
    release_quality["ready"] = audit.get("quality_status") == "passed" and all(
        checks.values()
    )
    if audit.get("quality_status") == "not_run":
        quality_status = "not_run"
    else:
        quality_status = "passed" if release_quality["ready"] else "failed"
    manifest = {
        "format": "complexity-atlas-instruct-tokenized-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "tokenization",
        "phase_status": "passed",
        "quality_status": quality_status,
        "tokenizer": tokenizer_config["encoding_name"],
        "tokenizer_sha256": directory_sha256(tokenizer_root),
        "chat_template_id": CHAT_TEMPLATE_ID,
        "chat_template_sha256": file_sha256(chat_template_path),
        "instruction_sources": projection["instruction_sources"],
        "instruction_sources_sha256": projection["instruction_sources_sha256"],
        "surface_selection": projection["surface_selection"],
        "heldout_evaluation": projection["heldout_evaluation"],
        "projection_manifest_sha256": file_sha256(
            artifact_root / "projection-manifest.json"
        ),
        "audit_manifest_sha256": file_sha256(audit_path)
        if audit_path.exists()
        else None,
        "projected_parquet": projection["projected_parquet"],
        "projection_audit": audit.get("projection_audit"),
        "reasoning_envelope_quality": audit.get("reasoning_envelope_quality"),
        "statistical_quality_audit": audit.get("statistical_quality_audit"),
        "casual_conversation_quality": audit.get("casual_conversation_quality"),
        "model_facing_style_quality": audit.get("model_facing_style_quality"),
        "model_facing_repetition_quality": audit.get("model_facing_repetition_quality"),
        "train_eval_structure_overlap": audit.get("train_eval_structure_overlap"),
        "partitions": partitions,
        "total_examples": sum(item["examples"] for item in partitions.values()),
        "total_tokens": sum(item["num_tokens"] for item in partitions.values()),
        "total_supervised_tokens": sum(
            item["supervised_tokens"] for item in partitions.values()
        ),
        "release_quality": release_quality,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name in [*partitions, "chat_template.json", "manifest.json"]:
        destination = artifact_root / name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        (temporary / name).replace(destination)
    temporary.rmdir()
    return manifest
