from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .behavior_audit import audit_v2_behavior
from .chat import (
    CHAT_TEMPLATE_ID,
    chat_template_contract,
    render_history_prefix,
    validate_training_messages,
)
from .composition_audit import audit_v2_composition
from .distribution_audit import audit_v2_distribution
from .gates import v2_gate_progress
from .integrity_audit import audit_v2_integrity
from .length_audit import audit_v2_lengths
from .near_duplicate_audit import audit_v2_near_duplicates
from .registry import render_complete_v2, v2_generation_progress
from .roadmap import audit_v2_family_roadmap, roadmap_markdown
from .split_audit import audit_v2_splits
from .tokenization_audit import audit_v2_tokenization
from .tokenizer import IGNORE_INDEX, directory_sha256, file_sha256, load_encoding


TOKEN_DTYPE = np.dtype("<u4")
LABEL_DTYPE = np.dtype("<i4")
_PARTITIONS = {"train": "train", "validation": "eval", "test": "test"}


def _require_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"output already exists: {path}; choose a new path or remove it explicitly"
        )
    path.mkdir(parents=True)


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_v2_release(output_root: Path) -> dict[str, Any]:
    """Render and persist V2 without executing any statistical quality audit."""

    _require_new_directory(output_root)
    rows = render_complete_v2()
    projected_path = output_root / "projected.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows),
        projected_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
        row_group_size=16_384,
    )
    split_counts = dict(sorted(Counter(str(row["split"]) for row in rows).items()))
    task_counts = dict(sorted(Counter(str(row["task"]) for row in rows).items()))
    manifest = {
        "format": "complexity-card-corpus-v2-release-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase_status": "built",
        "quality_status": "not_run",
        "tests_executed_during_build": False,
        "statistical_audits_executed_during_build": False,
        "example_limit": None,
        "examples": len(rows),
        "splits": split_counts,
        "tasks": task_counts,
        "generation": v2_generation_progress(),
        "gate_inventory": v2_gate_progress(),
        "projected": {
            "path": projected_path.name,
            "bytes": projected_path.stat().st_size,
            "sha256": file_sha256(projected_path),
        },
    }
    _json(output_root / "manifest.json", manifest)
    return manifest


def audit_v2_release(
    artifact_root: Path,
    *,
    tokenizer_root: Path | None = None,
) -> dict[str, Any]:
    """Run the heavy release gates as an explicit phase after generation."""

    manifest_path = artifact_root / "manifest.json"
    projected_path = artifact_root / "projected.parquet"
    manifest = json.loads(manifest_path.read_text())
    if file_sha256(projected_path) != manifest["projected"]["sha256"]:
        raise ValueError("projected Parquet digest differs from the build manifest")
    rows = pq.read_table(projected_path).to_pylist()
    audits = {
        "behavior": audit_v2_behavior(rows),
        "integrity": audit_v2_integrity(rows),
        "distribution": audit_v2_distribution(rows),
        "composition": audit_v2_composition(rows),
        "near_duplicates": audit_v2_near_duplicates(rows),
        "lengths": audit_v2_lengths(rows),
        "splits": audit_v2_splits(rows),
    }
    if tokenizer_root is not None:
        audits["tokenization"] = audit_v2_tokenization(rows, tokenizer_root)
    roadmap = audit_v2_family_roadmap(
        rows,
        tokenizer_root=tokenizer_root,
        require_splits=True,
    )
    family_passed = (
        roadmap["complete_gate_contract"]
        and roadmap["split_audit"]["passed"]
        and all(
            family["priority"] == "PASS"
            for family in roadmap["families"].values()
        )
    )
    passed = (
        all(bool(audit["passed"]) for audit in audits.values())
        and family_passed
    )
    report = {
        "format": "complexity-card-corpus-v2-release-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "artifact_sha256": manifest["projected"]["sha256"],
        "audits": audits,
        "family_roadmap": {
            "passed": family_passed,
            "families": len(roadmap["families"]),
            "priority_counts": roadmap["priority_counts"],
            "rows": roadmap["rows"],
            "train_rows": roadmap["train_rows"],
        },
    }
    _json(artifact_root / "audit.json", report)
    _json(artifact_root / "roadmap.json", roadmap)
    (artifact_root / "roadmap.md").write_text(roadmap_markdown(roadmap) + "\n")
    manifest["phase_status"] = "audited"
    manifest["quality_status"] = "passed" if passed else "failed"
    manifest["audit"] = {
        "path": "audit.json",
        "sha256": file_sha256(artifact_root / "audit.json"),
        "tokenizer_checked": tokenizer_root is not None,
    }
    _json(manifest_path, manifest)
    return report


def tokenize_v2_release(
    artifact_root: Path,
    tokenizer_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Create framework-compatible assistant-only SFT shards after a green audit."""

    manifest = json.loads((artifact_root / "manifest.json").read_text())
    if manifest.get("quality_status") != "passed":
        raise ValueError("V2 tokenization requires a separately completed green audit")
    _require_new_directory(output_root)
    rows = pq.read_table(artifact_root / "projected.parquet").to_pylist()
    encoding, tokenizer_config = load_encoding(tokenizer_root)
    contract = chat_template_contract()
    eos_token = str(tokenizer_config.get("eos_token", contract["eos_token"]))
    try:
        eos_id = encoding.encode_single_token(eos_token)
    except KeyError:
        eos_id = encoding.eot_token
    _json(output_root / "chat_template.json", contract)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_PARTITIONS[str(row["split"])], []).append(row)
    partition_manifests = {}
    for partition, partition_rows in sorted(grouped.items()):
        root = output_root / partition
        root.mkdir()
        input_path = root / "input_ids.bin"
        label_path = root / "labels.bin"
        examples_path = root / "examples.jsonl"
        offset = 0
        supervised_tokens = 0
        with (
            input_path.open("wb") as input_handle,
            label_path.open("wb") as label_handle,
            examples_path.open("w", encoding="utf-8") as examples_handle,
        ):
            for row in partition_rows:
                messages = list(row["messages"])
                validate_training_messages(messages)
                prefix = render_history_prefix(messages[:-1], contract)
                prefix_ids = encoding.encode(prefix, disallowed_special=())
                response_ids = encoding.encode(
                    str(row["response"]), disallowed_special=()
                )
                full = [*prefix_ids, *response_ids, eos_id]
                targets = [*([IGNORE_INDEX] * len(prefix_ids)), *response_ids, eos_id]
                input_ids = full[:-1]
                labels = targets[1:]
                np.asarray(input_ids, dtype=TOKEN_DTYPE).tofile(input_handle)
                np.asarray(labels, dtype=LABEL_DTYPE).tofile(label_handle)
                supervised = sum(label != IGNORE_INDEX for label in labels)
                examples_handle.write(
                    json.dumps(
                        {
                            "example_id": row["example_id"],
                            "hand_id": row["example_id"],
                            "source_representation": "card_hand",
                            "training_representation": "natural_instruction",
                            "conditioning_cards": {},
                            "response_card_hand": "v2",
                            "reasoning_envelope": bool(row["reasoning_envelope"]),
                            "reasoning_card_hand": "",
                            "cards": ["prompt", "answer"],
                            "task": row["task"],
                            "structure_signature": row["example_id"],
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
        index = {
            "format": "complexity-sft-token-shard-v2",
            "assistant_supervision": "final_assistant_only",
            "history_assistant_turns": "masked_context",
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
            "source_sha256": manifest["projected"]["sha256"],
            "input_ids_sha256": file_sha256(input_path),
            "labels_sha256": file_sha256(label_path),
            "examples_sha256": file_sha256(examples_path),
        }
        _json(root / "sft.idx.json", index)
        partition_manifests[partition] = index
    token_manifest = {
        "format": "complexity-card-corpus-v2-tokenized-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quality_status": "passed",
        "release_quality": {
            "ready": True,
            "assistant_only_loss": True,
            "reasoning_envelope_version": "card-corpus-v2-think-final-v1",
            "source_audit_sha256": manifest["audit"]["sha256"],
        },
        "source": {
            "format": manifest["format"],
            "examples": manifest["examples"],
            "projected_sha256": manifest["projected"]["sha256"],
        },
        "partitions": partition_manifests,
        "total_examples": sum(item["examples"] for item in partition_manifests.values()),
        "total_tokens": sum(item["num_tokens"] for item in partition_manifests.values()),
        "total_supervised_tokens": sum(
            item["supervised_tokens"] for item in partition_manifests.values()
        ),
    }
    _json(output_root / "manifest.json", token_manifest)
    return token_manifest


__all__ = ("audit_v2_release", "build_v2_release", "tokenize_v2_release")
