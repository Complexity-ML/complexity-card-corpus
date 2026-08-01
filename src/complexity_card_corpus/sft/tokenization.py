from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..release_targets import (
    TARGET_POST_TRAINING_ROWS,
    TARGET_POST_TRAINING_SUPERVISED_TOKENS,
)
from ..build import file_sha256
from ..chat_template import (
    CHAT_TEMPLATE_ID,
    chat_template_contract,
    render_system_prefix,
    render_user_turn,
)
from ..tokenize import directory_sha256, load_encoding
from ..training_cards import TrainingCards
from .evaluation import _audit_sft_projection, load_heldout_evaluation
from .language import _card_sections, _render_messages
from .projection import (
    _project_sft_conversation,
)
from .selection import (
    _balance_task_families,
    _deduplicate_exact_prompts,
    _deduplicate_exact_responses,
    _deduplicate_structural_rows,
)
from .schema import IGNORE_INDEX, LABEL_DTYPE, PROJECTED_SFT_SCHEMA, TOKEN_DTYPE


TRAIN_MAX_EXAMPLES_PER_FAMILY = 15_000
TRAIN_MAX_PER_STRUCTURE = 8
TRAIN_EXTRACTION_MAX_PER_STRUCTURE = 32

SUPPLEMENT_TASK_ALIASES = {
    "empathetic_dialogue": "conversation_empathy",
    "practical_dialogue": "practical_action",
}

_PRACTICAL_STAGE_TASKS = {
    "ask_for_missing_detail": "context_clarification",
    "present_bounded_options": "planning_comparison",
}


def _canonical_supplement_task(row: dict[str, Any]) -> str:
    """Map a conversation surface by its realized behavior, not its source bucket."""

    original_task = row["task"]
    try:
        stages = json.loads(row["answer_json"]).get("dialogue_stages", [])
    except (json.JSONDecodeError, TypeError):
        stages = []
    if original_task == "empathetic_dialogue":
        if stages and stages[-1] == "invite_detail_without_assumption":
            return "context_clarification"
        return "conversation_empathy"
    if original_task != "practical_dialogue":
        return SUPPLEMENT_TASK_ALIASES.get(original_task, original_task)
    if stages:
        return _PRACTICAL_STAGE_TASKS.get(stages[-1], "practical_action")
    return "practical_action"


def _load_instruction_sources(
    instructions_path: Path,
    supplementary_instruction_paths: list[Path] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Load one canonical corpus plus optional original conversation surfaces."""

    paths = [instructions_path, *(supplementary_instruction_paths or [])]
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        digest = file_sha256(path)
        source_rows = pq.read_table(path).to_pylist()
        aliases: Counter[str] = Counter()
        if index:
            for row in source_rows:
                original_task = row["task"]
                row["task"] = _canonical_supplement_task(row)
                if row["task"] != original_task:
                    aliases[f"{original_task}->{row['task']}"] += 1
        rows.extend(source_rows)
        sources.append(
            {
                "path": str(path),
                "sha256": digest,
                "examples": len(source_rows),
                "task_aliases": dict(sorted(aliases.items())),
            }
        )
    example_counts = Counter(row["example_id"] for row in rows)
    duplicate_ids = [key for key, count in example_counts.items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"instruction sources contain duplicate id: {duplicate_ids[0]}")
    material = "\n".join(source["sha256"] for source in sources)
    combined_sha256 = hashlib.sha256(material.encode()).hexdigest()
    return sorted(rows, key=lambda row: row["example_id"]), sources, combined_sha256


def _encode_messages(
    messages: list[dict[str, str]],
    example_id: str,
    task: str,
    answer_json: str,
    encoding,
    eos_id: int,
    chat_template: dict[str, Any],
    projection: tuple[list[dict[str, str]], TrainingCards] | None = None,
) -> tuple[list[int], list[int], TrainingCards]:
    """Serialize a naturalized conversation with assistant-only supervision."""

    if projection is None:
        projection = _project_sft_conversation(
            messages,
            example_id=example_id,
            task=task,
            answer_json=answer_json,
        )
    projected_messages, cards = projection
    full_ids: list[int] = []
    target_labels: list[int] = []
    system_tokens = encoding.encode(
        render_system_prefix(chat_template),
        disallowed_special=(),
    )
    full_ids.extend(system_tokens)
    target_labels.extend([IGNORE_INDEX] * len(system_tokens))
    for message in projected_messages:
        if message["role"] == "user":
            tokens = encoding.encode(
                render_user_turn(message["content"], chat_template),
                disallowed_special=(),
            )
            full_ids.extend(tokens)
            target_labels.extend([IGNORE_INDEX] * len(tokens))
            continue
        if message["role"] != "assistant":
            raise ValueError(f"unsupported SFT role: {message['role']}")
        prefix = encoding.encode(
            chat_template["assistant_prefix"],
            disallowed_special=(),
        )
        response = encoding.encode(message["content"], disallowed_special=())
        full_ids.extend(prefix)
        target_labels.extend([IGNORE_INDEX] * len(prefix))
        full_ids.extend(response)
        target_labels.extend(response)
        full_ids.append(eos_id)
        target_labels.append(eos_id)
    # Causal alignment: logits at position t predict token t+1. Supervision is
    # active only when that next token belongs to an assistant response.
    return full_ids[:-1], target_labels[1:], cards


def tokenize_instruction_dataset(
    instructions_path: Path,
    tokenizer_root: Path,
    output_root: Path,
    heldout_evaluation_path: Path | None = None,
    supplementary_instruction_paths: list[Path] | None = None,
) -> dict[str, Any]:
    encoding, tokenizer_config = load_encoding(tokenizer_root)
    eos_token = tokenizer_config.get("eos_token", "<|endoftext|>")
    try:
        eos_id = encoding.encode_single_token(eos_token)
    except KeyError:
        eos_id = encoding.eot_token
    chat_template = chat_template_contract()
    chat_template["eos_token"] = eos_token
    source_rows, instruction_sources, instruction_sources_sha256 = (
        _load_instruction_sources(
            instructions_path,
            supplementary_instruction_paths,
        )
    )
    evaluation_sha256: str | None = None
    evaluation_provenance: dict[str, int] = {}
    if heldout_evaluation_path is not None:
        source_rows = [row for row in source_rows if row["split"] == "train"]
        heldout_rows = load_heldout_evaluation(heldout_evaluation_path)
        evaluation_provenance = dict(
            sorted(
                Counter(
                    json.loads(row["answer_json"])["evaluation_source"]
                    for row in heldout_rows
                ).items()
            )
        )
        for row in heldout_rows:
            provenance = json.loads(row["answer_json"])["evaluation_source"]
            if provenance != "separately_authored":
                row["split"] = "diagnostic"
        source_rows.extend(heldout_rows)
        evaluation_sha256 = file_sha256(heldout_evaluation_path)

    projected_rows: list[dict[str, Any]] = []
    for row in source_rows:
        projected_messages, cards = _project_sft_conversation(
            row["messages"],
            example_id=row["example_id"],
            task=row["task"],
            answer_json=row["answer_json"],
        )
        prompt = _render_messages(projected_messages[:-1])
        target = projected_messages[-1]["content"]
        projected_rows.append(
            {
                **row,
                "_projected_messages": projected_messages,
                "_projected_prompt": prompt,
                "_projected_target": target,
                "_conditioning_cards": cards,
            }
        )
    projection_audit = _audit_sft_projection(projected_rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in projected_rows:
        partition = {
            "train": "train",
            "validation": "eval",
            "test": "test",
            "diagnostic": "diagnostic",
        }[row["split"]]
        grouped[partition].append(row)

    exact_deduplication: dict[str, Any] = {}
    exact_prompt_deduplication: dict[str, Any] = {}
    family_balance: dict[str, Any] = {}
    deduplication: dict[str, Any] = {}
    for partition, partition_rows in list(grouped.items()):
        partition_rows, exact_deduplication[partition] = _deduplicate_exact_responses(
            partition_rows
        )
        partition_rows, exact_prompt_deduplication[partition] = (
            _deduplicate_exact_prompts(partition_rows)
        )
        structure_limit = TRAIN_MAX_PER_STRUCTURE if partition == "train" else 10_000
        per_task_limits = (
            {"extraction_classification": TRAIN_EXTRACTION_MAX_PER_STRUCTURE}
            if partition == "train"
            else None
        )
        partition_rows, deduplication[partition] = _deduplicate_structural_rows(
            partition_rows,
            max_per_structure=structure_limit,
            per_task_limits=per_task_limits,
        )
        if partition == "train":
            partition_rows, family_balance[partition] = _balance_task_families(
                partition_rows,
                max_examples_per_family=TRAIN_MAX_EXAMPLES_PER_FAMILY,
            )
        else:
            counts = dict(
                sorted(Counter(row["task"] for row in partition_rows).items())
            )
            family_balance[partition] = {
                "input_examples": len(partition_rows),
                "kept_examples": len(partition_rows),
                "dropped_for_family_balance": 0,
                "maximum_examples_per_family": None,
                "before": counts,
                "after": counts,
            }
        grouped[partition] = partition_rows

    train_structures = {
        (row["task"], row["_structure_signature"]) for row in grouped.get("train", [])
    }
    eval_structures = {
        (row["task"], row["_structure_signature"]) for row in grouped.get("eval", [])
    }
    overlap = train_structures & eval_structures
    if heldout_evaluation_path is not None and overlap:
        sample = next(iter(overlap))
        raise ValueError(
            "held-out evaluation shares a normalized answer structure with training: "
            f"{sample}"
        )

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    chat_template_path = temporary / "chat_template.json"
    chat_template_path.write_text(
        json.dumps(chat_template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    projected_records = [
        {
            "example_id": row["example_id"],
            "task": row["task"],
            "mode": row["mode"],
            "difficulty": row["difficulty"],
            "domain": row["domain"],
            "language": row["language"],
            "split": "validation" if partition == "eval" else partition,
            "messages": row["_projected_messages"],
            "prompt": row["_projected_prompt"],
            "response": row["_projected_target"],
            "structure_signature": row["_structure_signature"],
            "source_representation": (
                "card_hand"
                if _card_sections(row["messages"]) is not None
                else "conversation"
            ),
            "source": row["source"],
            "license": row["license"],
            "version": row["version"],
        }
        for partition, partition_rows in sorted(grouped.items())
        for row in partition_rows
    ]
    projected_path = temporary / "projected.parquet"
    pq.write_table(
        pa.Table.from_pylist(projected_records, schema=PROJECTED_SFT_SCHEMA),
        projected_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )

    manifests: dict[str, Any] = {}
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
            inputs_path.open("wb") as inputs_handle,
            labels_path.open("wb") as labels_handle,
            examples_path.open("w", encoding="utf-8") as examples_handle,
        ):
            for row in partition_rows:
                has_card_hand = _card_sections(row["messages"]) is not None
                input_ids, labels, conditioning_cards = _encode_messages(
                    row["messages"],
                    row["example_id"],
                    row["task"],
                    row["answer_json"],
                    encoding,
                    eos_id,
                    chat_template,
                    projection=(
                        row["_projected_messages"],
                        row["_conditioning_cards"],
                    ),
                )
                np.asarray(input_ids, dtype=TOKEN_DTYPE).tofile(inputs_handle)
                np.asarray(labels, dtype=LABEL_DTYPE).tofile(labels_handle)
                for card_name, value in conditioning_cards.as_dict().items():
                    conditioning_counts[card_name][value] += 1
                supervised = sum(label != IGNORE_INDEX for label in labels)
                examples_handle.write(
                    json.dumps(
                        {
                            "example_id": row["example_id"],
                            "hand_id": row["example_id"],
                            "source_representation": (
                                "card_hand" if has_card_hand else "conversation"
                            ),
                            "training_representation": (
                                "natural_multi_turn"
                                if len(row["_projected_messages"]) > 2
                                else "natural_instruction"
                            ),
                            "conditioning_cards": conditioning_cards.as_dict(),
                            "cards": (
                                ["situation", "data", "rule", "goal"]
                                if has_card_hand
                                else []
                            ),
                            "task": row["task"],
                            "structure_signature": row["_structure_signature"],
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
            "chat_template_sha256": file_sha256(chat_template_path),
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
            "source_sha256": instruction_sources_sha256,
            "evaluation_source_sha256": evaluation_sha256,
            "input_ids_sha256": file_sha256(inputs_path),
            "labels_sha256": file_sha256(labels_path),
            "examples_sha256": file_sha256(examples_path),
            "conditioning_card_counts": {
                name: dict(sorted(counts.items()))
                for name, counts in sorted(conditioning_counts.items())
            },
        }
        (root / "sft.idx.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        manifests[partition] = metadata
    manifest = {
        "format": "complexity-atlas-instruct-tokenized-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": tokenizer_config["encoding_name"],
        "chat_template_id": CHAT_TEMPLATE_ID,
        "chat_template_sha256": file_sha256(chat_template_path),
        "serialization": (
            "System followed by alternating natural User and Assistant turns; "
            "every assistant content span and eos token is supervised"
        ),
        "training_projection": chat_template["training_projection"],
        "instruction_sources": instruction_sources,
        "instruction_sources_sha256": instruction_sources_sha256,
        "projection_audit": projection_audit,
        "exact_response_deduplication": exact_deduplication,
        "exact_prompt_deduplication": exact_prompt_deduplication,
        "family_balance": family_balance,
        "structural_deduplication": deduplication,
        "train_eval_structure_overlap": len(overlap),
        "heldout_evaluation": (
            {
                "path": str(heldout_evaluation_path),
                "sha256": evaluation_sha256,
                "method": (
                    "separately_authored_gold_with_diagnostic_companion"
                    if "source_separated_diagnostic" in evaluation_provenance
                    else next(iter(evaluation_provenance))
                ),
                "provenance_counts": evaluation_provenance,
            }
            if heldout_evaluation_path is not None
            else None
        ),
        "projected_parquet": {
            "path": projected_path.name,
            "examples": len(projected_records),
            "bytes": projected_path.stat().st_size,
            "sha256": file_sha256(projected_path),
            "splits": dict(
                sorted(Counter(row["split"] for row in projected_records).items())
            ),
        },
        "partitions": manifests,
        "total_examples": sum(item["examples"] for item in manifests.values()),
        "total_tokens": sum(item["num_tokens"] for item in manifests.values()),
        "total_supervised_tokens": sum(
            item["supervised_tokens"] for item in manifests.values()
        ),
    }
    train_records = [row for row in projected_records if row["split"] == "train"]
    train_family_counts = dict(
        sorted(Counter(row["task"] for row in train_records).items())
    )
    train_count = len(train_records)
    family_shares = {
        task: round(count / train_count, 6) if train_count else 0.0
        for task, count in train_family_counts.items()
    }
    exact_train_responses = len({row["response"] for row in train_records})
    exact_train_prompts = len({row["prompt"] for row in train_records})
    multi_turn_count = sum(len(row["messages"]) > 2 for row in train_records)
    genuine_multi_turn_count = sum(
        len(row["messages"]) > 2 and row["source_representation"] == "conversation"
        for row in train_records
    )
    synthetic_multi_turn_count = multi_turn_count - genuine_multi_turn_count
    difficulty_counts = dict(
        sorted(Counter(row["difficulty"] for row in train_records).items())
    )
    response_length_bands = Counter()
    for row in train_records:
        words = len(row["response"].split())
        if words <= 25:
            response_length_bands["direct_1_25"] += 1
        elif words <= 45:
            response_length_bands["short_26_45"] += 1
        elif words <= 80:
            response_length_bands["standard_46_80"] += 1
        else:
            response_length_bands["extended_81_plus"] += 1
    response_length_bands = Counter(
        {
            name: response_length_bands[name]
            for name in (
                "direct_1_25",
                "short_26_45",
                "standard_46_80",
                "extended_81_plus",
            )
        }
    )
    distinct_train_structures = len(
        {(row["task"], row["structure_signature"]) for row in train_records}
    )
    quality_checks = {
        "no_exact_duplicate_train_responses": exact_train_responses == train_count,
        "no_exact_duplicate_train_prompts": exact_train_prompts == train_count,
        "at_least_fourteen_training_families": len(train_family_counts) >= 14,
        "training_examples_at_least_100k": (
            train_count >= TARGET_POST_TRAINING_ROWS
        ),
        "maximum_family_share_at_most_15_percent": max(
            family_shares.values(), default=0.0
        )
        <= 0.15,
        "minimum_family_share_at_least_2_percent": min(
            family_shares.values(), default=0.0
        )
        >= 0.02,
        "has_easy_medium_and_hard_examples": set(difficulty_counts)
        == {"easy", "medium", "hard"},
        "easy_examples_are_at_least_20_percent": (
            difficulty_counts.get("easy", 0) / train_count if train_count else 0.0
        )
        >= 0.20,
        "genuine_multi_turn_share_at_least_10_percent": (
            genuine_multi_turn_count / train_count if train_count else 0.0
        )
        >= 0.10,
        "synthetic_multi_turn_share_at_most_5_percent": (
            synthetic_multi_turn_count / train_count if train_count else 1.0
        )
        <= 0.05,
        "four_response_length_bands_each_at_least_5_percent": all(
            count / train_count >= 0.05 if train_count else False
            for count in response_length_bands.values()
        ),
        "distinct_structure_share_at_least_20_percent": (
            distinct_train_structures / train_count if train_count else 0.0
        )
        >= 0.20,
        "heldout_evaluation_has_at_least_28_authored_examples": (
            manifests.get("eval", {}).get("examples", 0) >= 28
        ),
        "diagnostic_companion_has_500_to_1000_examples": 500
        <= manifests.get("diagnostic", {}).get("examples", 0)
        <= 1_000,
        "training_supervised_tokens_at_least_10m": (
            manifests.get("train", {}).get("supervised_tokens", 0)
            >= TARGET_POST_TRAINING_SUPERVISED_TOKENS
        ),
    }
    manifest["release_quality"] = {
        "ready": all(quality_checks.values()),
        "checks": quality_checks,
        "train_family_counts": train_family_counts,
        "train_family_shares": family_shares,
        "difficulty_counts": difficulty_counts,
        "response_length_bands": dict(response_length_bands),
        "distinct_train_structures": distinct_train_structures,
        "distinct_train_structure_share": round(
            distinct_train_structures / train_count if train_count else 0.0, 6
        ),
        "multi_turn_examples": multi_turn_count,
        "multi_turn_share": round(
            multi_turn_count / train_count if train_count else 0.0, 6
        ),
        "genuine_multi_turn_examples": genuine_multi_turn_count,
        "genuine_multi_turn_share": round(
            genuine_multi_turn_count / train_count if train_count else 0.0, 6
        ),
        "synthetic_multi_turn_examples": synthetic_multi_turn_count,
        "synthetic_multi_turn_share": round(
            synthetic_multi_turn_count / train_count if train_count else 0.0, 6
        ),
        "exact_train_response_uniqueness_ratio": round(
            exact_train_responses / train_count if train_count else 0.0, 6
        ),
        "exact_train_prompt_uniqueness_ratio": round(
            exact_train_prompts / train_count if train_count else 0.0, 6
        ),
        "minimum_supervised_training_tokens": (
            TARGET_POST_TRAINING_SUPERVISED_TOKENS
        ),
        "target_training_examples": TARGET_POST_TRAINING_ROWS,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return manifest
