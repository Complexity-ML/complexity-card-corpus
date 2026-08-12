from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..quality_audit import audit_rows_quality
from ..build import file_sha256
from ..chat_template import (
    CHAT_TEMPLATE_ID,
    chat_template_contract,
    render_system_prefix,
    render_user_turn,
)
from ..conversation_quality import audit_casual_conversation_quality
from ..tokenize import directory_sha256, load_encoding
from ..training_cards import (
    RESPONSE_STRUCTURE_SIBLING_TASKS,
    TrainingCards,
    deal_training_cards,
    projected_difficulty,
)
from .evaluation import (
    _audit_sft_projection,
    audit_sft_repetition_quality,
    load_heldout_evaluation,
)
from .language import _card_sections, _render_messages
from .projection import (
    _project_sft_conversation,
)
from .reasoning_envelope import (
    REASONING_ENVELOPE_VERSION,
    audit_reasoning_envelopes,
    parse_reasoning_envelope,
    reasoning_envelope_card_hand,
)
from .selection import (
    _normalized_structure,
    _balance_response_card_hands,
    _balance_task_domains,
    _balance_task_families,
    _deduplicate_exact_prompts,
    _deduplicate_exact_responses,
    _deduplicate_structural_rows,
)
from .schema import IGNORE_INDEX, LABEL_DTYPE, PROJECTED_SFT_SCHEMA, TOKEN_DTYPE
from .surface_selection import select_balanced_sft_surfaces
from .surface_variation import SurfaceVariationBalancer


TRAIN_QUALITY_MAX_RESPONSE_CARD_HAND_SHARE = 0.05
SOURCE_READ_BATCH_SIZE = 10_000
SURFACE_REWRITE_BATCH_SIZE = 10_000
_INSTRUCTION_COLUMNS = (
    "example_id",
    "task",
    "mode",
    "domain",
    "language",
    "split",
    "messages",
    "answer_json",
    "source",
    "license",
    "version",
)
# A 5% domain ceiling forces all twenty-domain families into an exact uniform
# distribution and discards otherwise diverse, unique supervision as soon as
# one domain loses more rows during exact/structural deduplication.  Ten percent
# still prevents a single semantic domain from dominating a family while
# preserving enough naturally uneven material for the release-scale corpus.
TRAIN_QUALITY_MAX_DOMAIN_SHARE = 0.10
TRAIN_QUALITY_MAX_STYLE_PREFIX_SHARE = 0.05
TRAIN_QUALITY_MAX_MULTISCALE_REPETITION_SHARE = 0.045
TRAIN_QUALITY_REPETITION_SAMPLE_PER_FAMILY = 10_000
TRAIN_QUALITY_MIN_AUTHORED_CONVERSATION_SHARE = 0.05
TRAIN_QUALITY_REASONING_ENVELOPE_SHARE = (0.15, 0.25)

_STYLE_WORD = re.compile(r"[a-z0-9']+")
_STYLE_NUMBER = re.compile(r"\d+(?:[.,:/]\d+)*(?:%|°c)?", re.IGNORECASE)
_MODEL_FACING_META_STEMS = (
    "a useful transfer question",
    "the available evidence",
    "the supplied details",
    "the claim should remain",
    "the process can be summarized",
)

SUPPLEMENT_TASK_ALIASES = {
    "empathetic_dialogue": "conversation_empathy",
    "practical_dialogue": "practical_action",
}

_PRACTICAL_STAGE_TASKS = {
    "ask_for_missing_detail": "context_clarification",
    "present_bounded_options": "planning_comparison",
}


def _project_surface(
    row: dict[str, Any], selection_key: str
) -> tuple[list[dict[str, str]], TrainingCards]:
    return _project_sft_conversation(
        row["messages"],
        example_id=selection_key,
        task=row["task"],
        answer_json=row["answer_json"],
        reasoning_envelope_version=row.get("_reasoning_envelope_version"),
        reasoning_seed=row["example_id"],
    )


def _reasoning_projection_fields(row: dict[str, Any]) -> dict[str, Any]:
    envelope = parse_reasoning_envelope(row["_projected_target"])
    if envelope is None:
        return {
            "reasoning_envelope": False,
            "reasoning_trace": "",
            "final_response": row["_projected_target"],
            "reasoning_card_hand": "",
        }
    return {
        "reasoning_envelope": True,
        "reasoning_trace": envelope.think,
        "final_response": envelope.final,
        "reasoning_card_hand": reasoning_envelope_card_hand(
            row["task"], row["example_id"]
        ),
    }


def _audit_model_facing_style(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure repeated prose seen by the loss, not internal card metadata."""

    train_records = [row for row in records if row["split"] == "train"]
    task_counts = Counter(row["task"] for row in train_records)
    stem_counts: dict[str, Counter[str]] = defaultdict(Counter)
    prefix_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train_records:
        task = row["task"]
        response = row["response"].casefold()
        for stem in _MODEL_FACING_META_STEMS:
            if stem in response:
                stem_counts[task][stem] += 1
        protocol_free = re.sub(r"</?(?:think|final)>", " ", response)
        prefixes: set[str] = set()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", protocol_free):
            normalized = _STYLE_NUMBER.sub(" number ", sentence)
            words = _STYLE_WORD.findall(normalized)
            if len(words) >= 8:
                prefixes.add(" ".join(words[:6]))
        prefix_counts[task].update(prefixes)

    task_audits: dict[str, Any] = {}
    prefix_failures: list[str] = []
    stem_failures: list[str] = []
    for task, total in sorted(task_counts.items()):
        prefixes = prefix_counts[task]
        stems = stem_counts[task]
        maximum_prefix, maximum_count = (
            prefixes.most_common(1)[0] if prefixes else ("", 0)
        )
        maximum_share = maximum_count / total if total else 0.0
        structured_prose_exempt = task == "extraction_classification"
        if (
            not structured_prose_exempt
            and maximum_share > TRAIN_QUALITY_MAX_STYLE_PREFIX_SHARE
        ):
            prefix_failures.append(task)
        stem_shares = {
            stem: count / total if total else 0.0
            for stem, count in stems.items()
        }
        if any(
            share > TRAIN_QUALITY_MAX_STYLE_PREFIX_SHARE
            for share in stem_shares.values()
        ):
            stem_failures.append(task)
        task_audits[task] = {
            "examples": total,
            "maximum_six_word_prefix": maximum_prefix,
            "maximum_six_word_prefix_count": maximum_count,
            "maximum_six_word_prefix_share": round(maximum_share, 6),
            "structured_prose_exempt": structured_prose_exempt,
            "forbidden_meta_stem_counts": dict(sorted(stems.items())),
            "forbidden_meta_stem_shares": {
                stem: round(share, 6)
                for stem, share in sorted(stem_shares.items())
            },
        }
    checks = {
        "no_forbidden_model_facing_meta_stems": not stem_failures,
        "maximum_six_word_sentence_prefix_share_at_most_5_percent_per_family": (
            not prefix_failures
        ),
    }
    return {
        "tasks": task_audits,
        "prefix_failure_tasks": prefix_failures,
        "meta_stem_failure_tasks": stem_failures,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _deal_surface(row: dict[str, Any], selection_key: str) -> TrainingCards:
    try:
        metadata = json.loads(row["answer_json"]) if row["answer_json"] else {}
    except json.JSONDecodeError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return deal_training_cards(
        task=row["task"],
        mode="chat" if len(row["messages"]) > 2 else "instruct",
        example_id=selection_key,
        metadata=metadata,
    )


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
        parquet = pq.ParquetFile(path)
        source_rows = [
            row
            for batch in parquet.iter_batches(
                batch_size=SOURCE_READ_BATCH_SIZE,
                columns=list(_INSTRUCTION_COLUMNS),
            )
            for row in batch.to_pylist()
        ]
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
        raise ValueError(
            f"instruction sources contain duplicate id: {duplicate_ids[0]}"
        )
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
    """Project and tokenize SFT data without hidden scale-dependent truncation.

    Exact duplicates are always removed. Family, structure, domain, and card-hand
    caps are opt-in recovery controls; by default every other compatible row is
    preserved and distribution quality is reported as ratios in the manifest.
    """
    if target_training_examples is not None and target_training_examples < 1:
        raise ValueError("target_training_examples must be positive")
    if target_supervised_tokens is not None and target_supervised_tokens < 1:
        raise ValueError("target_supervised_tokens must be positive")
    if reasoning_envelope_version not in {None, REASONING_ENVELOPE_VERSION}:
        raise ValueError(
            f"unsupported reasoning envelope version: {reasoning_envelope_version}"
        )
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
    if casual_registry_path is not None:
        # Import lazily so the conversational renderer can reuse the SFT schema
        # without creating a module-import cycle.
        from ..conversational import render_casual_conversation_rows

        casual_rows, casual_summary = render_casual_conversation_rows(
            casual_registry_path,
            seed=42,
            validation_percent=5,
        )
        existing_ids = {row["example_id"] for row in source_rows}
        duplicate_ids = existing_ids & {
            row["example_id"] for row in casual_rows
        }
        if duplicate_ids:
            raise ValueError(
                "native casual family duplicates instruction id: "
                + min(duplicate_ids)
            )
        source_rows.extend(casual_rows)
        source_rows.sort(key=lambda row: row["example_id"])
        instruction_sources.append(
            {
                "kind": "generated_native_family",
                "family": "casual_conversation",
                "registry_path": str(casual_registry_path),
                "registry_sha256": casual_summary["registry"]["sha256"],
                "sha256": casual_summary["content_sha256"],
                "examples": len(casual_rows),
                "task_aliases": {},
                "semantic_capacity": casual_summary["semantic_capacity"],
                "audit": casual_summary["audit"],
            }
        )
        material = "\n".join(source["sha256"] for source in instruction_sources)
        instruction_sources_sha256 = hashlib.sha256(material.encode()).hexdigest()
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

    for row in source_rows:
        row["_reasoning_envelope_version"] = reasoning_envelope_version

    projected_rows, surface_selection = select_balanced_sft_surfaces(
        source_rows,
        dealer=_deal_surface,
        projector=_project_surface,
        workers=workers,
    )
    for row in projected_rows:
        row["_source_representation"] = (
            "card_hand" if _card_sections(row["messages"]) is not None else "conversation"
        )
        del row["messages"]
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
    domain_balance: dict[str, Any] = {}
    response_card_balance: dict[str, Any] = {}
    deduplication: dict[str, Any] = {}
    for partition, partition_rows in list(grouped.items()):
        partition_rows, exact_deduplication[partition] = _deduplicate_exact_responses(
            partition_rows
        )
        partition_rows, exact_prompt_deduplication[partition] = (
            _deduplicate_exact_prompts(partition_rows)
        )
        structure_limit = max_per_structure if partition == "train" else None
        per_task_limits = (
            {"extraction_classification": max_per_structure * 4}
            if partition == "train" and max_per_structure is not None
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
                max_examples_per_family=max_examples_per_family,
            )
            family_balance[partition]["policy"] = {
                "default": "preserve_all_and_audit_ratios",
                "manual_cap": max_examples_per_family,
            }
            partition_rows, domain_balance[partition] = _balance_task_domains(
                partition_rows,
                maximum_share=max_domain_share,
            )
            partition_rows, response_card_balance[partition] = (
                _balance_response_card_hands(
                    partition_rows,
                    maximum_share=max_response_card_hand_share,
                )
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
            domain_balance[partition] = {
                "input_examples": len(partition_rows),
                "kept_examples": len(partition_rows),
                "dropped_overrepresented_domains": 0,
                "requested_maximum_share": None,
                "tasks_requiring_tank_hydration": [],
                "tasks": {},
            }
            response_card_balance[partition] = {
                "input_examples": len(partition_rows),
                "kept_examples": len(partition_rows),
                "dropped_overrepresented_response_hands": 0,
                "requested_maximum_share": None,
                "tasks": {},
            }
        grouped[partition] = partition_rows

    # Phrase reservoirs are balanced against the rows that will actually be
    # trained. Applying this earlier lets later deduplication skew a balanced
    # reservoir back above its ceiling.
    phrase_balancer = SurfaceVariationBalancer()
    rewritten_train = grouped.get("train", [])
    for batch_start in range(0, len(rewritten_train), SURFACE_REWRITE_BATCH_SIZE):
        batch = rewritten_train[
            batch_start : batch_start + SURFACE_REWRITE_BATCH_SIZE
        ]
        for row in batch:
            # Independently authored conversations are already model-facing
            # prose. Phrase balancing is for generated card hands; applying it
            # here can corrupt natural turns (for example, rewriting "I want"
            # to "My immediate goal" after a comma).
            if row["_source_representation"] == "conversation":
                continue
            envelope = parse_reasoning_envelope(row["_projected_target"])
            if envelope is not None:
                rewritten_prefix = phrase_balancer.rewrite_messages(
                    row["_projected_messages"][:-1],
                    task=row["task"],
                    example_id=row["example_id"],
                )
                rewritten_final = phrase_balancer.rewrite_messages(
                    [{"role": "assistant", "content": envelope.final}],
                    task=row["task"],
                    example_id=row["example_id"],
                )[0]["content"]
                target = (
                    f"<think>\n{envelope.think}\n</think>\n"
                    f"<final>\n{rewritten_final}\n</final>"
                )
                messages = [
                    *rewritten_prefix,
                    {**row["_projected_messages"][-1], "content": target},
                ]
            else:
                messages = phrase_balancer.rewrite_messages(
                    row["_projected_messages"],
                    task=row["task"],
                    example_id=row["example_id"],
                )
            row["_projected_messages"] = messages
            row["_projected_prompt"] = _render_messages(messages[:-1])
            row["_projected_target"] = messages[-1]["content"]
            row["_structure_signature"] = _normalized_structure(
                row["_projected_target"]
            )
    rewritten_train, post_variation_response_deduplication = (
        _deduplicate_exact_responses(rewritten_train)
    )
    rewritten_train, post_variation_prompt_deduplication = (
        _deduplicate_exact_prompts(rewritten_train)
    )
    grouped["train"] = rewritten_train
    surface_selection["phrase_variation"] = phrase_balancer.audit()
    surface_selection["post_variation_exact_response_deduplication"] = (
        post_variation_response_deduplication
    )
    surface_selection["post_variation_exact_prompt_deduplication"] = (
        post_variation_prompt_deduplication
    )

    projection_audit = _audit_sft_projection(
        [row for rows in grouped.values() for row in rows]
    )
    reasoning_envelope_quality = audit_reasoning_envelopes(
        [row for rows in grouped.values() for row in rows],
        enabled=reasoning_envelope_version == REASONING_ENVELOPE_VERSION,
    )
    if not reasoning_envelope_quality["passed"]:
        raise ValueError(
            "reasoning envelope quality audit failed: "
            + json.dumps(
                {
                    "checks": reasoning_envelope_quality["checks"],
                    "calculation_failure_examples": reasoning_envelope_quality[
                        "calculation_failure_examples"
                    ],
                },
                sort_keys=True,
            )
        )
    statistical_quality_audit = audit_rows_quality(
        [row for rows in grouped.values() for row in rows],
        input_label="model-facing SFT projection",
        prompt_key="_projected_prompt",
        response_key="_projected_target",
        sample_size=None,
        near_duplicate_threshold=0.95,
        max_features=None,
        workers=workers,
    )

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
            "difficulty": projected_difficulty(
                row["_conditioning_cards"],
                row["_projected_messages"],
            ),
            "domain": row["domain"],
            "language": row["language"],
            "split": "validation" if partition == "eval" else partition,
            "messages": row["_projected_messages"],
            "prompt": row["_projected_prompt"],
            "response": row["_projected_target"],
            **_reasoning_projection_fields(row),
            "structure_signature": row["_structure_signature"],
            "response_card_hand": row[
                "_conditioning_cards"
            ].response_structure_signature,
            "source_representation": (
                row["_source_representation"]
            ),
            "source": row["source"],
            "license": row["license"],
            "version": row["version"],
        }
        for partition, partition_rows in sorted(grouped.items())
        for row in partition_rows
    ]
    casual_conversation_quality = audit_casual_conversation_quality(projected_records)
    model_facing_style_quality = _audit_model_facing_style(projected_records)
    model_facing_repetition_quality = audit_sft_repetition_quality(
        grouped.get("train", []),
        maximum_share=TRAIN_QUALITY_MAX_MULTISCALE_REPETITION_SHARE,
        maximum_examples_per_task=TRAIN_QUALITY_REPETITION_SAMPLE_PER_FAMILY,
    )
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
                has_card_hand = row["_source_representation"] == "card_hand"
                input_ids, labels, conditioning_cards = _encode_messages(
                    [],
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
                            "response_card_hand": (
                                conditioning_cards.response_structure_signature
                            ),
                            "reasoning_envelope": _reasoning_projection_fields(row)[
                                "reasoning_envelope"
                            ],
                            "reasoning_card_hand": _reasoning_projection_fields(row)[
                                "reasoning_card_hand"
                            ],
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
        "surface_selection": surface_selection,
        "statistical_quality_audit": statistical_quality_audit,
        "casual_conversation_quality": casual_conversation_quality,
        "model_facing_style_quality": model_facing_style_quality,
        "model_facing_repetition_quality": model_facing_repetition_quality,
        "reasoning_envelope_quality": reasoning_envelope_quality,
        "exact_response_deduplication": exact_deduplication,
        "exact_prompt_deduplication": exact_prompt_deduplication,
        "family_balance": family_balance,
        "domain_balance": domain_balance,
        "response_card_balance": response_card_balance,
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
    core_family_counts = {
        task: count
        for task, count in train_family_counts.items()
        if task != "casual_conversation"
    }
    core_train_count = sum(core_family_counts.values())
    core_family_shares = {
        task: count / core_train_count if core_train_count else 0.0
        for task, count in core_family_counts.items()
    }
    exact_train_responses = len({row["response"] for row in train_records})
    exact_train_prompts = len({row["prompt"] for row in train_records})
    multi_turn_count = sum(len(row["messages"]) > 2 for row in train_records)
    authored_conversation_count = sum(
        row["source_representation"] == "conversation" for row in train_records
    )
    authored_conversation_share = (
        authored_conversation_count / train_count if train_count else 0.0
    )
    reasoning_envelope_count = sum(
        bool(row["reasoning_envelope"]) for row in train_records
    )
    reasoning_envelope_share = (
        reasoning_envelope_count / train_count if train_count else 0.0
    )
    authored_multi_turn_count = sum(
        len(row["messages"]) > 2 and row["source_representation"] == "conversation"
        for row in train_records
    )
    linked_card_multi_turn_count = multi_turn_count - authored_multi_turn_count
    multi_turn_share = multi_turn_count / train_count if train_count else 0.0
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
    card_hands_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    card_siblings_by_task: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for row in train_records:
        task = row["task"]
        hand = row["response_card_hand"]
        card_hands_by_task[task][hand] += 1
        if task in RESPONSE_STRUCTURE_SIBLING_TASKS:
            axes = hand.split("|")
            if len(axes) != 4:
                raise ValueError(f"invalid response-card hand signature: {hand!r}")
            for omitted, name in enumerate(
                ("order", "bridge", "layout", "opening")
            ):
                sibling = "|".join(
                    value for index, value in enumerate(axes) if index != omitted
                )
                card_siblings_by_task[task][f"without_{name}"][sibling] += 1
    maximum_card_hand_share = max(
        (
            max(counts.values()) / sum(counts.values())
            for counts in card_hands_by_task.values()
            if counts
        ),
        default=0.0,
    )
    maximum_card_sibling_share = max(
        (
            max(counts.values()) / sum(counts.values())
            for dimensions in card_siblings_by_task.values()
            for counts in dimensions.values()
            if counts
        ),
        default=0.0,
    )
    quality_checks = {
        "no_exact_duplicate_train_responses": exact_train_responses == train_count,
        "no_exact_duplicate_train_prompts": exact_train_prompts == train_count,
        "at_least_fourteen_training_families": len(train_family_counts) >= 14,
        "maximum_family_share_at_most_15_percent": max(
            family_shares.values(), default=0.0
        )
        <= 0.15,
        "minimum_family_share_at_least_2_percent": min(
            core_family_shares.values(), default=0.0
        )
        >= 0.02,
        "has_easy_medium_and_hard_examples": set(difficulty_counts)
        == {"easy", "medium", "hard"},
        "easy_examples_are_at_least_20_percent": (
            difficulty_counts.get("easy", 0) / train_count if train_count else 0.0
        )
        >= 0.20,
        "multi_turn_share_between_10_and_30_percent": 0.10
        <= multi_turn_share
        <= 0.30,
        "four_response_length_bands_each_at_least_5_percent": all(
            count / train_count >= 0.05 if train_count else False
            for count in response_length_bands.values()
        ),
        "distinct_structure_share_at_least_20_percent": (
            distinct_train_structures / train_count if train_count else 0.0
        )
        >= 0.20,
        "maximum_response_card_hand_share_at_most_5_percent": (
            maximum_card_hand_share <= TRAIN_QUALITY_MAX_RESPONSE_CARD_HAND_SHARE
        ),
        "maximum_response_card_sibling_share_at_most_5_percent": (
            maximum_card_sibling_share
            <= TRAIN_QUALITY_MAX_RESPONSE_CARD_HAND_SHARE
        ),
        "sklearn_statistical_quality_audit_passed": statistical_quality_audit[
            "passed"
        ],
        "model_facing_style_repetition_audit_passed": (
            model_facing_style_quality["passed"]
        ),
        "all_fourteen_core_families_have_multiscale_repetition_metrics": (
            len(core_family_counts) == 14
            and set(core_family_counts)
            <= set(model_facing_repetition_quality["tasks"])
            and all(
                model_facing_repetition_quality["tasks"][task]["audited"]
                for task in core_family_counts
            )
        ),
        "supervised_response_multiscale_repetition_audit_passed": (
            model_facing_repetition_quality["supervised_passed"]
        ),
        "authored_conversation_share_at_least_5_percent": (
            authored_conversation_share
            >= TRAIN_QUALITY_MIN_AUTHORED_CONVERSATION_SHARE
        ),
        "heldout_evaluation_has_at_least_28_authored_examples": (
            manifests.get("eval", {}).get("examples", 0) >= 28
        ),
        "diagnostic_companion_has_500_to_1000_examples": 500
        <= manifests.get("diagnostic", {}).get("examples", 0)
        <= 1_000,
    }
    if require_casual_conversation:
        quality_checks["casual_conversation_is_present"] = (
            train_family_counts.get("casual_conversation", 0) > 0
        )
        quality_checks["casual_conversation_quality_passed"] = (
            casual_conversation_quality["passed"]
        )
    if reasoning_envelope_version == REASONING_ENVELOPE_VERSION:
        quality_checks["reasoning_envelope_v18_quality_passed"] = (
            reasoning_envelope_quality["passed"]
        )
        quality_checks["reasoning_envelope_share_between_15_and_25_percent"] = (
            TRAIN_QUALITY_REASONING_ENVELOPE_SHARE[0]
            <= reasoning_envelope_share
            <= TRAIN_QUALITY_REASONING_ENVELOPE_SHARE[1]
        )
    if target_training_examples is not None:
        quality_checks["training_examples_reach_requested_target"] = (
            train_count >= target_training_examples
        )
    if target_supervised_tokens is not None:
        quality_checks["supervised_tokens_reach_requested_target"] = (
            manifests.get("train", {}).get("supervised_tokens", 0)
            >= target_supervised_tokens
        )
    manifest["release_quality"] = {
        "ready": all(quality_checks.values()),
        "checks": quality_checks,
        "train_family_counts": train_family_counts,
        "train_family_shares": family_shares,
        "core_train_family_shares": {
            task: round(share, 6)
            for task, share in sorted(core_family_shares.items())
        },
        "required_casual_conversation": require_casual_conversation,
        "reasoning_envelope_version": reasoning_envelope_version,
        "reasoning_envelope_examples": reasoning_envelope_count,
        "reasoning_envelope_share": round(reasoning_envelope_share, 6),
        "model_facing_style_quality": model_facing_style_quality,
        "model_facing_repetition_quality": model_facing_repetition_quality,
        "difficulty_counts": difficulty_counts,
        "response_length_bands": dict(response_length_bands),
        "distinct_train_structures": distinct_train_structures,
        "distinct_train_structure_share": round(
            distinct_train_structures / train_count if train_count else 0.0, 6
        ),
        "distinct_response_card_hands_by_family": {
            task: len(counts) for task, counts in sorted(card_hands_by_task.items())
        },
        "maximum_response_card_hand_share": round(maximum_card_hand_share, 6),
        "maximum_response_card_sibling_share": round(
            maximum_card_sibling_share, 6
        ),
        "response_card_sibling_neighbourhoods_by_family": {
            task: {
                dimension: {
                    "distinct": len(counts),
                    "maximum_share": round(
                        max(counts.values(), default=0) / sum(counts.values()), 6
                    ),
                }
                for dimension, counts in sorted(dimensions.items())
            }
            for task, dimensions in sorted(card_siblings_by_task.items())
        },
        "statistical_quality_audit": statistical_quality_audit,
        "multi_turn_examples": multi_turn_count,
        "multi_turn_share": round(
            multi_turn_count / train_count if train_count else 0.0, 6
        ),
        "authored_multi_turn_examples": authored_multi_turn_count,
        "authored_multi_turn_share": round(
            authored_multi_turn_count / train_count if train_count else 0.0, 6
        ),
        "linked_card_multi_turn_examples": linked_card_multi_turn_count,
        "linked_card_multi_turn_share": round(
            linked_card_multi_turn_count / train_count if train_count else 0.0, 6
        ),
        "authored_conversation_examples": authored_conversation_count,
        "authored_conversation_share": round(authored_conversation_share, 6),
        "exact_train_response_uniqueness_ratio": round(
            exact_train_responses / train_count if train_count else 0.0, 6
        ),
        "exact_train_prompt_uniqueness_ratio": round(
            exact_train_prompts / train_count if train_count else 0.0, 6
        ),
        "target_supervised_training_tokens": target_supervised_tokens,
        "target_training_examples": target_training_examples,
        "scale_policy": (
            "manual_targets"
            if target_training_examples is not None
            or target_supervised_tokens is not None
            else "report_realized_scale_without_implicit_target"
        ),
        "selection_policy": {
            "default": "preserve_all_non_exact_rows",
            "exact_duplicates": "always_removed",
            "manual_max_examples_per_family": max_examples_per_family,
            "manual_max_per_structure": max_per_structure,
            "manual_max_domain_share": max_domain_share,
            "manual_max_response_card_hand_share": max_response_card_hand_share,
            "quality_maximum_family_share": 0.15,
            "quality_minimum_family_share": 0.02,
            "quality_maximum_domain_share": TRAIN_QUALITY_MAX_DOMAIN_SHARE,
            "quality_maximum_response_card_hand_share": (
                TRAIN_QUALITY_MAX_RESPONSE_CARD_HAND_SHARE
            ),
            "quality_maximum_style_prefix_share": (
                TRAIN_QUALITY_MAX_STYLE_PREFIX_SHARE
            ),
            "quality_minimum_authored_conversation_share": (
                TRAIN_QUALITY_MIN_AUTHORED_CONVERSATION_SHARE
            ),
            "quality_reasoning_envelope_share_range": list(
                TRAIN_QUALITY_REASONING_ENVELOPE_SHARE
            ),
        },
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return manifest
