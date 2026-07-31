from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .build import file_sha256
from .tokenize import directory_sha256, load_encoding


INSTRUCTION_SCHEMA = pa.schema(
    [
        ("example_id", pa.string()),
        ("task", pa.string()),
        ("mode", pa.string()),
        ("difficulty", pa.string()),
        ("dataset_id", pa.string()),
        ("domain", pa.string()),
        ("language", pa.string()),
        ("split", pa.string()),
        (
            "messages",
            pa.list_(
                pa.struct(
                    [
                        ("role", pa.string()),
                        ("content", pa.string()),
                    ]
                )
            ),
        ),
        ("prompt", pa.string()),
        ("response", pa.string()),
        ("rendered_text", pa.string()),
        ("source_keys", pa.list_(pa.string())),
        ("evidence", pa.list_(pa.string())),
        ("answer_json", pa.string()),
        ("source", pa.string()),
        ("source_urls", pa.list_(pa.string())),
        ("license", pa.string()),
        ("version", pa.string()),
    ]
)

TOKEN_DTYPE = np.dtype("<u4")
LABEL_DTYPE = np.dtype("<i4")
IGNORE_INDEX = -100


def _stable_index(value: str, size: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % size


def _pick(value: str, choices: tuple[str, ...]) -> str:
    return choices[_stable_index(value, len(choices))]


def _example_id(
    task: str,
    dataset_id: str,
    source_keys: Iterable[str],
    messages: list[dict[str, str]],
) -> str:
    material = "|".join(
        (task, dataset_id, *source_keys, _render_messages(messages))
    )
    suffix = hashlib.sha256(material.encode()).hexdigest()[:16]
    return f"atlas-instruct:{task}:{suffix}"


def _render_messages(messages: list[dict[str, str]]) -> str:
    labels = {"user": "User", "assistant": "Assistant"}
    return "\n".join(
        f"{labels[message['role']]}: {message['content']}" for message in messages
    )


def _row(
    *,
    task: str,
    difficulty: str,
    card: dict[str, Any],
    messages: list[dict[str, str]],
    source_keys: list[str],
    evidence: list[str],
    answer_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not messages or messages[0]["role"] != "user":
        raise ValueError("instruction conversations must begin with a user message")
    if messages[-1]["role"] != "assistant":
        raise ValueError("instruction conversations must end with an assistant message")
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected or not message["content"].strip():
            raise ValueError("instruction roles must alternate and contain text")
    return {
        "example_id": _example_id(task, card["dataset_id"], source_keys, messages),
        "task": task,
        "mode": "instruct" if len(messages) == 2 else "chat",
        "difficulty": difficulty,
        "dataset_id": card["dataset_id"],
        "domain": card["domain"],
        "language": card["language"],
        "split": card["split"],
        "messages": messages,
        "prompt": messages[0]["content"],
        "response": messages[-1]["content"],
        "rendered_text": _render_messages(messages),
        "source_keys": source_keys,
        "evidence": evidence,
        "answer_json": (
            json.dumps(answer_json, sort_keys=True, ensure_ascii=False)
            if answer_json is not None
            else ""
        ),
        "source": card["source"],
        "source_urls": card["source_urls"],
        "license": card["license"],
        "version": card["version"],
    }


def _entity_rows(
    card: dict[str, Any],
    *,
    max_attributes_per_card: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    name = card["name"]
    key = card["key"]
    task_key = f"{card['dataset_id']}:{key}"
    question = _pick(
        task_key,
        (
            f"What is {name} according to the atlas?",
            f"Give a concise, grounded description of {name}.",
            f"Summarize the atlas entry for {name} without adding new facts.",
        ),
    )
    rows.append(
        _row(
            task="entity_summary",
            difficulty="easy",
            card=card,
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": card["description"]},
            ],
            source_keys=[key],
            evidence=[card["description"]],
        )
    )

    attributes = json.loads(card["attributes_json"])
    selected_attributes = sorted(
        attributes.items(),
        key=lambda item: hashlib.sha256(
            f"{task_key}:{item[0]}".encode()
        ).digest(),
    )[:max_attributes_per_card]
    for attribute, value in sorted(selected_attributes):
        readable = attribute.replace("_", " ")
        question = _pick(
            f"{task_key}:{attribute}",
            (
                f"What is the recorded {readable} of {name}?",
                f"In the atlas, which value is listed for {name}'s {readable}?",
                f"Answer from the card only: what is {name}'s {readable}?",
            ),
        )
        answer = f"The recorded {readable} of {name} is {json.dumps(value, ensure_ascii=False)}."
        rows.append(
            _row(
                task="attribute_query",
                difficulty="easy",
                card=card,
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                source_keys=[key],
                evidence=[f"{attribute}={json.dumps(value, ensure_ascii=False)}"],
            )
        )

    if card["facts"]:
        fact = card["facts"][_stable_index(task_key, len(card["facts"]))]
        rows.append(
            _row(
                task="recorded_fact",
                difficulty="easy",
                card=card,
                messages=[
                    {
                        "role": "user",
                        "content": f"State one documented fact about {name}.",
                    },
                    {"role": "assistant", "content": fact},
                ],
                source_keys=[key],
                evidence=[fact],
            )
        )

    answer_object = {
        "key": key,
        "kind": card["kind"],
        "name": name,
        "summary": card["summary"],
        "attributes": attributes,
    }
    answer_text = json.dumps(answer_object, sort_keys=True, ensure_ascii=False)
    rows.append(
        _row(
            task="structured_extraction",
            difficulty="medium",
            card=card,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Return the atlas record for {name} as JSON with exactly "
                        "these fields: key, kind, name, summary, attributes."
                    ),
                },
                {"role": "assistant", "content": answer_text},
            ],
            source_keys=[key],
            evidence=[card["summary"], card["attributes_json"]],
            answer_json=answer_object,
        )
    )

    if attributes:
        attribute = sorted(attributes)[_stable_index(f"followup:{task_key}", len(attributes))]
        value = attributes[attribute]
        readable = attribute.replace("_", " ")
        rows.append(
            _row(
                task="grounded_followup",
                difficulty="medium",
                card=card,
                messages=[
                    {"role": "user", "content": f"Briefly identify {name}."},
                    {"role": "assistant", "content": card["summary"]},
                    {"role": "user", "content": f"What is its recorded {readable}?"},
                    {
                        "role": "assistant",
                        "content": f"Its recorded {readable} is {json.dumps(value, ensure_ascii=False)}.",
                    },
                ],
                source_keys=[key],
                evidence=[card["summary"], f"{attribute}={json.dumps(value, ensure_ascii=False)}"],
            )
        )
    return rows


def build_instruction_dataset(
    corpus_root: Path,
    output_root: Path,
    *,
    max_attributes_per_card: int = 2,
    max_relations_per_card: int = 2,
    max_paths_per_card: int = 1,
) -> dict[str, Any]:
    if (
        max_attributes_per_card < 0
        or max_relations_per_card < 0
        or max_paths_per_card < 0
    ):
        raise ValueError("per-card limits cannot be negative")
    cards = pq.read_table(corpus_root / "cards.parquet").to_pylist()
    relations = pq.read_table(corpus_root / "relations.parquet").to_pylist()
    documents = pq.read_table(corpus_root / "documents.parquet").to_pylist()
    card_index = {(row["dataset_id"], row["key"]): row for row in cards}

    rows: list[dict[str, Any]] = []
    for card in cards:
        rows.extend(
            _entity_rows(
                card,
                max_attributes_per_card=max_attributes_per_card,
            )
        )

    relations_by_card: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for relation in relations:
        relations_by_card[(relation["dataset_id"], relation["from_key"])].append(relation)
    for source_ref, source_relations in sorted(relations_by_card.items()):
        source_card = card_index[source_ref]
        for relation in sorted(
            source_relations,
            key=lambda item: (item["relation"], item["to_dataset_id"], item["to_key"]),
        )[:max_relations_per_card]:
            target_ref = (relation["to_dataset_id"], relation["to_key"])
            target_card = card_index[target_ref]
            relation_name = relation["relation"].replace("_", " ")
            answer = relation["detail"] or (
                f"{source_card['name']} {relation_name} {target_card['name']}."
            )
            rows.append(
                _row(
                    task="direct_relation",
                    difficulty="medium",
                    card=source_card,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"How does the atlas connect {source_card['name']} "
                                f"to {target_card['name']}?"
                            ),
                        },
                        {"role": "assistant", "content": answer},
                    ],
                    source_keys=[source_card["key"], target_card["key"]],
                    evidence=[answer],
                    answer_json={
                        "from": source_card["key"],
                        "relation": relation["relation"],
                        "to": target_card["key"],
                    },
                )
            )

    cards_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        cards_by_group[(card["dataset_id"], card["kind"])].append(card)
    for group_cards in cards_by_group.values():
        ordered = sorted(group_cards, key=lambda item: item["key"])
        for left, right in zip(ordered[::2], ordered[1::2]):
            answer = (
                f"{left['name']}: {left['summary']}\n"
                f"{right['name']}: {right['summary']}"
            )
            rows.append(
                _row(
                    task="compare_records",
                    difficulty="medium",
                    card=left,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Compare the recorded atlas summaries of {left['name']} "
                                f"and {right['name']}. Do not infer beyond the cards."
                            ),
                        },
                        {"role": "assistant", "content": answer},
                    ],
                    source_keys=[left["key"], right["key"]],
                    evidence=[left["summary"], right["summary"]],
                )
            )

    path_counts: Counter[tuple[str, str]] = Counter()
    for document in sorted(documents, key=lambda item: item["document_id"]):
        if document["template"] != "path" or len(document["source_keys"]) < 2:
            continue
        start_ref = (document["dataset_id"], document["source_keys"][0])
        if path_counts[start_ref] >= max_paths_per_card:
            continue
        start_card = card_index[start_ref]
        end_card = card_index[(document["dataset_id"], document["source_keys"][-1])]
        answer = document["text"].split("\n\n", 1)[-1].strip()
        rows.append(
            _row(
                task="multi_hop_path",
                difficulty="hard",
                card=start_card,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Using only recorded links, trace a path from {start_card['name']} "
                            f"to {end_card['name']}."
                        ),
                    },
                    {"role": "assistant", "content": answer},
                ],
                source_keys=document["source_keys"],
                evidence=[answer],
            )
        )
        path_counts[start_ref] += 1

    rows.sort(key=lambda row: row["example_id"])
    example_ids = [row["example_id"] for row in rows]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("instruction example IDs are not unique")
    source_splits = {
        (card["dataset_id"], card["key"]): card["split"] for card in cards
    }
    for row in rows:
        if any(
            source_splits[(row["dataset_id"], key)] != row["split"]
            for key in row["source_keys"]
        ):
            raise ValueError(f"split leakage in {row['example_id']}")

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    table = pa.Table.from_pylist(rows, schema=INSTRUCTION_SCHEMA)
    parquet_path = temporary / "instructions.parquet"
    pq.write_table(
        table,
        parquet_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    counts = {
        "examples": len(rows),
        "examples_by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "examples_by_task": dict(sorted(Counter(row["task"] for row in rows).items())),
        "examples_by_mode": dict(sorted(Counter(row["mode"] for row in rows).items())),
        "source_cards": len(cards),
        "source_relations": len(relations),
    }
    manifest = {
        "format": "complexity-atlas-instruct-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY-NC 4.0",
        "counts": counts,
        "generation": {
            "method": "deterministic templates over authored linked cards",
            "max_attributes_per_card": max_attributes_per_card,
            "max_relations_per_card": max_relations_per_card,
            "max_paths_per_card": max_paths_per_card,
            "model_generated": False,
        },
        "source_corpus": {
            "path": str(corpus_root.resolve()),
            "cards_sha256": file_sha256(corpus_root / "cards.parquet"),
            "relations_sha256": file_sha256(corpus_root / "relations.parquet"),
            "documents_sha256": file_sha256(corpus_root / "documents.parquet"),
        },
        "files": {
            "instructions.parquet": {
                "bytes": parquet_path.stat().st_size,
                "sha256": file_sha256(parquet_path),
            }
        },
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return manifest


def _encode_messages(messages: list[dict[str, str]], encoding, eos_id: int) -> tuple[list[int], list[int]]:
    full_ids: list[int] = []
    target_labels: list[int] = []
    for message in messages:
        if message["role"] == "user":
            segment = f"User: {message['content']}\n"
            tokens = encoding.encode(segment, disallowed_special=())
            full_ids.extend(tokens)
            target_labels.extend([IGNORE_INDEX] * len(tokens))
        else:
            prefix = encoding.encode("Assistant: ", disallowed_special=())
            response = encoding.encode(f"{message['content']}\n", disallowed_special=())
            full_ids.extend(prefix)
            target_labels.extend([IGNORE_INDEX] * len(prefix))
            full_ids.extend(response)
            target_labels.extend(response)
    full_ids.append(eos_id)
    target_labels.append(eos_id)
    # Causal alignment: logits at position t predict token t+1. Supervision is
    # active only when that next token belongs to an assistant response.
    return full_ids[:-1], target_labels[1:]


def tokenize_instruction_dataset(
    instructions_path: Path,
    tokenizer_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    encoding, tokenizer_config = load_encoding(tokenizer_root)
    eos_token = tokenizer_config.get("eos_token", "<|endoftext|>")
    try:
        eos_id = encoding.encode_single_token(eos_token)
    except KeyError:
        eos_id = encoding.eot_token
    rows = sorted(
        pq.read_table(instructions_path).to_pylist(),
        key=lambda row: row["example_id"],
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        partition = {"train": "train", "validation": "eval", "test": "test"}[row["split"]]
        grouped[partition].append(row)

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    manifests: dict[str, Any] = {}
    for partition, partition_rows in sorted(grouped.items()):
        root = temporary / partition
        root.mkdir()
        inputs_path = root / "input_ids.bin"
        labels_path = root / "labels.bin"
        examples_path = root / "examples.jsonl"
        offset = 0
        supervised_tokens = 0
        with inputs_path.open("wb") as inputs_handle, labels_path.open("wb") as labels_handle, examples_path.open("w", encoding="utf-8") as examples_handle:
            for row in partition_rows:
                input_ids, labels = _encode_messages(row["messages"], encoding, eos_id)
                np.asarray(input_ids, dtype=TOKEN_DTYPE).tofile(inputs_handle)
                np.asarray(labels, dtype=LABEL_DTYPE).tofile(labels_handle)
                supervised = sum(label != IGNORE_INDEX for label in labels)
                examples_handle.write(
                    json.dumps(
                        {
                            "example_id": row["example_id"],
                            "task": row["task"],
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
            "source_sha256": file_sha256(instructions_path),
            "input_ids_sha256": file_sha256(inputs_path),
            "labels_sha256": file_sha256(labels_path),
            "examples_sha256": file_sha256(examples_path),
        }
        (root / "sft.idx.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        manifests[partition] = metadata
    manifest = {
        "format": "complexity-atlas-instruct-tokenized-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": tokenizer_config["encoding_name"],
        "serialization": "User: <content>\\nAssistant: <content>\\n",
        "partitions": manifests,
        "total_examples": sum(item["examples"] for item in manifests.values()),
        "total_tokens": sum(item["num_tokens"] for item in manifests.values()),
        "total_supervised_tokens": sum(item["supervised_tokens"] for item in manifests.values()),
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return manifest
