from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .semantic_audit import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    SentenceEmbedder,
    _load_embedder,
)


def _naturalize(field: str) -> str:
    return field.replace("_", " ").strip()


def _contract_text(contract: tuple[str, ...]) -> str:
    fields = ", ".join(_naturalize(field) for field in contract)
    return f"A correct response contains these semantic components: {fields}."


def _intent_text(surface_intent: str) -> str:
    return (
        f"The requested response should {surface_intent.strip().rstrip('.').lower()}."
    )


def _contract_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    contracts_by_pair: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
    for row in rows:
        answer = json.loads(str(row["answer_json"]))
        family = str(answer["family"])
        intent = str(answer["intent"])
        contract = tuple(answer["card_hand"]["completion_contract"])
        if not contract:
            raise ValueError(f"empty completion contract for {family}:{intent}")
        pair = (family, intent)
        contracts_by_pair[pair].add(contract)
        observed.setdefault(
            pair,
            {
                "family": family,
                "intent": intent,
                "surface_intent": str(answer["surface_intent"]),
                "contract": contract,
            },
        )
    ambiguous = {
        f"{family}:{intent}": sorted(map(list, contracts))
        for (family, intent), contracts in contracts_by_pair.items()
        if len(contracts) != 1
    }
    if ambiguous:
        raise ValueError(f"family-intent pairs expose multiple contracts: {ambiguous}")
    return [observed[key] for key in sorted(observed)]


def audit_contract_rows_with_embeddings(
    rows: list[dict[str, Any]],
    *,
    input_label: str,
    embedder: SentenceEmbedder,
    model_name: str,
    model_revision: str,
    batch_size: int = 128,
) -> dict[str, Any]:
    records = _contract_records(rows)
    if not records:
        raise ValueError("completion-contract audit requires at least one contract")

    intent_embeddings = np.asarray(
        embedder.encode(
            [_intent_text(record["surface_intent"]) for record in records],
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    contract_embeddings = np.asarray(
        embedder.encode(
            [_contract_text(record["contract"]) for record in records],
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    if intent_embeddings.shape != contract_embeddings.shape:
        raise ValueError("intent and completion-contract embedding shapes differ")

    by_family: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_family[record["family"]].append(index)

    family_reports: dict[str, Any] = {}
    correct = 0
    reciprocal_ranks: list[float] = []
    margins: list[float] = []
    for family, indices in sorted(by_family.items()):
        unique_contracts: list[tuple[str, ...]] = []
        representative_indices: list[int] = []
        for index in indices:
            contract = records[index]["contract"]
            if contract not in unique_contracts:
                unique_contracts.append(contract)
                representative_indices.append(index)
        candidate_embeddings = contract_embeddings[representative_indices]
        results = []
        family_correct = 0
        for index in indices:
            scores = intent_embeddings[index] @ candidate_embeddings.T
            order = np.argsort(-scores)
            expected_contract = records[index]["contract"]
            expected_candidate = unique_contracts.index(expected_contract)
            rank = int(np.where(order == expected_candidate)[0][0]) + 1
            best_candidate = int(order[0])
            is_correct = unique_contracts[best_candidate] == expected_contract
            correct += int(is_correct)
            family_correct += int(is_correct)
            reciprocal_ranks.append(1.0 / rank)
            expected_score = float(scores[expected_candidate])
            alternative_scores = np.delete(scores, expected_candidate)
            margin = (
                expected_score - float(alternative_scores.max())
                if len(alternative_scores)
                else expected_score
            )
            margins.append(margin)
            results.append(
                {
                    "intent": records[index]["intent"],
                    "surface_intent": records[index]["surface_intent"],
                    "expected_contract": list(expected_contract),
                    "predicted_contract": list(unique_contracts[best_candidate]),
                    "expected_rank": rank,
                    "expected_cosine": round(expected_score, 6),
                    "margin_over_best_alternative": round(margin, 6),
                    "matched": is_correct,
                }
            )
        family_reports[family] = {
            "intents": len(indices),
            "distinct_contracts": len(unique_contracts),
            "top1_contract_match_ratio": family_correct / len(indices),
            "results": results,
        }

    pairwise = contract_embeddings @ contract_embeddings.T
    np.fill_diagonal(pairwise, -1.0)
    nearest_distinct: list[float] = []
    for index, record in enumerate(records):
        candidates = [
            other
            for other, candidate in enumerate(records)
            if candidate["contract"] != record["contract"]
        ]
        if candidates:
            nearest_distinct.append(float(pairwise[index, candidates].max()))

    report = {
        "format": "completion-contract-embedding-audit-v1",
        "input": input_label,
        "rows": len(rows),
        "family_intent_pairs": len(records),
        "families": len(by_family),
        "model": {
            "name": model_name,
            "revision": model_revision,
            "embedding_dimensions": int(contract_embeddings.shape[1]),
        },
        "method": {
            "comparison": "natural intent versus distinct completion contracts within its family",
            "shared_contracts": "intents sharing an identical legitimate contract are equivalent candidates",
            "warning": "embedding alignment is a diagnostic for human review, not proof of semantic correctness",
        },
        "summary": {
            "top1_contract_match_ratio": correct / len(records),
            "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
            "mean_expected_margin": float(np.mean(margins)),
            "mean_nearest_distinct_contract_cosine": (
                float(np.mean(nearest_distinct)) if nearest_distinct else 0.0
            ),
        },
        "family_reports": family_reports,
        "structural_checks": {
            "one_contract_per_family_intent_pair": True,
            "all_contracts_non_empty": True,
            "all_fourteen_families_present": len(by_family) == 14,
        },
    }
    report["passed_structural_checks"] = all(report["structural_checks"].values())
    return report


def audit_contract_dataset_with_embeddings(
    conversations_path: Path,
    output_path: Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    model_revision: str = DEFAULT_EMBEDDING_REVISION,
    device: str | None = None,
    batch_size: int = 128,
) -> dict[str, Any]:
    rows = pq.read_table(conversations_path, columns=["answer_json"]).to_pylist()
    embedder = _load_embedder(model_name, model_revision, device)
    report = audit_contract_rows_with_embeddings(
        rows,
        input_label=str(conversations_path.resolve()),
        embedder=embedder,
        model_name=model_name,
        model_revision=model_revision,
        batch_size=batch_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
