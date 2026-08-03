from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .semantic_audit import SentenceEmbedder, _load_embedder


PROPOSAL_REVIEW_FIELDS = (
    "token",
    "consensus",
    "current_definition",
    "proposed_definition",
    "family",
    "domain",
    "primary_model",
    "primary_signal",
    "primary_token_delta",
    "primary_bare_token_delta",
    "primary_lexical_query_delta",
    "primary_family_delta",
    "primary_current_family_rank",
    "primary_proposed_family_rank",
    "secondary_model",
    "secondary_signal",
    "secondary_token_delta",
    "secondary_bare_token_delta",
    "secondary_lexical_query_delta",
    "secondary_family_delta",
    "secondary_current_family_rank",
    "secondary_proposed_family_rank",
    "reviewer_decision",
    "reviewer_notes",
)


def _encode(
    embedder: SentenceEmbedder,
    texts: list[str],
    *,
    batch_size: int,
) -> np.ndarray:
    matrix = np.asarray(
        embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    if matrix.ndim != 2 or len(matrix) != len(texts):
        raise ValueError("embedder returned an invalid proposal-review matrix")
    return matrix


def _family_rank(
    definition_embedding: np.ndarray,
    family_centroids: np.ndarray,
    selected_family_index: int,
) -> tuple[float, int]:
    similarities = definition_embedding @ family_centroids.T
    selected_similarity = float(similarities[selected_family_index])
    rank = int(np.count_nonzero(similarities > selected_similarity) + 1)
    return selected_similarity, rank


def _model_signal(
    *,
    bare_token_delta: float,
    lexical_query_delta: float,
    family_delta: float,
    current_family_rank: int,
    proposed_family_rank: int,
) -> str:
    """Return a token-definition signal, not a correctness decision.

    Family movement remains in the report but cannot veto a proposal: this
    queue exists specifically because the current family assignment may be
    wrong. Using that family as ground truth would make the audit circular.
    """

    del family_delta, current_family_rank, proposed_family_rank
    probe_signals = {
        (
            "improved" if delta >= 0.020
            else "regressed" if delta <= -0.020
            else "inconclusive"
        )
        for delta in (bare_token_delta, lexical_query_delta)
    }
    if probe_signals == {"improved", "regressed"}:
        return "mixed"
    if "improved" in probe_signals:
        return "improved"
    if "regressed" in probe_signals:
        return "regressed"
    return "inconclusive"


def audit_definition_proposals_data(
    dictionary: dict[str, Any],
    proposal_document: dict[str, Any],
    *,
    embedder: SentenceEmbedder,
    model_name: str,
    model_revision: str,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Compare draft definitions with current statistical glosses by embedding.

    The comparison is deliberately diagnostic. It never promotes a proposal to
    the canonical dictionary and never treats embedding similarity as lexical
    truth.
    """

    words = dictionary.get("words")
    raw_proposals = proposal_document.get("definitions")
    if not isinstance(words, dict) or not words:
        raise ValueError("proposal review requires a non-empty dictionary")
    if not isinstance(raw_proposals, dict) or not raw_proposals:
        raise ValueError("proposal review requires a non-empty definitions object")
    proposals = {
        str(token): str(definition).strip()
        for token, definition in raw_proposals.items()
        if str(definition).strip()
    }
    unknown = sorted(set(proposals) - set(words))
    if unknown:
        raise ValueError(
            "definition proposals contain unknown tokens: " + ", ".join(unknown[:20])
        )

    records = []
    for token, payload in sorted(words.items()):
        selected = payload.get("selected") or {}
        family = str(selected.get("family", "")).strip()
        definition = str(payload.get("short_definition", "")).strip()
        if not family or not definition:
            continue
        records.append(
            {
                "token": str(token),
                "definition": definition,
                "family": family,
                "domain": str(selected.get("domain", "")),
            }
        )
    by_token = {record["token"]: record for record in records}
    missing = sorted(set(proposals) - set(by_token))
    if missing:
        raise ValueError(
            "proposal tokens lack a usable dictionary assignment: "
            + ", ".join(missing[:20])
        )

    definition_embeddings = _encode(
        embedder,
        [record["definition"] for record in records],
        batch_size=batch_size,
    )
    indices_by_family: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        indices_by_family[record["family"]].append(index)
    ordered_families = sorted(indices_by_family)
    centroids = []
    for family in ordered_families:
        centroid = definition_embeddings[indices_by_family[family]].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        centroids.append(centroid / norm if norm else centroid)
    family_centroids = np.asarray(centroids, dtype=np.float32)

    proposal_tokens = sorted(proposals)
    bare_token_embeddings = _encode(
        embedder,
        proposal_tokens,
        batch_size=batch_size,
    )
    lexical_query_embeddings = _encode(
        embedder,
        [f'What is the meaning of the word "{token}"?' for token in proposal_tokens],
        batch_size=batch_size,
    )
    proposal_embeddings = _encode(
        embedder,
        [proposals[token] for token in proposal_tokens],
        batch_size=batch_size,
    )
    record_index = {record["token"]: index for index, record in enumerate(records)}
    rows = []
    signal_counts = {
        "improved": 0,
        "inconclusive": 0,
        "mixed": 0,
        "regressed": 0,
    }
    for index, token in enumerate(proposal_tokens):
        record = by_token[token]
        current_embedding = definition_embeddings[record_index[token]]
        proposal_embedding = proposal_embeddings[index]
        bare_token_embedding = bare_token_embeddings[index]
        lexical_query_embedding = lexical_query_embeddings[index]
        current_bare_token_cosine = float(bare_token_embedding @ current_embedding)
        proposed_bare_token_cosine = float(bare_token_embedding @ proposal_embedding)
        current_lexical_query_cosine = float(
            lexical_query_embedding @ current_embedding
        )
        proposed_lexical_query_cosine = float(
            lexical_query_embedding @ proposal_embedding
        )
        selected_family_index = ordered_families.index(record["family"])
        current_family_cosine, current_family_rank = _family_rank(
            current_embedding, family_centroids, selected_family_index
        )
        proposed_family_cosine, proposed_family_rank = _family_rank(
            proposal_embedding, family_centroids, selected_family_index
        )
        bare_token_delta = proposed_bare_token_cosine - current_bare_token_cosine
        lexical_query_delta = (
            proposed_lexical_query_cosine - current_lexical_query_cosine
        )
        token_delta = (bare_token_delta + lexical_query_delta) / 2.0
        family_delta = proposed_family_cosine - current_family_cosine
        signal = _model_signal(
            bare_token_delta=bare_token_delta,
            lexical_query_delta=lexical_query_delta,
            family_delta=family_delta,
            current_family_rank=current_family_rank,
            proposed_family_rank=proposed_family_rank,
        )
        signal_counts[signal] += 1
        rows.append(
            {
                "token": token,
                "current_definition": record["definition"],
                "proposed_definition": proposals[token],
                "family": record["family"],
                "domain": record["domain"],
                "current_bare_token_cosine": current_bare_token_cosine,
                "proposed_bare_token_cosine": proposed_bare_token_cosine,
                "bare_token_delta": bare_token_delta,
                "current_lexical_query_cosine": current_lexical_query_cosine,
                "proposed_lexical_query_cosine": proposed_lexical_query_cosine,
                "lexical_query_delta": lexical_query_delta,
                "token_delta": token_delta,
                "current_family_cosine": current_family_cosine,
                "proposed_family_cosine": proposed_family_cosine,
                "family_delta": family_delta,
                "current_family_rank": current_family_rank,
                "proposed_family_rank": proposed_family_rank,
                "model_signal": signal,
            }
        )

    rows.sort(
        key=lambda row: (
            {"regressed": 0, "mixed": 1, "inconclusive": 2, "improved": 3}[
                row["model_signal"]
            ],
            row["token_delta"] + row["family_delta"],
            row["token"],
        )
    )
    return {
        "format": "definition-proposal-embedding-audit-v1",
        "model": {
            "name": model_name,
            "revision": model_revision,
        },
        "policy": {
            "automatic_definition_replacement": False,
            "embedding_signal_is_not_lexical_truth": True,
            "current_family_is_diagnostic_not_ground_truth": True,
            "token_alignment_probes": ["bare_token", "lexical_query"],
            "lexical_query_template": "What is the meaning of the word TOKEN?",
            "human_acceptance_required": True,
        },
        "definitions": len(rows),
        "signal_counts": signal_counts,
        "review": rows,
    }


def audit_definition_proposals(
    dictionary_path: Path,
    proposals_path: Path,
    output_path: Path,
    *,
    model_name: str,
    model_revision: str,
    device: str | None = None,
    batch_size: int = 128,
) -> dict[str, Any]:
    dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
    proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
    embedder = _load_embedder(model_name, model_revision, device)
    result = audit_definition_proposals_data(
        dictionary,
        proposals,
        embedder=embedder,
        model_name=model_name,
        model_revision=model_revision,
        batch_size=batch_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _consensus(primary_signal: str, secondary_signal: str) -> str:
    if primary_signal == secondary_signal == "improved":
        return "supported_by_both"
    if primary_signal == secondary_signal == "regressed":
        return "regressed_by_both"
    if {primary_signal, secondary_signal} == {"improved", "regressed"}:
        return "mixed"
    if primary_signal == secondary_signal == "mixed":
        return "mixed"
    if "mixed" in {primary_signal, secondary_signal}:
        if "improved" in {primary_signal, secondary_signal}:
            return "partially_supported"
        if "regressed" in {primary_signal, secondary_signal}:
            return "needs_revision"
        return "inconclusive"
    if "regressed" in {primary_signal, secondary_signal}:
        return "needs_revision"
    if "improved" in {primary_signal, secondary_signal}:
        return "partially_supported"
    return "inconclusive"


def merge_definition_proposal_audits_data(
    primary: dict[str, Any], secondary: dict[str, Any]
) -> dict[str, Any]:
    primary_rows = {row["token"]: row for row in primary["review"]}
    secondary_rows = {row["token"]: row for row in secondary["review"]}
    if set(primary_rows) != set(secondary_rows):
        raise ValueError("proposal audits do not cover the same token set")
    rows = []
    consensus_counts: dict[str, int] = defaultdict(int)
    for token in sorted(primary_rows):
        first = primary_rows[token]
        second = secondary_rows[token]
        if first["proposed_definition"] != second["proposed_definition"]:
            raise ValueError(f"proposal audits disagree on definition for {token}")
        consensus = _consensus(first["model_signal"], second["model_signal"])
        consensus_counts[consensus] += 1
        rows.append(
            {
                "token": token,
                "consensus": consensus,
                "current_definition": first["current_definition"],
                "proposed_definition": first["proposed_definition"],
                "family": first["family"],
                "domain": first["domain"],
                "primary_model": primary["model"]["name"],
                "primary_signal": first["model_signal"],
                "primary_token_delta": first["token_delta"],
                "primary_bare_token_delta": first["bare_token_delta"],
                "primary_lexical_query_delta": first["lexical_query_delta"],
                "primary_family_delta": first["family_delta"],
                "primary_current_family_rank": first["current_family_rank"],
                "primary_proposed_family_rank": first["proposed_family_rank"],
                "secondary_model": secondary["model"]["name"],
                "secondary_signal": second["model_signal"],
                "secondary_token_delta": second["token_delta"],
                "secondary_bare_token_delta": second["bare_token_delta"],
                "secondary_lexical_query_delta": second["lexical_query_delta"],
                "secondary_family_delta": second["family_delta"],
                "secondary_current_family_rank": second["current_family_rank"],
                "secondary_proposed_family_rank": second["proposed_family_rank"],
                "reviewer_decision": "pending",
                "reviewer_notes": "",
            }
        )
    priority = {
        "regressed_by_both": 0,
        "needs_revision": 1,
        "mixed": 2,
        "inconclusive": 3,
        "partially_supported": 4,
        "supported_by_both": 5,
    }
    rows.sort(key=lambda row: (priority[row["consensus"]], row["token"]))
    return {
        "format": "definition-proposal-cross-model-review-v1",
        "policy": {
            "automatic_definition_replacement": False,
            "consensus_is_a_review_signal_not_acceptance": True,
            "human_acceptance_required": True,
        },
        "primary_model": primary["model"],
        "secondary_model": secondary["model"],
        "rows": len(rows),
        "consensus_counts": dict(sorted(consensus_counts.items())),
        "review": rows,
    }


def merge_definition_proposal_audits(
    primary_path: Path,
    secondary_path: Path,
    output_json: Path,
    output_csv: Path,
) -> dict[str, Any]:
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    secondary = json.loads(secondary_path.read_text(encoding="utf-8"))
    result = merge_definition_proposal_audits_data(primary, secondary)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROPOSAL_REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(result["review"])
    return result
