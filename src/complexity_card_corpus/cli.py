from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from .build import build_corpus
from .conversation_blueprint import build_conversation_blueprints
from .conversational import build_casual_conversation_surface
from .contract_embedding_audit import audit_contract_dataset_with_embeddings
from .dictionary_review import write_dictionary_review
from .definition_acceptance import accept_definition_proposals
from .embedding_guidance import build_embedding_guidance
from .conversation_mine import build_conversation_mine, fetch_conversation_sources
from .surfaces import (
    build_conversation_surface,
    build_conversation_surface_pilot,
)
from .forge import write_forged_dataset
from .sft import build_instruction_dataset, tokenize_instruction_dataset
from .vocabulary import (
    audit_source_overlap,
    build_lexical_mine,
    build_vocabulary_placement,
    fetch_lexical_sources,
)
from .package import (
    package_for_hugging_face,
    package_instructions_for_hugging_face,
    package_sft_views_for_hugging_face,
)
from .posttrain import audit_human_review, build_post_training_corpus
from .proposal_embedding_review import (
    audit_definition_proposals,
    merge_definition_proposal_audits,
)
from .quality_audit import audit_dataset_quality
from .semantic_audit import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    audit_dataset_semantic_diversity,
)
from .scenarios import audit_scenario_tanks, build_scenario_forge
from .tokenize import tokenize_documents
from .vocabulary_gap import build_vocabulary_gap
from .vocabulary_wordnet_audit import audit_vocabulary_with_wordnet


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="card-corpus",
        description="Build linked knowledge cards into Parquet and o200k token shards.",
    )
    commands = root.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--max-path-depth", type=int, default=3)
    build.add_argument("--max-paths-per-card", type=int, default=4)

    forge = commands.add_parser("forge")
    forge.add_argument("--blueprint", type=Path, required=True)
    forge.add_argument("--output", type=Path, required=True)
    forge.add_argument("--force", action="store_true")

    scenario_forge = commands.add_parser("build-scenario-forge")
    scenario_forge.add_argument("--registry", type=Path, required=True)
    scenario_forge.add_argument("--output", type=Path, required=True)
    scenario_forge.add_argument(
        "--target-scenarios",
        type=int,
        required=True,
        help=(
            "requested total allocated dynamically by family weight and "
            "semantic capacity"
        ),
    )

    tank_audit = commands.add_parser("audit-scenario-tanks")
    tank_audit.add_argument("--registry", type=Path, required=True)

    post_training = commands.add_parser("build-post-training")
    post_training.add_argument("--scenarios", type=Path, required=True)
    post_training.add_argument("--output", type=Path, required=True)
    post_training.add_argument("--variants-per-scenario", type=int, default=8)
    post_training.add_argument(
        "--target-rows",
        type=int,
        default=0,
        help="optional scale objective; 0 reports realized rows without a target",
    )
    post_training.add_argument(
        "--max-examples-per-family",
        type=int,
        default=0,
        help=(
            "optional recovery cap applied to complete instruct/chat scenario "
            "bundles after exact deduplication; 0 preserves every valid row"
        ),
    )
    post_training.add_argument(
        "--review-scenarios",
        "--review-rows",
        dest="review_scenarios",
        type=int,
        default=140,
        help=(
            "number of unique source scenarios to review; one instruct and one "
            "chat row are emitted for each scenario"
        ),
    )
    post_training.add_argument("--seed", type=int, default=42)
    post_training.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="worker processes used for deterministic conversation rendering",
    )
    post_training.add_argument(
        "--vocabulary-placement",
        type=Path,
        help="statistical token-to-family placement produced by place-vocabulary",
    )

    post_training_review = commands.add_parser("audit-post-training-review")
    post_training_review.add_argument("--review", type=Path, required=True)

    tokenize = commands.add_parser("tokenize")
    tokenize.add_argument("--documents", type=Path, required=True)
    tokenize.add_argument("--tokenizer", type=Path, required=True)
    tokenize.add_argument("--output", type=Path, required=True)

    package = commands.add_parser("package-hf")
    package.add_argument("--corpus", type=Path, required=True)
    package.add_argument("--tokenized", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)

    build_instruct = commands.add_parser("build-instruct")
    build_instruct.add_argument("--corpus", type=Path, required=True)
    build_instruct.add_argument("--output", type=Path, required=True)
    build_instruct.add_argument("--max-attributes-per-card", type=int, default=2)
    build_instruct.add_argument("--max-relations-per-card", type=int, default=2)
    build_instruct.add_argument("--max-paths-per-card", type=int, default=1)

    fetch_conversation_mine = commands.add_parser("fetch-conversation-mine")
    fetch_conversation_mine.add_argument("--registry", type=Path, required=True)
    fetch_conversation_mine.add_argument("--raw", type=Path, required=True)

    conversation_mine = commands.add_parser("build-conversation-mine")
    conversation_mine.add_argument("--registry", type=Path, required=True)
    conversation_mine.add_argument("--raw", type=Path, required=True)
    conversation_mine.add_argument("--output", type=Path, required=True)
    conversation_mine.add_argument("--max-rows-per-source", type=int)

    lexical_mine = commands.add_parser("build-lexical-mine")
    lexical_mine.add_argument("--registry", type=Path, required=True)
    lexical_mine.add_argument("--raw", type=Path, required=True)
    lexical_mine.add_argument("--output", type=Path, required=True)
    lexical_mine.add_argument("--min-count", type=int, default=8)
    lexical_mine.add_argument("--max-capitalized-ratio", type=float, default=0.65)
    lexical_mine.add_argument("--delete-raw", action="store_true")
    lexical_mine.add_argument(
        "--scenarios",
        type=Path,
        help="optionally compare mined eight-token structures with Scenario Forge",
    )

    fetch_lexical = commands.add_parser("fetch-lexical-mine")
    fetch_lexical.add_argument("--registry", type=Path, required=True)
    fetch_lexical.add_argument("--raw", type=Path, required=True)

    source_overlap = commands.add_parser("audit-source-overlap")
    source_overlap.add_argument("--registry", type=Path, required=True)
    source_overlap.add_argument("--raw", type=Path, required=True)
    source_overlap.add_argument("--scenarios", type=Path, required=True)
    source_overlap.add_argument("--window-tokens", type=int, default=8)

    vocabulary_gap = commands.add_parser("build-vocabulary-gap")
    vocabulary_gap.add_argument("--lexicon", type=Path, required=True)
    vocabulary_gap.add_argument("--conversations", type=Path, required=True)
    vocabulary_gap.add_argument("--output", type=Path, required=True)
    vocabulary_gap.add_argument("--min-sources", type=int, default=2)
    vocabulary_gap.add_argument("--min-occurrences-per-source", type=int, default=20)
    vocabulary_gap.add_argument("--max-candidates", type=int, default=5_000)

    vocabulary_placement = commands.add_parser("place-vocabulary")
    vocabulary_placement.add_argument("--review", type=Path, required=True)
    vocabulary_placement.add_argument("--lexicon", type=Path, required=True)
    vocabulary_placement.add_argument("--registry", type=Path, required=True)
    vocabulary_placement.add_argument("--raw", type=Path, required=True)
    vocabulary_placement.add_argument("--scenarios", type=Path, required=True)
    vocabulary_placement.add_argument("--output", type=Path, required=True)
    vocabulary_placement.add_argument("--window-tokens", type=int, default=16)
    vocabulary_placement.add_argument("--accepted-definitions", type=Path)

    vocabulary_wordnet = commands.add_parser("audit-vocabulary-wordnet")
    vocabulary_wordnet.add_argument("--dictionary", type=Path, required=True)
    vocabulary_wordnet.add_argument("--output", type=Path, required=True)
    vocabulary_wordnet.add_argument("--lexicon", default="oewn:2025+")

    conversation_blueprints = commands.add_parser("build-conversation-blueprints")
    conversation_blueprints.add_argument("--mine", type=Path, required=True)
    conversation_blueprints.add_argument("--output", type=Path, required=True)
    conversation_blueprints.add_argument("--target-per-kind", type=int)
    conversation_blueprints.add_argument("--seed", type=int, default=42)
    conversation_blueprints.add_argument("--validation-percent", type=int, default=5)

    for command, default_examples in (
        ("build-conversation-surface", 10_000),
        ("build-conversation-surface-pilot", 512),
    ):
        conversation_surface = commands.add_parser(command)
        conversation_surface.add_argument("--blueprints", type=Path, required=True)
        conversation_surface.add_argument("--scenarios", type=Path, required=True)
        conversation_surface.add_argument("--output", type=Path, required=True)
        conversation_surface.add_argument(
            "--examples", type=int, default=default_examples
        )
        conversation_surface.add_argument("--seed", type=int, default=42)
        conversation_surface.add_argument("--validation-percent", type=int, default=5)

    casual_conversation = commands.add_parser("build-casual-conversation")
    casual_conversation.add_argument("--registry", type=Path, required=True)
    casual_conversation.add_argument("--output", type=Path, required=True)
    casual_conversation.add_argument(
        "--examples",
        type=int,
        help="rows to render; omitted means one row per semantic topic/context pair",
    )
    casual_conversation.add_argument("--seed", type=int, default=42)
    casual_conversation.add_argument("--validation-percent", type=int, default=5)

    tokenize_instruct = commands.add_parser("tokenize-instruct")
    tokenize_instruct.add_argument("--instructions", type=Path, required=True)
    tokenize_instruct.add_argument(
        "--supplement",
        type=Path,
        action="append",
        default=[],
        help=("additional original instruction/conversation Parquet; may be repeated"),
    )
    tokenize_instruct.add_argument(
        "--casual-registry",
        type=Path,
        default=Path("data/conversation/original/casual-conversation-decks-v1.json"),
        help="original casual card registry built and included automatically",
    )
    tokenize_instruct.add_argument(
        "--casual-output",
        type=Path,
        default=Path("build/casual-conversation-v16"),
        help="deterministic V16 casual build directory",
    )
    tokenize_instruct.add_argument(
        "--without-casual-conversation",
        action="store_true",
        help="opt out only for isolated tests or diagnostics",
    )
    tokenize_instruct.add_argument("--tokenizer", type=Path, required=True)
    tokenize_instruct.add_argument("--output", type=Path, required=True)
    tokenize_instruct.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="worker processes used for projection and statistical auditing",
    )
    tokenize_instruct.add_argument(
        "--max-examples-per-family",
        type=int,
        default=0,
        help="optional recovery cap; 0 preserves every non-duplicate row",
    )
    tokenize_instruct.add_argument(
        "--max-per-structure",
        type=int,
        default=0,
        help="optional normalized-structure cap; 0 preserves every structure",
    )
    tokenize_instruct.add_argument(
        "--max-domain-share",
        type=float,
        default=0,
        help="optional destructive domain-balance cap; 0 is audit-only",
    )
    tokenize_instruct.add_argument(
        "--max-response-card-hand-share",
        type=float,
        default=0,
        help="optional destructive card-hand cap; 0 is audit-only",
    )
    tokenize_instruct.add_argument(
        "--target-training-examples",
        type=int,
        default=0,
        help="optional release target; 0 reports realized scale without a target",
    )
    tokenize_instruct.add_argument(
        "--target-supervised-tokens",
        type=int,
        default=0,
        help="optional release target; 0 reports realized scale without a target",
    )
    tokenize_instruct.add_argument(
        "--heldout-evaluation",
        type=Path,
        help=(
            "separately authored evaluation JSON; generated validation rows are "
            "excluded when this option is provided"
        ),
    )

    package_instruct = commands.add_parser("package-instruct-hf")
    package_instruct.add_argument("--instructions", type=Path, required=True)
    package_instruct.add_argument("--tokenized", type=Path, required=True)
    package_instruct.add_argument("--output", type=Path, required=True)

    package_sft_views = commands.add_parser("package-sft-views-hf")
    package_sft_views.add_argument("--projected", type=Path, required=True)
    package_sft_views.add_argument("--output", type=Path, required=True)
    package_sft_views.add_argument("--release-slug", required=True)
    package_sft_views.add_argument("--max-rows-per-shard", type=int, default=50_000)
    package_sft_views.add_argument("--row-group-size", type=int, default=5_000)

    sklearn_audit = commands.add_parser("audit-sklearn")
    sklearn_audit.add_argument("--conversations", type=Path, required=True)
    sklearn_audit.add_argument("--output", type=Path, required=True)
    sklearn_audit.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="0 selects an adaptive value from the corpus size",
    )
    sklearn_audit.add_argument("--near-duplicate-threshold", type=float, default=0.95)
    sklearn_audit.add_argument(
        "--max-features",
        type=int,
        default=0,
        help="0 selects an adaptive TF-IDF vocabulary size",
    )
    sklearn_audit.add_argument(
        "--clusters",
        type=int,
        default=0,
        help="0 selects an adaptive MiniBatchKMeans cluster count",
    )
    sklearn_audit.add_argument(
        "--workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )

    embedding_audit = commands.add_parser("audit-embeddings")
    embedding_audit.add_argument("--conversations", type=Path, required=True)
    embedding_audit.add_argument("--output", type=Path, required=True)
    embedding_audit.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    embedding_audit.add_argument("--revision", default=DEFAULT_EMBEDDING_REVISION)
    embedding_audit.add_argument("--device")
    embedding_audit.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="0 selects the same adaptive deterministic sample as audit-sklearn",
    )
    embedding_audit.add_argument(
        "--clusters",
        type=int,
        default=0,
        help="0 selects an adaptive MiniBatchKMeans cluster count",
    )
    embedding_audit.add_argument(
        "--semantic-duplicate-threshold", type=float, default=0.98
    )
    embedding_audit.add_argument("--batch-size", type=int, default=128)
    embedding_audit.add_argument(
        "--workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )

    contract_embedding_audit = commands.add_parser("audit-contract-embeddings")
    contract_embedding_audit.add_argument("--conversations", type=Path, required=True)
    contract_embedding_audit.add_argument("--output", type=Path, required=True)
    contract_embedding_audit.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    contract_embedding_audit.add_argument(
        "--revision", default=DEFAULT_EMBEDDING_REVISION
    )
    contract_embedding_audit.add_argument("--device")
    contract_embedding_audit.add_argument("--batch-size", type=int, default=128)

    embedding_guidance = commands.add_parser("build-embedding-guidance")
    embedding_guidance.add_argument("--dictionary", type=Path, required=True)
    embedding_guidance.add_argument("--semantic-audit", type=Path, required=True)
    embedding_guidance.add_argument("--output", type=Path, required=True)
    embedding_guidance.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    embedding_guidance.add_argument("--revision", default=DEFAULT_EMBEDDING_REVISION)
    embedding_guidance.add_argument("--device")
    embedding_guidance.add_argument("--batch-size", type=int, default=128)
    embedding_guidance.add_argument("--alternatives-per-token", type=int, default=5)
    embedding_guidance.add_argument(
        "--workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )

    dictionary_review = commands.add_parser("build-dictionary-review")
    dictionary_review.add_argument("--primary-guidance", type=Path, required=True)
    dictionary_review.add_argument("--secondary-guidance", type=Path, required=True)
    dictionary_review.add_argument("--output-json", type=Path, required=True)
    dictionary_review.add_argument("--output-csv", type=Path, required=True)
    dictionary_review.add_argument("--proposals", type=Path)

    proposal_audit = commands.add_parser("audit-definition-proposals")
    proposal_audit.add_argument("--dictionary", type=Path, required=True)
    proposal_audit.add_argument("--proposals", type=Path, required=True)
    proposal_audit.add_argument("--output", type=Path, required=True)
    proposal_audit.add_argument("--model", required=True)
    proposal_audit.add_argument("--revision", required=True)
    proposal_audit.add_argument("--device")
    proposal_audit.add_argument("--batch-size", type=int, default=128)

    proposal_merge = commands.add_parser("merge-definition-proposal-audits")
    proposal_merge.add_argument("--primary", type=Path, required=True)
    proposal_merge.add_argument("--secondary", type=Path, required=True)
    proposal_merge.add_argument("--output-json", type=Path, required=True)
    proposal_merge.add_argument("--output-csv", type=Path, required=True)

    definition_acceptance = commands.add_parser("accept-definition-proposals")
    definition_acceptance.add_argument("--dictionary", type=Path, required=True)
    definition_acceptance.add_argument("--proposals", type=Path, required=True)
    definition_acceptance.add_argument("--review", type=Path, required=True)
    definition_acceptance.add_argument("--output-dictionary", type=Path, required=True)
    definition_acceptance.add_argument("--output-review", type=Path, required=True)
    definition_acceptance.add_argument("--placement", type=Path)
    definition_acceptance.add_argument("--output-placement", type=Path)
    definition_acceptance.add_argument("--accept-all", action="store_true")

    inspect = commands.add_parser("inspect")
    inspect.add_argument("--output", type=Path, required=True)
    return root


def _inspect(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    documents = pq.read_table(output / "documents.parquet")
    preview = [
        {
            "document_id": row["document_id"],
            "split": row["split"],
            "text": row["text"][:180],
        }
        for row in documents.slice(0, min(3, len(documents))).to_pylist()
    ]
    return {"counts": manifest["counts"], "preview": preview}


def main() -> None:
    args = parser().parse_args()
    if args.command == "build":
        result = build_corpus(
            args.source,
            args.output,
            max_path_depth=args.max_path_depth,
            max_paths_per_card=args.max_paths_per_card,
        )
        print(json.dumps(result["counts"], indent=2, sort_keys=True))
    elif args.command == "forge":
        result = write_forged_dataset(
            args.blueprint,
            args.output,
            force=args.force,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "build-scenario-forge":
        result = build_scenario_forge(
            args.registry,
            args.output,
            target_scenarios=args.target_scenarios,
        )
        print(
            json.dumps(
                {"counts": result["counts"], "audit": result["audit"]},
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "audit-scenario-tanks":
        print(json.dumps(audit_scenario_tanks(args.registry), indent=2, sort_keys=True))
    elif args.command == "build-post-training":
        result = build_post_training_corpus(
            args.scenarios,
            args.output,
            variants_per_scenario=args.variants_per_scenario,
            review_scenarios=args.review_scenarios,
            seed=args.seed,
            vocabulary_placement_path=args.vocabulary_placement,
            workers=args.workers,
            target_rows=(args.target_rows or None),
            max_examples_per_family=(args.max_examples_per_family or None),
        )
        print(
            json.dumps(
                {
                    "audit": result["audit"],
                    "human_review": result["human_review"],
                    "training_ready": result["training_ready"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "audit-post-training-review":
        print(json.dumps(audit_human_review(args.review), indent=2, sort_keys=True))
    elif args.command == "tokenize":
        result = tokenize_documents(args.documents, args.tokenizer, args.output)
        print(
            json.dumps(
                {
                    "documents": result["total_documents"],
                    "tokens": result["total_tokens"],
                    "partitions": sorted(result["partitions"]),
                },
                indent=2,
            )
        )
    elif args.command == "package-hf":
        result = package_for_hugging_face(
            args.corpus,
            args.tokenized,
            args.output,
        )
        print(
            json.dumps(
                {
                    "files": len(result["files"]),
                    "bytes": sum(item["bytes"] for item in result["files"].values()),
                    "output": str(args.output.resolve()),
                },
                indent=2,
            )
        )
    elif args.command == "build-instruct":
        result = build_instruction_dataset(
            args.corpus,
            args.output,
            max_attributes_per_card=args.max_attributes_per_card,
            max_relations_per_card=args.max_relations_per_card,
            max_paths_per_card=args.max_paths_per_card,
        )
        print(json.dumps(result["counts"], indent=2, sort_keys=True))
    elif args.command == "fetch-conversation-mine":
        result = fetch_conversation_sources(args.registry, args.raw)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "build-conversation-mine":
        result = build_conversation_mine(
            args.registry,
            args.raw,
            args.output,
            max_rows_per_source=args.max_rows_per_source,
        )
        print(
            json.dumps(
                {"counts": result["counts"], "audit": result["audit"]},
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "build-lexical-mine":
        result = build_lexical_mine(
            args.registry,
            args.raw,
            args.output,
            min_count=args.min_count,
            max_capitalized_ratio=args.max_capitalized_ratio,
            delete_raw=args.delete_raw,
            scenarios_path=args.scenarios,
        )
        print(json.dumps(result["audit"], indent=2, sort_keys=True))
    elif args.command == "fetch-lexical-mine":
        result = fetch_lexical_sources(args.registry, args.raw)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "audit-source-overlap":
        result = audit_source_overlap(
            args.registry,
            args.raw,
            args.scenarios,
            window_tokens=args.window_tokens,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "build-vocabulary-gap":
        result = build_vocabulary_gap(
            args.lexicon,
            args.conversations,
            args.output,
            min_sources=args.min_sources,
            min_occurrences_per_source=args.min_occurrences_per_source,
            max_candidates=args.max_candidates,
        )
        print(json.dumps(result["audit"], indent=2, sort_keys=True))
    elif args.command == "place-vocabulary":
        result = build_vocabulary_placement(
            args.review,
            args.lexicon,
            args.registry,
            args.raw,
            args.scenarios,
            args.output,
            window_tokens=args.window_tokens,
            accepted_definitions_path=args.accepted_definitions,
        )
        print(json.dumps(result["audit"], indent=2, sort_keys=True))
    elif args.command == "audit-vocabulary-wordnet":
        result = audit_vocabulary_with_wordnet(
            args.dictionary,
            args.output,
            lexicon=args.lexicon,
        )
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
    elif args.command == "build-conversation-blueprints":
        result = build_conversation_blueprints(
            args.mine,
            args.output,
            target_per_kind=args.target_per_kind,
            seed=args.seed,
            validation_percent=args.validation_percent,
        )
        print(
            json.dumps(
                {"counts": result["counts"], "audit": result["audit"]},
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command in {
        "build-conversation-surface",
        "build-conversation-surface-pilot",
    }:
        if args.command == "build-conversation-surface":
            result = build_conversation_surface(
                args.blueprints,
                args.scenarios,
                args.output,
                examples=args.examples,
                seed=args.seed,
                validation_percent=args.validation_percent,
            )
        else:
            result = build_conversation_surface_pilot(
                args.blueprints,
                args.scenarios,
                args.output,
                pilot_size=args.examples,
                seed=args.seed,
                validation_percent=args.validation_percent,
            )
        print(
            json.dumps(
                {"counts": result["counts"], "audit": result["audit"]},
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "build-casual-conversation":
        result = build_casual_conversation_surface(
            args.registry,
            args.output,
            examples=args.examples,
            seed=args.seed,
            validation_percent=args.validation_percent,
        )
        print(
            json.dumps(
                {"counts": result["counts"], "audit": result["audit"]},
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "tokenize-instruct":
        supplements = list(args.supplement)
        require_casual_conversation = not args.without_casual_conversation
        if require_casual_conversation:
            build_casual_conversation_surface(
                args.casual_registry,
                args.casual_output,
                seed=42,
                validation_percent=5,
            )
            casual_path = args.casual_output / "conversations.parquet"
            if casual_path.resolve() not in {path.resolve() for path in supplements}:
                supplements.append(casual_path)
        result = tokenize_instruction_dataset(
            args.instructions,
            args.tokenizer,
            args.output,
            heldout_evaluation_path=args.heldout_evaluation,
            supplementary_instruction_paths=supplements,
            workers=args.workers,
            max_examples_per_family=(args.max_examples_per_family or None),
            max_per_structure=(args.max_per_structure or None),
            max_domain_share=(args.max_domain_share or None),
            max_response_card_hand_share=(args.max_response_card_hand_share or None),
            target_training_examples=(args.target_training_examples or None),
            target_supervised_tokens=(args.target_supervised_tokens or None),
            require_casual_conversation=require_casual_conversation,
        )
        print(
            json.dumps(
                {
                    "examples": result["total_examples"],
                    "tokens": result["total_tokens"],
                    "supervised_tokens": result["total_supervised_tokens"],
                    "partitions": sorted(result["partitions"]),
                },
                indent=2,
            )
        )
    elif args.command == "package-instruct-hf":
        result = package_instructions_for_hugging_face(
            args.instructions,
            args.tokenized,
            args.output,
        )
        print(
            json.dumps(
                {
                    "files": len(result["files"]),
                    "bytes": sum(item["bytes"] for item in result["files"].values()),
                    "output": str(args.output.resolve()),
                },
                indent=2,
            )
        )
    elif args.command == "package-sft-views-hf":
        result = package_sft_views_for_hugging_face(
            args.projected,
            args.output,
            release_slug=args.release_slug,
            max_rows_per_shard=args.max_rows_per_shard,
            row_group_size=args.row_group_size,
        )
        print(
            json.dumps(
                {
                    "examples": result["examples"],
                    "parquet_shards": len(result["parquet_shards"]),
                    "views": result["views"],
                    "output": str(args.output.resolve()),
                },
                indent=2,
            )
        )
    elif args.command == "audit-sklearn":
        result = audit_dataset_quality(
            args.conversations,
            args.output,
            sample_size=args.sample_size or None,
            near_duplicate_threshold=args.near_duplicate_threshold,
            max_features=args.max_features or None,
            cluster_count=args.clusters or None,
            workers=args.workers,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "audit-embeddings":
        result = audit_dataset_semantic_diversity(
            args.conversations,
            args.output,
            model_name=args.model,
            model_revision=args.revision,
            device=args.device,
            sample_size=args.sample_size or None,
            cluster_count=args.clusters or None,
            semantic_duplicate_threshold=args.semantic_duplicate_threshold,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "audit-contract-embeddings":
        result = audit_contract_dataset_with_embeddings(
            args.conversations,
            args.output,
            model_name=args.model,
            model_revision=args.revision,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "build-embedding-guidance":
        result = build_embedding_guidance(
            args.dictionary,
            args.semantic_audit,
            args.output,
            model_name=args.model,
            model_revision=args.revision,
            device=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
            alternatives_per_token=args.alternatives_per_token,
        )
        print(
            json.dumps(
                {
                    "dictionary_cards": result["dictionary"]["cards"],
                    "families": len(result["family_priorities"]),
                    "semantic_decks": sum(
                        len(decks) for decks in result["semantic_decks"].values()
                    ),
                    "output": str(args.output.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "build-dictionary-review":
        result = write_dictionary_review(
            args.primary_guidance,
            args.secondary_guidance,
            args.output_json,
            args.output_csv,
            args.proposals,
        )
        print(
            json.dumps(
                {
                    "rows": result["rows"],
                    "status_counts": result["status_counts"],
                    "proposed_definitions": result["proposed_definitions"],
                    "output_json": str(args.output_json.resolve()),
                    "output_csv": str(args.output_csv.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "audit-definition-proposals":
        result = audit_definition_proposals(
            args.dictionary,
            args.proposals,
            args.output,
            model_name=args.model,
            model_revision=args.revision,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(
            json.dumps(
                {
                    "definitions": result["definitions"],
                    "signal_counts": result["signal_counts"],
                    "output": str(args.output.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "merge-definition-proposal-audits":
        result = merge_definition_proposal_audits(
            args.primary,
            args.secondary,
            args.output_json,
            args.output_csv,
        )
        print(
            json.dumps(
                {
                    "rows": result["rows"],
                    "consensus_counts": result["consensus_counts"],
                    "output_json": str(args.output_json.resolve()),
                    "output_csv": str(args.output_csv.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "accept-definition-proposals":
        result = accept_definition_proposals(
            args.dictionary,
            args.proposals,
            args.review,
            args.output_dictionary,
            args.output_review,
            accept_all=args.accept_all,
            placement_path=args.placement,
            output_placement_path=args.output_placement,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "inspect":
        print(json.dumps(_inspect(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
