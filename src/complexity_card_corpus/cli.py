from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from .build import build_corpus
from .conversation_blueprint import build_conversation_blueprints
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
)
from .posttrain import audit_human_review, build_post_training_corpus
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

    tank_audit = commands.add_parser("audit-scenario-tanks")
    tank_audit.add_argument("--registry", type=Path, required=True)

    post_training = commands.add_parser("build-post-training")
    post_training.add_argument("--scenarios", type=Path, required=True)
    post_training.add_argument("--output", type=Path, required=True)
    post_training.add_argument("--variants-per-scenario", type=int, default=4)
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

    tokenize_instruct = commands.add_parser("tokenize-instruct")
    tokenize_instruct.add_argument("--instructions", type=Path, required=True)
    tokenize_instruct.add_argument(
        "--supplement",
        type=Path,
        action="append",
        default=[],
        help=(
            "additional original instruction/conversation Parquet; may be repeated"
        ),
    )
    tokenize_instruct.add_argument("--tokenizer", type=Path, required=True)
    tokenize_instruct.add_argument("--output", type=Path, required=True)
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
        result = build_scenario_forge(args.registry, args.output)
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
    elif args.command == "tokenize-instruct":
        result = tokenize_instruction_dataset(
            args.instructions,
            args.tokenizer,
            args.output,
            heldout_evaluation_path=args.heldout_evaluation,
            supplementary_instruction_paths=args.supplement,
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
    elif args.command == "inspect":
        print(json.dumps(_inspect(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
