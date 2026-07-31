from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from .build import build_corpus
from .conversation import build_conversation_dataset
from .conversation_blueprint import build_conversation_blueprints
from .conversation_mine import build_conversation_mine, fetch_conversation_sources
from .conversation_surface import (
    build_conversation_surface,
    build_conversation_surface_pilot,
)
from .forge import write_forged_dataset
from .instruct import build_instruction_dataset, tokenize_instruction_dataset
from .package import (
    package_for_hugging_face,
    package_instructions_for_hugging_face,
)
from .scenario_forge import build_scenario_forge
from .tokenize import tokenize_documents


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

    build_conversation = commands.add_parser("build-conversation")
    build_conversation.add_argument("--output", type=Path, required=True)
    build_conversation.add_argument("--examples", type=int, default=2_000)
    build_conversation.add_argument("--seed", type=int, default=42)
    build_conversation.add_argument("--validation-percent", type=int, default=5)

    fetch_conversation_mine = commands.add_parser("fetch-conversation-mine")
    fetch_conversation_mine.add_argument("--registry", type=Path, required=True)
    fetch_conversation_mine.add_argument("--raw", type=Path, required=True)

    conversation_mine = commands.add_parser("build-conversation-mine")
    conversation_mine.add_argument("--registry", type=Path, required=True)
    conversation_mine.add_argument("--raw", type=Path, required=True)
    conversation_mine.add_argument("--output", type=Path, required=True)
    conversation_mine.add_argument("--max-rows-per-source", type=int)

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
    tokenize_instruct.add_argument("--tokenizer", type=Path, required=True)
    tokenize_instruct.add_argument("--output", type=Path, required=True)

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
    elif args.command == "build-conversation":
        result = build_conversation_dataset(
            args.output,
            examples=args.examples,
            seed=args.seed,
            validation_percent=args.validation_percent,
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
