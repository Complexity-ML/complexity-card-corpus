from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from .build import build_corpus
from .mosaic import build_mosaic, package_mosaic_for_hugging_face
from .mosaic_stream import (
    build_mosaic_shards,
    strip_atlas_from_mosaic,
    tokenize_mosaic_shards,
)
from .oasst1 import import_oasst1
from .package import package_alignment_for_hugging_face, package_for_hugging_face
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

    tokenize = commands.add_parser("tokenize")
    tokenize.add_argument("--documents", type=Path, required=True)
    tokenize.add_argument("--tokenizer", type=Path, required=True)
    tokenize.add_argument("--output", type=Path, required=True)

    package = commands.add_parser("package-hf")
    package.add_argument("--corpus", type=Path, required=True)
    package.add_argument("--tokenized", type=Path, required=True)
    package.add_argument("--alignment", type=Path)
    package.add_argument("--output", type=Path, required=True)

    package_posttrain = commands.add_parser("package-posttrain-hf")
    package_posttrain.add_argument("--alignment", type=Path, required=True)
    package_posttrain.add_argument("--output", type=Path, required=True)

    mosaic = commands.add_parser("build-mosaic")
    mosaic.add_argument("--registry", type=Path, required=True)
    mosaic.add_argument("--raw", type=Path, required=True)
    mosaic.add_argument("--output", type=Path, required=True)
    mosaic.add_argument("--max-rows-per-source", type=int)
    mosaic.add_argument("--validation-per-mille", type=int, default=5)
    mosaic.add_argument("--workers", type=int, default=4)

    package_mosaic = commands.add_parser("package-mosaic-hf")
    package_mosaic.add_argument("--mosaic", type=Path, required=True)
    package_mosaic.add_argument("--tokenized", type=Path, required=True)
    package_mosaic.add_argument("--output", type=Path, required=True)

    mosaic_shards = commands.add_parser("build-mosaic-shards")
    mosaic_shards.add_argument("--registry", type=Path, required=True)
    mosaic_shards.add_argument("--raw", type=Path, required=True)
    mosaic_shards.add_argument("--output", type=Path, required=True)
    mosaic_shards.add_argument("--validation-per-mille", type=int, default=5)
    mosaic_shards.add_argument("--workers", type=int, default=4)
    mosaic_shards.add_argument("--batch-size", type=int, default=8192)

    tokenize_mosaic = commands.add_parser("tokenize-mosaic-shards")
    tokenize_mosaic.add_argument("--corpus", type=Path, required=True)
    tokenize_mosaic.add_argument("--tokenizer", type=Path, required=True)
    tokenize_mosaic.add_argument(
        "--target-train-tokens",
        type=int,
        default=4_000_000_000,
    )
    tokenize_mosaic.add_argument(
        "--target-eval-tokens",
        type=int,
        default=20_000_000,
    )
    tokenize_mosaic.add_argument("--workers", type=int, default=8)
    tokenize_mosaic.add_argument("--batch-size", type=int, default=256)

    strip_mosaic = commands.add_parser("strip-atlas-mosaic")
    strip_mosaic.add_argument("--corpus", type=Path, required=True)

    oasst1 = commands.add_parser("import-oasst1")
    oasst1.add_argument("--raw", type=Path, required=True)
    oasst1.add_argument("--output", type=Path, required=True)
    oasst1.add_argument("--offline", action="store_true")
    oasst1.add_argument("--max-characters", type=int, default=16_000)
    oasst1.add_argument("--max-messages", type=int, default=10)

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
            alignment_root=args.alignment,
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
    elif args.command == "import-oasst1":
        result = import_oasst1(
            args.raw,
            args.output,
            download=not args.offline,
            max_characters=args.max_characters,
            max_messages=args.max_messages,
        )
        print(json.dumps(result["counts"], indent=2, sort_keys=True))
    elif args.command == "package-posttrain-hf":
        result = package_alignment_for_hugging_face(
            args.alignment,
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
    elif args.command == "build-mosaic":
        result = build_mosaic(
            args.registry,
            args.raw,
            args.output,
            max_rows_per_source=args.max_rows_per_source,
            validation_per_mille=args.validation_per_mille,
            workers=args.workers,
        )
        print(json.dumps(result["counts"], indent=2, sort_keys=True))
    elif args.command == "package-mosaic-hf":
        result = package_mosaic_for_hugging_face(
            args.mosaic,
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
    elif args.command == "build-mosaic-shards":
        result = build_mosaic_shards(
            args.registry,
            args.raw,
            args.output,
            validation_per_mille=args.validation_per_mille,
            workers=args.workers,
            batch_size=args.batch_size,
        )
        print(json.dumps(result["counts"], indent=2, sort_keys=True))
    elif args.command == "tokenize-mosaic-shards":
        result = tokenize_mosaic_shards(
            args.corpus,
            args.tokenizer,
            target_train_tokens=args.target_train_tokens,
            target_eval_tokens=args.target_eval_tokens,
            workers=args.workers,
            batch_size=args.batch_size,
        )
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
    elif args.command == "strip-atlas-mosaic":
        result = strip_atlas_from_mosaic(args.corpus)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "inspect":
        print(json.dumps(_inspect(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
