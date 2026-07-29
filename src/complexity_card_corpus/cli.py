from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from .build import build_corpus
from .oasst1 import import_oasst1
from .package import package_for_hugging_face
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
    elif args.command == "inspect":
        print(json.dumps(_inspect(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
