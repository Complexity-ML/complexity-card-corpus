from __future__ import annotations

import argparse
import json
from pathlib import Path

from .v2 import audit_v2_release, build_v2_release, tokenize_v2_release


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="card-corpus",
        description="Build, audit, and tokenize Complexity Card Corpus V2.",
    )
    commands = root.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build",
        help="phase 1: render V2 only; never run audits or tests",
    )
    build.add_argument("--output", type=Path, required=True)

    audit = commands.add_parser(
        "audit",
        help="phase 2: run the complete V2 release gates on an existing build",
    )
    audit.add_argument("--artifact", type=Path, required=True)
    audit.add_argument(
        "--tokenizer",
        type=Path,
        help="optionally include the tokenizer/loss-mask release gate",
    )

    tokenize = commands.add_parser(
        "tokenize",
        help="phase 3: create SFT shards from a separately audited green build",
    )
    tokenize.add_argument("--artifact", type=Path, required=True)
    tokenize.add_argument("--tokenizer", type=Path, required=True)
    tokenize.add_argument("--output", type=Path, required=True)

    inspect = commands.add_parser("inspect")
    inspect.add_argument("--artifact", type=Path, required=True)
    return root


def _summary(manifest: dict) -> dict:
    return {
        key: manifest[key]
        for key in (
            "format",
            "phase_status",
            "quality_status",
            "examples",
            "splits",
            "total_examples",
            "total_tokens",
            "total_supervised_tokens",
        )
        if key in manifest
    }


def main() -> None:
    args = parser().parse_args()
    if args.command == "build":
        result = build_v2_release(args.output)
    elif args.command == "audit":
        result = audit_v2_release(args.artifact, tokenizer_root=args.tokenizer)
    elif args.command == "tokenize":
        result = tokenize_v2_release(args.artifact, args.tokenizer, args.output)
    elif args.command == "inspect":
        result = json.loads((args.artifact / "manifest.json").read_text())
    else:  # pragma: no cover
        raise ValueError(args.command)
    print(json.dumps(_summary(result) or result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
