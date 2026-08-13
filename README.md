# Complexity Card Corpus V2

Card Corpus V2 is an authored English SFT dataset generator built from
role-separated decks, compatible subcards, and `VariableBy2D` reservoirs.
V1 has been removed. The repository now has one generator, one release contract,
and one CLI.

## Architecture

```text
src/complexity_card_corpus/
├── cli.py                 # build, audit, tokenize, inspect
├── variable_by/matrix.py  # reusable VariableBy2D primitive
└── v2/
    ├── families/          # the 15 authored family generators
    ├── registry.py        # uncapped family registry and split assignment
    ├── release.py         # the three explicit release phases
    └── *_audit.py         # release gates, never called by build
tests/                    # V2 tests only
```

There are no compatibility scripts or hidden V1 entry points. Generated files
live under `build/` and are not source code.

## Pipeline

Generation, auditing, and tokenization are deliberately separate phases.
`build` performs no pytest run, statistical audit, tokenizer check, or hidden
selection cap.

```bash
# Phase 1 — render every registered V2 example
uv run card-corpus build \
  --output build/card-corpus-v2

# Phase 2 — run release gates explicitly
uv run card-corpus audit \
  --artifact build/card-corpus-v2 \
  --tokenizer /path/to/complexity-framework/tokenizer

# Phase 3 — create framework-compatible assistant-only SFT shards
uv run card-corpus tokenize \
  --artifact build/card-corpus-v2 \
  --tokenizer /path/to/complexity-framework/tokenizer \
  --output build/card-corpus-v2-tokenized
```

The tokenization phase refuses an artifact whose separate audit is not green.
Existing output directories are never overwritten implicitly.

## Tests

The default suite contains fast V2 contract and regression tests only:

```bash
uv run pytest
```

Full-capacity family renders are retained as explicit slow tests and are not
part of the default developer loop:

```bash
uv run pytest -m slow -o addopts=""
```

Release-quality validation belongs to `card-corpus audit`, not to `build`.

## Full valid capacity

The registry renders every valid scenario combination. It has no global target,
sampling quota, or 400K truncation layer.

| Family | Examples |
|---|---:|
| `brainstorming_creativity` | 384 |
| `casual_conversation` | 45,794 |
| `context_clarification` | 9,216 |
| `conversation_empathy` | 128 |
| `critique_revision` | 64 |
| `explanation_learning` | 32,776 |
| `extraction_classification` | 3,456 |
| `grounded_qa` | 4,608 |
| `planning_comparison` | 384 |
| `practical_action` | 384 |
| `reasoning_verification` | 108,000 |
| `safety_uncertainty` | 384 |
| `summarization_synthesis` | 64 |
| `troubleshooting` | 384 |
| `writing_transformation` | 64 |
| **Total** | **206,090** |

The release split contains 201,587 training, 2,044 validation, and 2,459 test
examples. Split assignment is deterministic and keeps exact and normalized
structural groups from leaking across partitions.

## Current contract

- 15 registered families, including `casual_conversation`;
- no synthetic 400K cap or per-family truncation;
- exactly one supervised assistant target per row;
- optional `<think>/<final>` only for reasoning rows;
- deterministic train, validation, and test splits;
- prompt, thinking, and answer surfaces separated by role;
- independent gates for correctness, rendering, repetition, diversity, length,
  split leakage, and tokenizer/loss-mask alignment.

The generated manifest records `tests_executed_during_build: false` and
`statistical_audits_executed_during_build: false` so phase separation remains
machine-checkable. A release becomes trainable only after the separate audit
sets `quality_status` to `passed`; all 15 family roadmaps must pass as part of
that decision.
