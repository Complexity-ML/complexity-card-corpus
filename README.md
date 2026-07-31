# Complexity Card Corpus

An original, local-first corpus built from Complexity-authored knowledge cards,
semantic scenarios and deterministic generation rules.

The editable source is the card or scenario schema. Parquet is the canonical
dataset artifact. o200k `uint32` token streams are derived training artifacts.
No third-party dataset is included in the published Complexity corpus.

## Project principles

- **Original material only.** Released records originate from the schemas,
  semantic atoms, facts and worlds authored in this repository.
- **Inspectable before tokenization.** Every training artifact has a readable
  Parquet or JSONL representation.
- **Deterministic generation.** The same registry and seed produce the same
  cards, relations, scenarios and hashes.
- **Graph-first grounding.** Instruction examples retain their source-card keys
  and evidence instead of inventing unsupported answers.
- **One release license.** Complexity-authored dataset content is released under
  **CC BY-NC 4.0**.

## Original collections

| Collection | Domain | Contents | Status |
| --- | --- | --- | --- |
| Aethoria | Fantasy | Curated characters, locations, factions, artifacts and lore | Ready |
| Aethoria Grand Archive | Fantasy | 10,000 generated characters connected to locations, factions, artifacts, rituals and omens | Ready |
| Prismwilds | Speculative natural history | Curated creatures, habitats, guilds, relics and food | Ready |
| Prismwilds Grand Codex | Speculative natural history | 10,000 generated creatures and a connected field-world graph | Ready |
| Meridian Isles | Fantasy | Original island and relationship cards | Ready |
| Computing Foundations | Computing | Original technical concept cards | Ready |
| Systems Foundations | Computing | Original systems concept cards | Ready |
| Scenario Forge | General assistant | 2,000 structured semantic scenarios across seven families | Semantic review in progress |

The checked-in source graph currently contains **21,602 cards** and **143,026
typed relations** across seven original collections.

## Atlas Forge

Atlas Forge expands editable thematic blueprints into large linked-card decks.
It traverses declared key slots deterministically, creates stable identifiers
and validates every relation target.

```bash
uv run card-corpus forge \
  --blueprint data/forge/prismwilds-v1.blueprint.json \
  --output data/source/prismwilds-v1 \
  --force

uv run card-corpus forge \
  --blueprint data/forge/prismwilds-grand-codex-v1.blueprint.json \
  --output data/source/prismwilds-grand-codex-v1 \
  --force

uv run card-corpus forge \
  --blueprint data/forge/aethoria-grand-archive-v1.blueprint.json \
  --output data/source/aethoria-grand-archive-v1 \
  --force
```

An archetype may declare `keySlots`. Their Cartesian capacity must cover the
requested card count unless the name template also contains the unique
`{index}` field.

```json
{
  "count": 4,
  "slots": {
    "prefix": ["Prism00", "Prism01"],
    "suffix": ["ling00", "ling01"]
  },
  "keySlots": ["prefix", "suffix"],
  "nameTemplate": "{prefix}{suffix}"
}
```

## Card data model

Each directory under `data/source/` contains:

- `dataset.json`: identity, domain, language, version, split and license;
- `cards.json`: stable keys, types, summaries, facts, attributes and typed
  relations.

The corpus build produces:

- `cards.parquet`: normalized cards;
- `relations.parquet`: directed graph edges;
- `documents.parquet`: deterministic entity, neighborhood and multi-hop path
  documents;
- `manifest.json`: counts, hashes and complete build provenance.

Connected collections are assigned wholly to one split so the same graph does
not leak into train and validation through different cards.

## Scenario Forge

Scenario Forge creates inspectable assistant scenarios without generating model
dialogue. Each card combines:

```text
family + domain-compatible intent
       + intent/state-compatible outcome
       + domain-compatible constraint
       + current state + prioritized risk/state fallback
       + domain-specific trigger + deterministic narrative frame
```

Compatibility is explicit in the registry. Domain constrains both intent and
constraint, intent and state jointly constrain the outcome, and state priority
intersected with risk determines the fallback. Hashing is applied only after
those semantic rules: it distributes compatible combinations, selects one of
12 authored narrative frames, assigns stable identities and verifies final
content. It does not decide safety or semantic compatibility.

The current registry compiles exactly 2,000 unique semantic signatures:

- 600 practical actions and services;
- 400 explanations and learning scenarios;
- 300 troubleshooting scenarios;
- 250 writing and transformation scenarios;
- 200 planning and comparison scenarios;
- 150 conversational and empathetic scenarios;
- 100 safety, refusal and uncertainty scenarios.

```bash
uv run card-corpus build-scenario-forge \
  --registry data/scenario-forge/scenario-forge-v1.json \
  --output build/scenario-forge
```

`scenarios.parquet` is canonical and `scenarios.jsonl` is intended for human
review. Every card carries a unique title, objective and original situation, a
domain-specific trigger, a `creation_hash` over its semantic signature and a
`verification_hash` over its canonical final content. The explicit
`model_generated_dialogue=false` field distinguishes composed scenario prose
from generated dialogue. The artifact manifest separately records SHA-256
hashes for each output file.

The audit enforces 2,000 unique situations, titles, objectives, IDs,
signatures, payloads and hash pairs; exactly 1,900 training and 100 validation
cards; exact family allocation; domain balance; complete axis coverage;
family-specific payload contracts; and a 100% compatibility match. It rejects
unknown matrix references, incompatible combinations and content changed after
compilation.

## Original instruction tuning

`build-instruct` derives instruction and multi-turn examples directly from the
original card graph. Deterministic templates cover summaries, attributes,
facts, direct relations, comparisons, structured extraction, grounded
follow-ups and multi-hop paths. No language model writes or paraphrases rows.

The current complete build contains **204,471 examples**:

- 182,872 one-turn instructions;
- 21,599 multi-turn chats;
- 204,264 training examples;
- 207 validation examples.

Every example retains source-card keys and evidence. Connected graph decks stay
in one split.

## Build locally

```bash
cd /Users/boris/Dev/complexity-card-corpus
uv sync --extra dev

uv run card-corpus build \
  --source data/source \
  --output build/corpus

uv run card-corpus build-instruct \
  --corpus build/corpus \
  --output build/atlas-instruct

uv run card-corpus build-scenario-forge \
  --registry data/scenario-forge/scenario-forge-v1.json \
  --output build/scenario-forge

uv run pytest -q
```

## Tokenize with o200k

```bash
uv run card-corpus tokenize \
  --documents build/corpus/documents.parquet \
  --tokenizer /Users/boris/Dev/complexity-framework/tokenizer-o200k \
  --output build/tokenized/o200k

uv run card-corpus tokenize-instruct \
  --instructions build/atlas-instruct \
  --tokenizer /Users/boris/Dev/complexity-framework/tokenizer-o200k \
  --output build/atlas-instruct-o200k
```

`tokens.bin` uses little-endian `uint32`, because o200k token IDs do not fit in
`uint16`. An EOS token is appended after each rendered document.

The causal-SFT artifact contains:

```text
train/input_ids.bin
train/labels.bin
train/examples.jsonl
train/sft.idx.json
eval/...
manifest.json
```

Labels use `-100` for user prefixes and padding. Only assistant tokens and the
terminating EOS token contribute to the supervised loss.

## Release boundary

The official Complexity release may contain only:

- records whose source metadata identifies Complexity original authorship;
- deterministic derivatives of those records;
- manifests, audits and token shards derived exclusively from those records;
- CC BY-NC 4.0 dataset content.

Third-party corpora are **not part of the Complexity Card Corpus release**.
Their inputs and derivatives must never be packaged, published or represented
as Complexity-owned datasets.

## License

The original dataset schemas, card content, semantic registries, generated
world content, deterministic rendered prose and derived dataset artifacts are
offered under **Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0)**.

This license statement applies only to material authored by Complexity. The
release boundary above intentionally excludes third-party dataset content.
