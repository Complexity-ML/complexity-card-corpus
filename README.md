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
| Post-training conversations | General assistant | 4,000 source-grouped instruct/chat examples plus a stratified review queue | Human review required |

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
intersected with risk determines the fallback. A seeded dynamic composer favors
the least-used compatible narrative frames and changes the wording when the
dataset seed changes. Hashing assigns stable semantic identities and verifies
final content; it does not select language, safety rules, or compatibility.

The implementation is split into explicit layers:

```text
data/scenario-forge/                 editable dataset registry
scenario_language.py                dynamic language composition
english_morphology.py              verb inflection and clause realization
scenario_integrity.py               identity and verification hashes
scenario_forge.py                   Python build/audit API
cli.py                              command-line adapter
```

The CLI calls the same Python API used by tests and other applications. Dataset
definitions do not import the CLI or persistence code.

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
family-specific payload contracts; and a 100% compatibility match. Validation
holds out complete `(family, domain, intent)` groups, so a semantic group cannot
appear in both partitions. The audit rejects unknown matrix references,
incompatible combinations and content changed after compilation. A separate
surface-composition lint checks all 2,000 rows, every family/frame cell,
punctuation, question-family contracts, suspicious verb chains and the presence
of state, constraint and outcome anchors. It is a deterministic template audit,
not a claim of general grammatical certification.

Intent labels are stored as lemma-first verb phrases rather than enumerated
sentences. The English morphology layer derives base, third-person singular,
past, past-participle and present-participle forms, then realizes agreement,
aspect, negation, modality and question order. Regular forms follow productive
rules; genuinely irregular lemmas use a compact exception table. The manifest
records the number of checked intent phrases, lemmas and realized forms.

### Original post-training conversations

The post-training builder turns every Scenario Forge card into one instruct
example and one four-turn chat. State and constraint anchors each occur once in
the source dialogue, common `a`/`an` errors are corrected from pronunciation
rules, and the upstream scenario audit keeps mean scenario-sentence length
between 14 and 20 words with 0.1–0.3 transition words per sentence. Conversation
and response lengths are reported separately in the post-training audit.

```bash
uv run card-corpus build-post-training \
  --scenarios build/scenario-forge/scenarios.parquet \
  --output build/post-training \
  --variants-per-scenario 2 \
  --review-scenarios 70
```

Both variants inherit the split of their source `scenario_id`. The audit checks
two explicit holdout contracts: zero shared scenario IDs and zero shared
`(family, domain, intent)` groups between train and validation. This does not
claim zero lexical or semantic near-duplicate leakage.

The two modes remain a paired review unit but no longer copy the same opening:
the instruct prompt uses the scenario trigger, while chat selects an independent
state-bearing opener. The build rejects an exact first-user-message match or a
chat opener that is a literal prefix of its paired instruct prompt. State,
constraint and desired outcome remain shared semantic anchors.

Exact-string conversation and response uniqueness are reported as descriptive
checks, not as evidence of narrative individualization. The response composer
uses eight original openings, eight action frames and eight constraint frames
for each of seven task families, then balances twelve narrative orders. The
diversity report exposes the realized combinations, MATTR-100, distinct and
singleton 4/8-gram ratios, maximum repeated-phrase coverage and the most
repeated n-grams. These measurements are separated for user prompts, all
assistant messages and final assistant responses. `distinct_ngram_ratio` means
distinct n-gram windows divided by all windows; it is not a singleton rate.

Fallback actions are realized through 84 original formulations and conclusions
through 24 independently balanced frames. The build fails when any one surface
form reaches 5% of final responses. Semantic fallback categories are reported
separately because an intentionally frequent safety policy is not the same as a
repeated sentence. A second audit replaces subject, intent, state, constraint,
outcome, fallback and domain context with placeholders before measuring exact
skeleton and 4/8-gram diversity. This prevents variable values from hiding a
repetitive response structure. The build also rejects masked eight-token prose
that reaches 5% of final responses. Raw source anchors remain visible in the
unmasked statistics, so a deliberately repeated constraint cannot be mistaken
for a repeated response template.

`human_review.csv` samples 70 **unique source scenarios**, ten from each of the
seven families, while cycling through risk level, split and domain. Both the
instruct and chat rendering of every selected scenario are included, producing
140 review rows with exactly 70 rows per mode. This equal-family stratification
is a quality-control design, not a population-proportional estimate.

The manifest deliberately sets `training_ready=false` and
`release_ready=false`. A reviewer must grade semantic accuracy, constraint
following, language quality, individualization and safety before either flag
can be changed. Every completed row also records the reviewer and a timezone-aware
UTC timestamp. Passing automated tests is not a substitute for this review.

After the CSV is completed, the explicit gate reports readiness only when all
140 rows are marked `approved`, every quality field is marked `pass`, and review
provenance is complete:

```bash
uv run card-corpus audit-post-training-review \
  --review build/post-training/human_review.csv
```

When all 70 source scenarios pass, the report includes the one-sided 95%
zero-failure bound that would apply to an independent random sample. It is
labelled as descriptive only because the actual sample is stratified. A clean
70-scenario review therefore supports a bounded quality-control statement, not
a claim that the complete corpus is error-free.

### Private lexical mining and anti-copy audit

Pinned third-party dialogue sources may be inspected locally to measure their
structure and extract **single normalized vocabulary tokens only**. The lexical
mine never writes utterances, source order, record identifiers, phrases or
n-grams. Its output is a private candidate lexicon, is marked
`release_ready=false`, retains each source license and revision, and requires
human approval before a word can inform an original language palette.

```bash
uv run card-corpus fetch-lexical-mine \
  --registry /private/path/conversation-sources.json \
  --raw /private/path/raw

uv run card-corpus build-lexical-mine \
  --registry /private/path/conversation-sources.json \
  --raw /private/path/raw \
  --output build/private-lexical-mine \
  --scenarios build/scenario-forge

uv run card-corpus audit-source-overlap \
  --registry /private/path/conversation-sources.json \
  --raw /private/path/raw \
  --scenarios build/scenario-forge \
  --window-tokens 8
```

The mine reports comparable aggregate statistics per source and conversation
role: document count, mean/median/p95 length, question rate, type-token ratio
and retained-vocabulary coverage. Exact normalized documents and sentences are
counted only through transient BLAKE2b digests. The released audit contains
`unique`, `2-4`, `5-9`, `10-24` and `25+` repetition levels, but no digest or
source text. When Scenario Forge is supplied, the mine also maps each transient
eight-token window to coarse classes such as determiner, pronoun, auxiliary,
transition, preposition and content. It reports distribution divergence for
window shapes, sentence openings, sentence endings and transition positions.
Masked windows also receive the same `unique`, `2-4`, `5-9`, `10-24` and `25+`
repetition profile. This is the reference used to match structural repetition
without borrowing source wording. The comparison reports per-level deltas and
total-variation distance between the external reference and generated corpus.
The abstract comparison does not retain lexical n-grams and is a style
diagnostic, not proof of grammatical correctness. The overlap audit separately
hashes source windows only in memory and fails if generated titles, triggers,
situations or goals reproduce an eight-token source sequence. Neither source
text nor source-window hashes are retained.

An eight-word length is not treated as an automatic reuse permission. A common
functional expression may be added only through a separate human-reviewed
palette after checking that it is conventional, necessary for the domain and
compatible with the source terms. Automated mining never promotes a lexical
window into released training text.

The current aggregate comparison and its acceptance criteria are documented in
[the post-training surface reference audit](docs/post-training-reference-audit.md).

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

uv run card-corpus build-post-training \
  --scenarios build/scenario-forge/scenarios.parquet \
  --output build/post-training

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
