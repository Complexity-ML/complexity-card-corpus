# Complexity Atlas datasets

A local-first pipeline for authoring an English, multi-domain corpus as linked
knowledge cards. The card graph is the editable source; Parquet is the
canonical dataset artifact; o200k `uint32` streams are derived training
artifacts.

The first source is the original Aethoria catalog from
`complexity-source-mcp`. A small computing domain demonstrates that the schema
is not tied to fantasy.

## Data model

Each source directory contains:

- `dataset.json`: dataset-level provenance, domain, language, version and split;
- `cards.json`: entities with a stable key, type, summary, description, facts,
  tags, attributes and typed relations.

The build produces:

- `cards.parquet`: one normalized row per card;
- `relations.parquet`: one row per directed edge;
- `documents.parquet`: deterministic English entity, neighborhood and
  multi-hop path documents;
- `manifest.json`: counts, hashes and build provenance.

Connected source collections are assigned to one split in `dataset.json`.
This avoids leaking the same graph into train and validation through different
cards.

## Local build

```bash
cd /Users/boris/Dev/complexity-card-corpus
uv sync --extra dev
uv run card-corpus build \
  --source data/source \
  --output build/corpus
uv run card-corpus tokenize \
  --documents build/corpus/documents.parquet \
  --tokenizer /Users/boris/Dev/complexity-framework/tokenizer-o200k \
  --output build/tokenized/o200k
uv run card-corpus import-oasst1 \
  --raw build/raw/oasst1 \
  --output build/alignment/oasst1
uv run card-corpus package-hf \
  --corpus build/corpus \
  --tokenized build/tokenized/o200k \
  --output build/hf-pretrain
uv run card-corpus package-posttrain-hf \
  --alignment build/alignment/oasst1 \
  --output build/hf-posttrain
uv run card-corpus inspect --output build/corpus
uv run pytest
```

`tokens.bin` uses little-endian `uint32`, because o200k token IDs do not fit in
`uint16`. An EOS token is appended after each rendered document.

## Instruct and chat modes

The OASST1 importer pins the official dataset revision and creates two
alignment-card views:

- `instruct.parquet`: the highest-ranked valid assistant response to each
  accepted English root prompt;
- `chat.parquet`: one quality-selected multi-turn path per accepted tree.

Messages are human-authored, non-synthetic OASST1 rows with successful review,
language, safety and quality filters. Each output row retains the source tree
and message IDs, Apache-2.0 license, source revision and a deterministic
`User:`/`Assistant:` rendering. Alignment remains structured Parquet so a later
SFT loader can mask user tokens and calculate loss only on assistant responses.

## Hugging Face datasets

The pipeline publishes two separate artifacts:

- `Complexity Atlas Pretrain`: linked-card documents, normalized graph tables
  and derived o200k token shards;
- `Complexity Atlas Posttrain`: filtered OASST1 instruction and chat cards.

The pretraining upload contains inspectable Parquet and derived token shards:

```text
README.md
manifest.json
data/train.parquet
data/validation.parquet
tables/cards_train.parquet
tables/cards_validation.parquet
tables/relations_train.parquet
tables/relations_validation.parquet
tokenized/o200k/train/tokens.bin
tokenized/o200k/train/tokens.idx.json
tokenized/o200k/eval/tokens.bin
tokenized/o200k/eval/tokens.idx.json
```

Create the Hugging Face repository as a **private dataset** first:

```bash
hf repo create Pacific-i64/complexity-atlas-pretrain \
  --repo-type dataset --private
hf repo create Pacific-i64/complexity-atlas-posttrain \
  --repo-type dataset --private
hf upload Pacific-i64/complexity-atlas-pretrain \
  build/hf-pretrain . --repo-type dataset
hf upload Pacific-i64/complexity-atlas-posttrain \
  build/hf-posttrain . --repo-type dataset
```

No upload is performed by this project. Dataset licensing must be selected
before making the repository public; the current source metadata intentionally
uses `proprietary`.
