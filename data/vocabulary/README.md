# Vocabulary dictionary

This directory contains a statistical vocabulary layer for Scenario Forge.
It retains isolated normalized words and aggregate counts only. No external
sentence, phrase, source ordering or record identifier is included.

## Files

- `vocabulary-dictionary-v1.json` contains 4,097 entries. Each entry has one
  selected generation placement and up to four alternative statistical usages,
  a 101-cell family/domain score vector, masked-context neighbours and source
  occurrence counts. `short_definition` summarizes the dominant masked context;
  every alternative usage has its own `usage_gloss`.
- `vocabulary-placement-v1.csv` maps every word to the single placement used
  when rendering the current post-training corpus. It also exposes the short
  definition, confidence tier, selected rank and serialized alternative usages
  so the CSV can be reviewed without opening the full dictionary.
- `vocabulary-placement-audit-v1.json` records coverage, capacity, placement
  methods and classification status.
- `vocabulary-definition-review-v1.csv` is the editable human-review queue
  formed from the union of two local embedding audits. It preserves the current
  statistical gloss, records both scores, and leaves proposed definitions and
  decisions empty. The file is evidence for review, not an automatic rewrite.
- `vocabulary-definition-proposals-v1.json` contains original draft definitions
  for entries independently flagged by both local embedding models. These
  drafts remain proposals until the CSV decision is changed by a reviewer.
- `definition-proposals-review-v1.csv` compares those drafts with the current
  statistical glosses using two pinned local embedding models. Its consensus
  field is a diagnostic signal only. Accepted rows record the explicit operator
  decision that promotes the corresponding original draft into the canonical
  dictionary while retaining the prior `statistical_gloss`.
- `vocabulary-wordnet-summary-v1.json` reports an optional external semantic
  proxy comparison. It contains no WordNet definitions.

The selected placement is an authoring decision, not a claim that a word has
only one meaning. `statistical_usages` explicitly preserves several candidate
uses. Entries marked `review_required` must not be treated as approved lexical
classifications.

All data artifacts in this directory are covered by `DATASET_LICENSE.md`.
