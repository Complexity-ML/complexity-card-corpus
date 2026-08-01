# Post-training surface reference audit

This audit compares Scenario Forge with four established post-training
datasets at the level of **aggregate surface statistics only**. It is not a
ranking of dataset quality, and it does not imply that matching any single
number makes a corpus suitable for training.

No third-party utterance, phrase, record order, identifier or n-gram is stored
in this repository. Source artifacts were pinned, verified, scanned locally
and deleted after the aggregate mine completed. The private candidate lexicon
retains only normalized single tokens and is explicitly not release-ready.

## Reference snapshots

| Dataset | Pinned revision | Declared license | Extracted text segments |
|---|---|---|---:|
| `HuggingFaceH4/no_robots` | `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b` | CC BY-NC 4.0 | 22,359 |
| `HuggingFaceH4/ultrachat_200k` | `8049631c405ae6576f93f445c6b8166f76f5505a` | MIT | 1,315,476 |
| `databricks/databricks-dolly-15k` | `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | CC BY-SA 3.0 | 34,371 |
| `OpenAssistant/oasst1` | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | Apache-2.0 | 39,543 |

For OASST1, the scan retained English, non-deleted, positively reviewed
messages. UltraChat used the `train_sft` split. No Robots and Dolly used their
published training data.

## Comparable surface statistics

| Corpus | Mean words | Median | P95 | Observed vocabulary | Question rate | Unique documents | Unique sentences |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scenario Forge, initial surface layer | 67.97 | 68 | 74 | 779 | 0.0% | 100.0% | 29.3% |
| **Scenario Forge, current** | **119.94** | **119** | **135** | **1,225** | **27.7%** | **100.0%** | **47.0%** |
| No Robots | 71.29 | 29 | 278 | 52,577 | 17.2% | 99.2% | 94.9% |
| UltraChat 200k | 119.22 | 67 | 378 | 374,871 | 32.0% | 100.0% | 86.1% |
| Dolly 15k | 44.73 | 14 | 175 | 63,623 | 25.7% | 97.5% | 88.5% |
| OASST1 | 70.51 | 31 | 246 | 56,846 | 27.2% | 98.6% | 90.8% |

The reference statistics count extracted messages or fields rather than whole
conversation trees. Sentence boundaries use punctuation heuristics, so these
figures are directional diagnostics rather than benchmark scores.

The current miner now preserves the conversation role only as an aggregate
partition. Prompt and assistant distributions are reported independently.
Within each partition, normalized document and sentence repetitions are grouped
into `unique`, `2-4`, `5-9`, `10-24` and `25+` occurrence levels. Counting uses
transient BLAKE2b digests; neither the text nor the digests enter the artifact.

The private mine can additionally compare eight-token windows after replacing
every word with a coarse structural class (for example `DETERMINER`, `PRONOUN`,
`AUXILIARY`, `TRANSITION`, `PREPOSITION` or `CONTENT`). Jensen--Shannon
divergence is measured for those abstract windows, sentence openings, sentence
endings and transition positions. This captures chaining rhythm without
retaining or reproducing source phrases. A low divergence is not, by itself, a
grammar score.

## Eight-token chaining snapshot

The first chaining snapshot uses the two pinned conversational sources whose
raw artifacts remained available locally: 166,739 Taskmaster utterances and
99,460 Empathetic Dialogues utterances. This reference is strongly biased
toward short dialogue turns, so it is useful for finding overlong or
over-signposted composition but is not a target distribution for a full
scenario card.

| Aggregate | Dialogue reference | Scenario Forge |
|---|---:|---:|
| Mean words per sentence | 7.92 | 26.70 |
| Transitions per sentence | 0.080 | 0.921 |
| Sentence-initial transition rate | 0.84% | 5.60% |
| Adjacent repeated-word rate | 1.17% | 0.00% |
| Abstract eight-token shapes | 229,500 | 8,753 |
| Abstract-shape entropy | 15.54 bits | 11.62 bits |

The Jensen--Shannon divergence is 0.804 bits for abstract eight-token shapes,
0.775 for sentence openings, 0.391 for sentence endings and 0.005 for
transition positions. The final number shows that transitions appear in
roughly the same relative positions; the remaining gap comes from longer,
more explicitly structured scenario prose and a much narrower template
inventory. This is now recorded as an optimization signal rather than hidden
behind document-level uniqueness.

## Findings

Scenario Forge is strong on structural coverage and document-level uniqueness:
all 15,000 compiled scenarios are distinct, and the validation split holds out
complete `(family, domain, intent)` groups. Its mean length is within the range
of the references.

The expanded original language layer raises question rate to 27.7% and keeps
every scenario unique while adding seven generalist families. Sentences average
19.99 words and transitions average 0.236 per sentence, both inside the current
acceptance bands. Exact sentence uniqueness falls to 47.0% at the larger scale,
which is expected from shared semantic anchors but remains a signal to improve.
The original measurable weakness was lexical breadth: 1,225 observed words in
Scenario Forge and 1,533 across the rendered conversations were narrow for a
general post-training corpus. The vocabulary-augmented build now observes
5,637 words while preserving the same 15,000 semantic scenarios.

The mined semantic-role labels are not suitable for automatic generation. A
new cross-source vocabulary-gap audit first removes words already present,
function words and candidates lacking independent support. With two pinned
sources and a minimum of 20 occurrences in each, it identifies 4,097 absent
single-token candidates. Masked-context placement maps all of them into the
101 realized family/domain cells and stores the selected use plus as many as
four alternative statistical contexts per word. The primary placement is only
the rendering choice; it is not a
single-sense lexical claim. Complete source phrases are never reintroduced.

The placement audit reports 2,111 statistically supported, 1,271 statistically
plausible and 715 review-required words. An optional Open English WordNet proxy
resolves 98.41% of entries; the selected family is in its derived top 1, 3 and
5 for 20.72%, 54.56% and 75.06% of comparable words. This comparison is not a
ground-truth accuracy score and WordNet definitions do not enter the corpus.

## Next acceptance targets

The next language layer should be accepted only when it:

1. preserves the 15,000 semantic signatures and exact 14,250/750 split;
2. records unique-sentence rate and improves it without reducing semantic coverage;
3. keeps question rate between 25% and 30%, only in appropriate families;
4. expands vocabulary through reviewed single-token candidates;
5. passes the transient eight-token source-overlap audit with zero matches.
6. keeps every fallback formulation and conclusion frame below 5% of final
   responses;
7. reports prompt, assistant-message and final-response diversity separately;
8. keeps the dominant exact response skeleton below 5% after semantic variables
   are replaced with placeholders;
9. keeps each family-level masked response template below the explicit 20%
   within-family ceiling;
10. keeps every masked eight-token prose sequence below 5% while retaining raw
    source-anchor repetition in a separate unmasked report.

## Current post-training surface gate

The current four-surface build contains 36,449 readable conversations after
exact transcript/response cleanup and a cap on dominant raw families. Its
model-facing projection starts from the 34,641 generated training rows, removes
5,270 exact response duplicates, caps 238 rows from the dominant family, then
removes 1,496 structurally repetitive rows. Prose structures retain at most 48
examples; exact-unique extraction JSON uses a schema-aware ceiling of 512 so a
valid field contract is not mistaken for prose duplication. This leaves 27,637
training examples and 3,139,173 supervised training tokens. With the held-out
evaluation split, the artifact contains 28,337 examples and 3,162,724
supervised tokens.

The retained projection has 100% exact final-response uniqueness, all 14 task
families, 11,026 easy, 7,605 medium and 9,006 hard examples, and 15,110 genuine
four-message exchanges (54.67%). The largest family holds 14.84% of training
examples and the smallest 2.66%. All four response-length bands exceed 5%, and
20,709 normalized `(family, structure)` pairs are distinct. The held-out
evaluation suite has 700 examples, exactly 50 per family: 28 separately
authored gold exchanges and 672 source-separated diagnostics built without
training renderers.

The readable surface still passes the anti-template diagnostics: the largest
masked eight-token message coverage is 3.49% and the largest within-family
masked-template share is 8.40%. These are repetition diagnostics, not a
correctness score. The 280-row stratified human review also remains pending.
All automated SFT release checks now pass. Publication still requires the
stratified human-review gate; the automated checks do not certify semantic or
safety correctness across the full corpus.
