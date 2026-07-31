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
| **Scenario Forge, current** | **126.16** | **127** | **141** | **898** | **27.5%** | **100.0%** | **81.1%** |
| No Robots | 71.29 | 29 | 278 | 52,577 | 17.2% | 99.2% | 94.9% |
| UltraChat 200k | 119.22 | 67 | 378 | 374,871 | 32.0% | 100.0% | 86.1% |
| Dolly 15k | 44.73 | 14 | 175 | 63,623 | 25.7% | 97.5% | 88.5% |
| OASST1 | 70.51 | 31 | 246 | 56,846 | 27.2% | 98.6% | 90.8% |

The reference statistics count extracted messages or fields rather than whole
conversation trees. Sentence boundaries use punctuation heuristics, so these
figures are directional diagnostics rather than benchmark scores.

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
all 2,000 compiled scenarios are distinct, and the validation split holds out
complete `(family, domain, intent)` groups. Its mean length is within the range
of the references.

The revised original language layer raises unique-sentence rate from 29.3% to
81.1% and question rate from 0.0% to 27.5%, without changing any semantic
signature. It is approaching the 86.1--94.9% sentence range of the reference
snapshots. The chaining audit also caused long three-sentence blocks to be
split into four or five auditable sentences. The remaining measurable
weaknesses are lexical breadth and over-signposting: 898 observed words and
0.921 transitions per sentence remain narrow and mechanically explicit for a
general post-training corpus.

The mined semantic-role labels are not suitable for automatic generation yet.
Transitions are reliable, but heuristic `intent`, `state`, `constraint` and
`outcome` labels contain syntactic and topical false positives. Candidate
vocabulary must therefore pass cross-source support, stop-word and morphology
filters, followed by human approval. Complete source phrases must never be
reintroduced.

## Next acceptance targets

The next language layer should be accepted only when it:

1. preserves the 2,000 semantic signatures and exact 1,900/100 split;
2. keeps unique-sentence rate at or above 80%;
3. keeps question rate between 25% and 30%, only in appropriate families;
4. expands vocabulary through reviewed single-token candidates;
5. passes the transient eight-token source-overlap audit with zero matches.
