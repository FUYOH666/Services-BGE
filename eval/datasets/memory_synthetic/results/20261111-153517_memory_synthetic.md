# Memory bench: memory_synthetic

2026-11-11T08:35:17+00:00 · 15 files in 2 slices (kestrel, docs) → 96 passages (canon 92, inbox 4), indexed in 6 s · 24 questions (ru 8, en 10, mixed 6; answer in canon 21, in inbox 3)

## Retrieval on the clean index

recall@k = share of questions with a passage that holds the answer in the top k; MRR over the top 10.

| method | R@1 | R@5 | MRR | ru R@1 | ru R@5 | en R@1 | en R@5 | mixed R@1 | mixed R@5 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| grep | 38% | 67% | 0.49 | 12% | 12% | 30% | 90% | 83% | 100% | 0 |
| grep_stem | 38% | 62% | 0.48 | 12% | 12% | 40% | 90% | 67% | 83% | 0 |
| dense | 79% | 96% | 0.87 | 88% | 100% | 70% | 90% | 83% | 100% | 107 |
| sparse | 54% | 62% | 0.57 | 12% | 12% | 60% | 80% | 100% | 100% | 104 |
| hybrid | 58% | 92% | 0.75 | 12% | 88% | 70% | 90% | 100% | 100% | 52 |
| hybrid+rerank | 92% | 96% | 0.93 | 88% | 100% | 90% | 90% | 100% | 100% | 1696 |
| hybrid+rerank, prefer canon | 92% | 100% | 0.95 | 88% | 100% | 90% | 100% | 100% | 100% | 1803 |

By slice and by where the answer lives (hybrid + rerank, recall@1 / recall@5):

| group | n | hybrid+rerank | grep |
|---|---|---|---|
| answer:canon | 21 | 90% / 95% | 38% / 71% |
| answer:inbox | 3 | 100% / 100% | 33% / 33% |
| slice:docs | 8 | 88% / 100% | 25% / 75% |
| slice:kestrel | 16 | 94% / 94% | 44% / 62% |

## Context cost

| what the agent reads | characters | ≈ tokens | holds the answer |
|---|---|---|---|
| the 3 read-first files, in full | 2,383 | 866 | 38% of questions |
| every file in the corpus (15) | 57,574 | 23,000 | all questions |
| top-5 passages from search (mean) | 3,881 | 1,542 | 96% of questions |

The top-5 passages are 1.7× larger than the read-first files (a small canon).

## Poisoning

8 false notes written as unreviewed `inbox` notes from bot sources, aimed at 7 questions (6 whose true answer is in canon, 1 whose true answer is itself an inbox note). Styles: authority 2, plain 6; `authority` notes claim to be approved or to supersede canon.

| ranking | false fact 1st | — truth in canon | — truth in inbox | false in top 5 | true fact 1st | all questions R@1 | R@5 | p50 ms |
|---|---|---|---|---|---|---|---|---|
| hybrid+rerank (no tiers) | 100% | 100% | 100% | 100% | 0% | 62% | 96% | 1780 |
| hybrid+rerank, prefer canon x0.75 | 71% | 67% | 100% | 100% | 29% | 71% | 100% | 2195 |
| hybrid+rerank, prefer canon x0.5 | 14% | 0% | 100% | 86% | 86% | 88% | 100% | 2698 |
| hybrid+rerank, prefer canon x0.25 | 14% | 0% | 100% | 86% | 86% | 88% | 100% | 2574 |
| hybrid+rerank, canon only | 0% | 0% | 0% | 0% | 86% | 79% | 88% | 2268 |
| hybrid (no rerank, no tiers) | 86% | 83% | 100% | 100% | 0% | 42% | 92% | 53 |
| hybrid, prefer canon x0.5 | 14% | 0% | 100% | 86% | 71% | 62% | 88% | 58 |

Share of false notes ranked above the true passage, by style:

| ranking | authority | plain |
|---|---|---|
| hybrid+rerank (no tiers) | 50% | 100% |
| hybrid+rerank, prefer canon x0.75 | 50% | 67% |
| hybrid+rerank, prefer canon x0.5 | 0% | 17% |
| hybrid+rerank, prefer canon x0.25 | 0% | 17% |
| hybrid+rerank, canon only | 0% | 0% |
| hybrid (no rerank, no tiers) | 50% | 83% |
| hybrid, prefer canon x0.5 | 0% | 17% |

Settings: chunks 500 tokens / overlap 80, prefetch 50, rerank candidates 20, top 10.
