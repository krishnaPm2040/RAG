# Evaluation results

Every setup is scored at the same k on the same questions. In brackets: change against the
baseline of the same chunking strategy (vectors only, default settings). `bm25-pages` is plain
BM25 over whole pages (no chunking, no tuning): the floor every setup must beat.
`-docfocus`, `-stemming` and `+synonyms` rows are the ablation for review items H1 / M5.
**qa_heldout.csv was never used to tune anything: judge the extras on it.**

## qa_test_set.csv (k = 5, md5 f80e5628, 50 questions)

| Setup | Hit@1 | Hit@3 | Hit@k | MRR@k | Doc hit@k | Key facts | No-answer refused | s/question | Won / lost vs baseline (hit@k) |
|---|---|---|---|---|---|---|---|---|---|
| `bm25-pages` | 0.575 | 0.775 | 0.800 | 0.676 | 0.900 |  |  | 0.0 |  |
| `recursive` | 0.725 | 0.850 | 0.875 | 0.784 | 0.925 | 0.685 | 0.900 | 36.9 |  |
| `recursive+hybrid` | 0.750 (+0.025) | 0.875 (+0.025) | 0.900 (+0.025) | 0.810 (+0.026) | 0.925 (+0.000) | 0.749 (+0.064) | 1.000 (+0.100) | 36.4 | +1 q027 / -0 |

## qa_heldout.csv (k = 5, md5 7ebc3535, 20 questions)

| Setup | Hit@1 | Hit@3 | Hit@k | MRR@k | Doc hit@k | Key facts | No-answer refused | s/question | Won / lost vs baseline (hit@k) |
|---|---|---|---|---|---|---|---|---|---|
| `bm25-pages` | 0.438 | 0.688 | 0.750 | 0.568 | 0.875 |  |  | 0.0 |  |
| `recursive` | 0.562 | 0.812 | 0.812 | 0.677 | 0.875 |  |  | 0.3 |  |
| `recursive+hybrid` | 0.562 | 0.750 | 0.750 | 0.656 | 0.875 | 0.762 | 1.000 | 35.1 |  |

Key facts = share of the numbers, amounts, dates and e-mail addresses of the expected answer
that appear in the bot's answer (review M7). With 40 answerable questions one question is 2.5
points: read the won / lost columns, not only the averages (review M8).

All settings of every run (models, chunk size, cut-off, test file md5) are in eval_results.csv.
