# Evaluation results

Every setup is scored at the same k on the same questions. In brackets: change against the
baseline of the same chunking strategy (vectors only, default settings). `bm25-pages` is plain
BM25 over whole pages (no chunking, no tuning): the floor every setup must beat.
`-docfocus`, `-stemming` and `+synonyms` rows are the ablation for review items H1 / M5.
**qa_heldout.csv was never used to tune anything: judge the extras on it.**

## qa_test_set.csv (k = 5, md5 75a1bd17, 50 questions)

| Setup | Hit@1 | Hit@3 | Hit@k | MRR@k | Doc hit@k | Key facts | No-answer refused | Faithfulness | Answer relevance | s/question | Won / lost vs baseline (hit@k) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bm25-pages` | 0.025 | 0.025 | 0.025 | 0.025 | 0.025 |  |  |  |  | 0.0 |  |
| `fixed` | 0.650 | 0.800 | 0.875 | 0.729 | 0.900 | 0.670 | 1.000 |  |  | 35.4 |  |
| `fixed+hybrid` | 0.700 (+0.050) | 0.850 (+0.050) | 0.925 (+0.050) | 0.777 (+0.048) | 0.925 (+0.025) | 0.769 (+0.099) | 1.000 (+0.000) |  |  | 39.0 | +2 q027 q030 / -0 |
| `fixed+hybrid+rerank` | 0.800 (+0.150) | 0.900 (+0.100) | 0.900 (+0.025) | 0.850 (+0.121) | 0.925 (+0.025) | 0.701 (+0.031) | 0.900 (-0.100) |  |  | 42.4 | +2 q027 q030 / -1 q019 |
| `fixed+hybrid+rerank+rewrite` | 0.800 (+0.150) | 0.900 (+0.100) | 0.900 (+0.025) | 0.850 (+0.121) | 0.925 (+0.025) | 0.738 (+0.068) | 0.800 (-0.200) |  |  | 250.9 | +2 q027 q030 / -1 q019 |
| `recursive` | 0.725 | 0.850 | 0.875 | 0.784 | 0.925 | 0.685 | 0.900 |  |  | 31.8 |  |
| `recursive+hybrid` | 0.750 (+0.025) | 0.875 (+0.025) | 0.900 (+0.025) | 0.810 (+0.026) | 0.925 (+0.000) | 0.749 (+0.064) | 1.000 (+0.100) |  |  | 35.5 | +1 q027 / -0 |
| `recursive+hybrid+rerank` | 0.775 (+0.050) | 0.925 (+0.075) | 0.925 (+0.050) | 0.842 (+0.058) | 0.925 (+0.000) | 0.733 (+0.048) | 0.900 (+0.000) |  |  | 39.1 | +2 q012 q027 / -0 |
| `recursive+hybrid+rerank+rewrite` | 0.800 (+0.075) | 0.925 (+0.075) | 0.925 (+0.050) | 0.854 (+0.070) | 0.925 (+0.000) | 0.772 (+0.087) | 0.900 (+0.000) |  |  | 47.6 | +2 q012 q027 / -0 |

## qa_heldout.csv (k = 5, md5 7ebc3535, 20 questions)

| Setup | Hit@1 | Hit@3 | Hit@k | MRR@k | Doc hit@k | Key facts | No-answer refused | Faithfulness | Answer relevance | s/question | Won / lost vs baseline (hit@k) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bm25-pages` | 0.062 | 0.062 | 0.062 | 0.062 | 0.062 |  |  |  |  | 0.0 |  |
| `recursive` (retrieval only) | 0.562 | 0.812 | 0.812 | 0.677 | 0.875 |  |  |  |  | 0.3 |  |
| `recursive+hybrid` (retrieval only) | 0.562 (+0.000) | 0.750 (-0.062) | 0.750 (-0.062) | 0.656 (-0.021) | 0.875 (+0.000) |  |  |  |  | 0.3 |  |
| `recursive+hybrid` | 0.562 | 0.750 | 0.750 | 0.656 | 0.875 | 0.762 | 1.000 |  |  | 35.1 |  |

Key facts = share of the numbers, amounts, dates and e-mail addresses of the expected answer
that appear in the bot's answer (review M7). With 40 answerable questions one question is 2.5
points: read the won / lost columns, not only the averages (review M8).
Faithfulness / answer relevance = RAGAS (python eval.py ... --ragas), graded by the same
local model (qwen3:8b), so treat them as rough.

All settings of every run (models, chunk size, cut-off, test file md5) are in eval_results.csv.
