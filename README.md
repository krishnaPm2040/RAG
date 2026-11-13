# RBI Circulars RAG Chatbot

Ask questions about RBI circulars. Answers come only from the PDFs, with citations [1], [2].
Everything runs locally with Ollama: no API key needed.

## Files

| File | What it does |
|---|---|
| `clean_pdfs.py` | PDFs in `raw_pdfs/` -> clean JSON in `output/json/` |
| `app.py` | JSON -> chunks (recursive, fixed, semantic) -> vector databases in `vector_db/` |
| `streamlit_app.py` | Chat UI (only opens the databases, never builds them) |
| `eval.py` | Hit-rate, MRR, doc hit-rate, no-answer check -> `eval_results.csv` |
| `qa_test_set.csv` | Test questions for `eval.py` |

## Setup (one time)

1. Install Ollama (https://ollama.com), then:
   ```
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```
2. Install Python libraries:
   ```
   python -m venv .venv
   .venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Linux/Mac)
   pip install -r requirements.txt
   ```

## Run (first time)

```
python clean_pdfs.py          # 1. PDF -> clean JSON
python app.py                 # 2. build all 3 databases (semantic is slow: do not stop it)
streamlit run streamlit_app.py   # 3. chat at http://localhost:8501
```

Build only one strategy: `python app.py --strategy=fixed`
Force a full rebuild:   `python app.py --rebuild`

## Adding new PDFs

1. Copy the new PDFs into `raw_pdfs/`
2. `python clean_pdfs.py`  (only new or changed PDFs are cleaned)
3. `python app.py`  (sees the documents changed and rebuilds the databases)
4. Restart Streamlit (stop it with Ctrl+C, then run it again)

## Sidebar options

- **Chunking strategy**: which database to search (recursive / fixed / semantic)
- **Hybrid search**: vectors (meaning) + BM25 (exact words, circular numbers)
- **Re-ranker**: fetch 20 chunks, keep the best 5 (better, a bit slower)
- **Query rewriting**: turns short/unclear questions into a clear search query
- **Show retrieved text**: shows the chunk text behind each source (for checking)

Each browser tab keeps its own settings.

## Evaluation

```
python eval.py --strategy=recursive                               # baseline
python eval.py --strategy=recursive --hybrid                      # + BM25
python eval.py --strategy=recursive --hybrid --rerank             # + re-ranker
python eval.py --strategy=recursive --hybrid --rerank --rewrite   # + query rewriting
```
Repeat with `--strategy=fixed` and `--strategy=semantic` to compare chunking.
Every run adds one row to `eval_results.csv` and writes `eval_details_<setup>.csv`.
`--ragas` (faithfulness, answer relevance) needs a separate Python 3.12 venv with `pip install ragas datasets`.

## Do not commit

`vector_db/`, `output/`, `.venv/` (add them to `.gitignore`).