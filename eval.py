"""
eval.py - Task 3: test the RAG chatbot (Task 4: re-run after each improvement)

Reads your test questions from qa_test_set.csv (or test_set.csv / test_set.json).
Works with your Excel file as it is: tab- or comma-separated, with or without
a header row. Columns (in this order when there is no header):
  id | question | expected_answer | source_doc_id | file_name | page | answer_type | notes
  - page can be "2, 5" for an answer spread over two pages (hit if ANY page is found)
  - answer_type "no_answer" = out-of-scope question: the bot must say it could not find it

  python eval.py --strategy=recursive                                  -> baseline
  python eval.py --strategy=recursive --hybrid                         -> + BM25
  python eval.py --strategy=recursive --hybrid --rerank                -> + re-ranker
  python eval.py --strategy=recursive --hybrid --rerank --rewrite      -> + query rewriting
  add --ragas to any line to also score faithfulness + answer relevance (slower)
  add --test=myfile.csv to use another question file
  add --k=5 to change the k used to score EVERY setup (default 5)

  python eval.py --report       -> eval_report.md: before/after table from eval_results.csv

Fair comparison: every setup is scored on the SAME k (Hit@1, Hit@3, Hit@k, MRR@k),
even though the chatbot itself gives 3 sources (5 with the re-ranker) to the LLM.

Every run:
  - adds one row to eval_results.csv            (compare before / after)
  - writes eval_details_<setup>.csv             (every question: hit, rank, answer)
"""

import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import app
from app import get_vector_db, llm, embeddings, CHUNK_STRATEGY

EVAL_K = 5                                      # Day 4 #1: the SAME k for every setup
for _arg in sys.argv:
    if _arg.startswith("--k="):
        EVAL_K = int(_arg.split("=", 1)[1])

# Works with your app.py as it is: uses its functions when they exist,
# otherwise does the same simple search + prompt itself.
USE_HYBRID = getattr(app, "USE_HYBRID", False)
USE_RERANK = getattr(app, "USE_RERANK", False)
USE_REWRITE = getattr(app, "USE_REWRITE", False)
MIN_RELEVANCE = getattr(app, "MIN_RELEVANCE", 0.0)


def retrieve(vector_db, question, k=None):
    if hasattr(app, "retrieve"):
        try:
            return app.retrieve(vector_db, question, top_k=k)
        except TypeError:                           # older app.py without top_k
            return app.retrieve(vector_db, question)
    return vector_db.similarity_search(question, k=k or 3, filter={"is_latest": True})


def answer_k():
    """How many sources the CHATBOT gives the LLM (3, or 5 with the re-ranker)."""
    return app.top_k_for(USE_RERANK) if hasattr(app, "top_k_for") else 3


def generate_answer(question, results):
    if hasattr(app, "generate_answer"):
        return app.generate_answer(question, results)
    context = "\n\n".join(
        f"[{i}] ({d.metadata['source']}, page {d.metadata['page']})\n{d.page_content}"
        for i, d in enumerate(results, start=1))
    messages = [
        ("system", f"You answer questions about {getattr(app, 'DOMAIN', 'RBI circulars')}. "
                   "Use ONLY the context below. If the answer is not in the context, say "
                   "'I could not find this in the documents.' Do not invent rules. "
                   "When you use information from the context, cite it inline using the "
                   "bracketed number, like [1] or [2], right after the relevant sentence."),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    return llm.invoke(messages).content


def top_relevance(vector_db, question):
    if hasattr(app, "top_relevance"):
        return app.top_relevance(vector_db, question)
    return None                                     # only with the newer app.py

TEST_FILES = [Path("qa_test_set.csv"), Path("output/qa_test_set.csv"),
              Path("test_set.csv"), Path("test_set.json")]
for arg in sys.argv:
    if arg.startswith("--test="):
        TEST_FILES = [Path(arg.split("=", 1)[1])]

RESULTS_FILE = Path("eval_results.csv")
NOT_FOUND = re.compile(r"could not find|not (?:available|mentioned|found|present) in the (?:documents|context)"
                       r"|no information|does not (?:contain|mention)", re.I)
COLUMNS = ["id", "question", "expected_answer", "source_doc_id", "file_name",
           "page", "answer_type", "notes"]


# ---------------------------------------------------------------------------
# 1. Load the test questions
# ---------------------------------------------------------------------------

def norm_name(text):
    """'Governor’s Statement.pdf' and 'governor_s_statement' compare equal."""
    text = re.sub(r"\.pdf$", "", (text or "").strip(), flags=re.I)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def parse_pages(text):
    """'2, 5' -> {2, 5};  '3' -> {3};  '' -> set()"""
    return {int(p) for p in re.findall(r"\d+", str(text or ""))}


def read_csv_rows(path):
    raw = path.read_text(encoding="utf-8-sig")
    delimiter = "\t" if raw.count("\t") > raw.count(",") / 3 else ","
    rows = list(csv.reader(raw.splitlines(), delimiter=delimiter))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if rows and rows[0] and rows[0][0].strip().lower() == "id":       # header row
        header = [h.strip().lower() for h in rows[0]]
        aliases = {"source": "file_name", "answer": "expected_answer", "type": "answer_type"}
        header = [aliases.get(h, h) for h in header]
        return [dict(zip(header, r)) for r in rows[1:]]
    return [dict(zip(COLUMNS, r)) for r in rows]                       # no header: by position


def load_cases():
    path = next((p for p in TEST_FILES if p.exists()), None)
    if path is None:
        sys.exit("No test file found. Put qa_test_set.csv next to eval.py, "
                 "or run with --test=yourfile.csv")
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        rows = [{"question": r["question"], "expected_answer": r.get("answer", ""),
                 "file_name": r.get("source", ""), "page": r.get("page", ""),
                 "answer_type": r.get("type", "")} for r in rows]
    else:
        rows = read_csv_rows(path)

    cases = []
    for n, r in enumerate(rows, start=1):
        question = (r.get("question") or "").strip()
        if not question:
            continue
        cases.append({
            "id": (r.get("id") or f"q{n:03d}").strip(),
            "question": question,
            "answer": (r.get("expected_answer") or "").strip(),
            "file": (r.get("file_name") or "").strip(),
            "doc_id": (r.get("source_doc_id") or "").strip(),
            "pages": parse_pages(r.get("page")),
            "type": (r.get("answer_type") or "").strip().lower() or "paragraph",
        })
    print(f"Test questions: {len(cases)} from {path}")
    return cases


# ---------------------------------------------------------------------------
# 2. Checks
# ---------------------------------------------------------------------------

def same_doc(doc, case):
    src = norm_name(doc.metadata.get("source"))
    return src in {norm_name(case["file"]), norm_name(case["doc_id"])} - {""}


def is_hit(doc, case):
    """Right file AND one of the expected pages."""
    try:
        page = int(doc.metadata.get("page"))
        page_end = int(doc.metadata.get("page_end") or page)
    except (TypeError, ValueError):
        return False
    # Fix 6: a chunk covers pages page..page_end (a table on pages 2-3 counts for
    # both). Old databases without page_end: allow the next page as well.
    if "page_end" not in doc.metadata:
        page_end = page + 1
    return same_doc(doc, case) and any(page <= p <= page_end for p in case["pages"])


def run_label():
    parts = [CHUNK_STRATEGY]
    parts += [name for name, on in (("hybrid", USE_HYBRID), ("rerank", USE_RERANK),
                                    ("rewrite", USE_REWRITE)) if on]
    if getattr(app, "SEARCH_SCOPE", "all") != "all":
        parts.append(app.SEARCH_SCOPE)
    if MIN_RELEVANCE:
        parts.append(f"cutoff{MIN_RELEVANCE}")
    return "+".join(parts)


def test_file():
    return next((p for p in TEST_FILES if p.exists()), None)


def run_settings(vector_db):
    """Day 4 #2: everything needed to tell runs apart later."""
    path = test_file()
    chat_model = (getattr(app, "HF_MODEL", "") if getattr(app, "LLM_PROVIDER", "") == "huggingface"
                  else getattr(app, "CHAT_MODEL", ""))
    return {
        "strategy": CHUNK_STRATEGY,
        "hybrid": USE_HYBRID, "rerank": USE_RERANK, "rewrite": USE_REWRITE,
        "scope": getattr(app, "SEARCH_SCOPE", "all"),
        "eval_k": EVAL_K, "answer_k": answer_k(),
        "chunk_size": getattr(app, "CHUNK_SIZE", ""), "chunk_overlap": getattr(app, "CHUNK_OVERLAP", ""),
        "db_version": getattr(app, "DB_VERSION", ""),
        "chunks_in_db": vector_db._collection.count(),
        "embed_model": getattr(app, "EMBED_MODEL", ""),
        "chat_model": f"{getattr(app, 'LLM_PROVIDER', 'ollama')}:{chat_model}",
        "min_relevance": MIN_RELEVANCE,
        "include_annual_reports": getattr(app, "INCLUDE_ANNUAL_REPORTS", ""),
        "test_file": path.name if path else "",
        "test_file_md5": hashlib.md5(path.read_bytes()).hexdigest()[:8] if path else "",
    }


# ---------------------------------------------------------------------------
# 3. Run every question
# ---------------------------------------------------------------------------

def run_questions(vector_db, cases):
    details = []
    for n, case in enumerate(cases, start=1):
        docs = retrieve(vector_db, case["question"], k=EVAL_K)        # scored at the SAME k
        used = docs[:answer_k()]                                      # what the chatbot answers from
        answer = generate_answer(case["question"], used) if used else "I could not find this in the documents."
        row = {"case": case, "docs": docs, "used": used, "answer": answer}
        try:
            row["score"] = top_relevance(vector_db, case["question"])   # for the cut-off advice
        except Exception:
            row["score"] = None

        if case["type"] == "no_answer":
            row["refused"] = bool(NOT_FOUND.search(answer))
            status = "OK (refused)" if row["refused"] else "WRONG (answered)"
        else:
            row["rank"] = next((i for i, d in enumerate(docs, start=1) if is_hit(d, case)), None)
            row["doc_rank"] = next((i for i, d in enumerate(docs, start=1) if same_doc(d, case)), None)
            status = (f"HIT @{row['rank']}" if row["rank"]
                      else "DOC ONLY" if row["doc_rank"] else "MISS")
        print(f"  [{n}/{len(cases)}] {case['id']:<5} {case['type']:<13} {status:<16} {case['question'][:55]}")
        details.append(row)
    return details


def suggest_cutoff(details):
    """MIN_RELEVANCE advice: real questions should score ABOVE it, no_answer ones BELOW.
    Day 4 #3: tuned on the DEV half only (questions 1, 3, 5, ...). The other half is
    untouched, so judge the cut-off on it: a cut-off tuned on the questions you score
    makes the results look better than they are."""
    dev = details[0::2]
    real = [d["score"] for d in dev if d["case"]["type"] != "no_answer" and d.get("score") is not None]
    none = [d["score"] for d in dev if d["case"]["type"] == "no_answer" and d.get("score") is not None]
    if not real or not none:
        return None
    lowest_real, highest_none = min(real), max(none)
    if highest_none < lowest_real:
        cut = (highest_none + lowest_real) / 2
        held = details[1::2]                                       # never used for tuning
        ok = sum(1 for d in held if d.get("score") is not None
                 and (d["score"] >= cut) == (d["case"]["type"] != "no_answer"))
        return (f"  Suggested MIN_RELEVANCE  : {cut:.2f}  (tuned on the dev half: no_answer max "
                f"{highest_none:.2f} < real min {lowest_real:.2f})\n"
                f"  ...on the other half     : {ok}/{len(held)} questions on the right side of {cut:.2f}")
    return (f"  MIN_RELEVANCE advice     : no clean cut (no_answer max {highest_none:.2f}, "
            f"real min {lowest_real:.2f}). Keep 0 and let the LLM refuse.")


def score(details):
    answerable = [d for d in details if d["case"]["type"] != "no_answer"]
    no_answer = [d for d in details if d["case"]["type"] == "no_answer"]
    s = {"answerable": len(answerable), "no_answer": len(no_answer)}
    if answerable:
        for k in (1, 3):
            s[f"hit@{k}"] = sum(1 for d in answerable if d["rank"] and d["rank"] <= k) / len(answerable)
        s["hit_rate"] = sum(1 for d in answerable if d["rank"]) / len(answerable)   # hit@EVAL_K
        s["mrr"] = sum(1 / d["rank"] for d in answerable if d["rank"]) / len(answerable)
        s["doc_hit_rate"] = sum(1 for d in answerable if d["doc_rank"]) / len(answerable)
        s["by_type"] = {}
        for t in sorted({d["case"]["type"] for d in answerable}):
            group = [d for d in answerable if d["case"]["type"] == t]
            s["by_type"][t] = (sum(1 for d in group if d["rank"]), len(group))
    if no_answer:
        s["no_answer_acc"] = sum(1 for d in no_answer if d["refused"]) / len(no_answer)
    return s


# ---------------------------------------------------------------------------
# 4. RAGAS: faithfulness + answer relevance (answerable questions only)
# ---------------------------------------------------------------------------

def score_ragas(details):
    from ragas import evaluate, EvaluationDataset, RunConfig
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    samples = [{"user_input": d["case"]["question"], "response": d["answer"],
                "retrieved_contexts": [doc.page_content for doc in d["used"]],
                "reference": d["case"]["answer"]}
               for d in details if d["case"]["type"] != "no_answer" and d["docs"]]
    result = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=LangchainLLMWrapper(llm),                       # Day 4 #4: the SAME model grades its answers
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        run_config=RunConfig(max_workers=2, timeout=300),   # gentle: avoids port errors
    )
    df = result.to_pandas()
    return df["faithfulness"].mean(), df["answer_relevancy"].mean(), df


# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------

def save_details(label, details, ragas_df=None):
    path = Path(f"eval_details_{label.replace('+', '_')}.csv")
    ragas_rows = iter(ragas_df.to_dict("records")) if ragas_df is not None else None
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "question", "expected_file", "expected_pages", "result",
                    "rank", "retrieved (file p.page)", "bot_answer", "expected_answer",
                    "faithfulness", "answer_relevance", "top_relevance_score"])
        for d in details:
            c = d["case"]
            got = " | ".join(f"{doc.metadata.get('source')} p.{doc.metadata.get('page')}" for doc in d["docs"])
            if c["type"] == "no_answer":
                result, rank, faith, rel = ("refused OK" if d["refused"] else "ANSWERED (wrong)"), "", "", ""
            else:
                result = "hit" if d["rank"] else "doc only" if d["doc_rank"] else "miss"
                rank = d["rank"] or ""
                faith = rel = ""
                if ragas_rows is not None and d["docs"]:
                    r = next(ragas_rows, {})
                    faith, rel = r.get("faithfulness", ""), r.get("answer_relevancy", "")
            w.writerow([c["id"], c["type"], c["question"], c["file"],
                        ", ".join(map(str, sorted(c["pages"]))), result, rank, got,
                        d["answer"], c["answer"], faith, rel,
                        f"{d['score']:.3f}" if d.get("score") is not None else ""])
    return path


SUMMARY_COLUMNS = ["time", "setup", "questions", "hit@1", "hit@3", "hit@k", "mrr@k", "doc_hit@k",
                   "no_answer_acc", "faithfulness", "answer_relevance", "ragas_judge"]


def save_summary(label, s, settings, faithfulness="", relevance=""):
    """Day 4 #2: one row per run, WITH its settings, so runs can be told apart later."""
    header = SUMMARY_COLUMNS + list(settings)
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open(encoding="utf-8-sig") as f:
            old_header = next(csv.reader(f), [])
        if old_header != header:                       # old format: keep it, start a new file
            old = RESULTS_FILE.with_name(f"eval_results_old_{datetime.now():%Y%m%d_%H%M}.csv")
            RESULTS_FILE.rename(old)
            print(f"(old eval_results.csv had other columns: kept as {old.name})")
    new_file = not RESULTS_FILE.exists()
    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else x
    with RESULTS_FILE.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), label,
                    s["answerable"] + s["no_answer"], fmt(s.get("hit@1", "")), fmt(s.get("hit@3", "")),
                    fmt(s.get("hit_rate", "")), fmt(s.get("mrr", "")), fmt(s.get("doc_hit_rate", "")),
                    fmt(s.get("no_answer_acc", "")), fmt(faithfulness), fmt(relevance),
                    settings["chat_model"] if faithfulness != "" else ""]
                   + [settings[c] for c in settings])


def write_report():
    """Day 5: the evidence. eval_report.md = every run, and the change against the baseline
    (same strategy, no hybrid / rerank / rewrite) of the same k and test file."""
    if not RESULTS_FILE.exists():
        sys.exit("No eval_results.csv yet: run eval.py first.")
    with RESULTS_FILE.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or "hit@k" not in rows[0]:
        sys.exit("eval_results.csv is in the old format: run the evals again with this eval.py.")
    latest = {}
    for r in rows:                                     # last run of each setup wins
        latest[(r["setup"], r["eval_k"], r["test_file_md5"])] = r

    def base_of(r):
        for b in latest.values():
            if (b["strategy"] == r["strategy"] and b["eval_k"] == r["eval_k"]
                    and b["test_file_md5"] == r["test_file_md5"] and b["scope"] == r["scope"]
                    and b["hybrid"] == b["rerank"] == b["rewrite"] == "False"
                    and b["min_relevance"] in ("0", "0.0")):
                return b
        return None

    def cell(r, col, base):
        v = r.get(col, "")
        if not v or base is None or base is r or not base.get(col):
            return v
        d = float(v) - float(base[col])
        return f"{v} ({d:+.3f})"

    k = rows[-1]["eval_k"]
    lines = ["# Evaluation results", "",
             f"All setups scored at the same k = {k} on `{rows[-1]['test_file']}` "
             f"(md5 {rows[-1]['test_file_md5']}). In brackets: change vs. the baseline of the same "
             f"chunking strategy (no hybrid / re-rank / rewrite, no cut-off).", "",
             "| Setup | Hit@1 | Hit@3 | Hit@k | MRR@k | Doc hit@k | No-answer refused | Faithfulness | Answer relevance |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(latest.values(), key=lambda r: (r["strategy"], r["setup"])):
        b = base_of(r)
        lines.append("| " + " | ".join([r["setup"]] + [cell(r, c, b) for c in
                     ("hit@1", "hit@3", "hit@k", "mrr@k", "doc_hit@k", "no_answer_acc",
                      "faithfulness", "answer_relevance")]) + " |")
    judges = {r["ragas_judge"] for r in latest.values() if r.get("ragas_judge")}
    if judges:
        lines += ["", f"Note: faithfulness and answer relevance were graded by {', '.join(sorted(judges))}, "
                      "the same model that writes the answers. A small model grading itself can be "
                      "too generous, so treat these two scores as rough."]
    lines += ["", "Settings of each run (chunk size, models, cut-off, test file) are in eval_results.csv."]
    Path("eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nSaved: eval_report.md")


if __name__ == "__main__":
    if "--report" in sys.argv:
        write_report()
        sys.exit()
    cases = load_cases()
    label = run_label()
    print(f"Setup: {label}\n")

    vector_db = get_vector_db()
    settings = run_settings(vector_db)
    details = run_questions(vector_db, cases)
    s = score(details)

    faithfulness = relevance = ""
    ragas_df = None
    if "--ragas" in sys.argv:
        print("\nRAGAS: grading answers (slow)...")
        faithfulness, relevance, ragas_df = score_ragas(details)

    print("\n" + "=" * 55)
    print(f"  Setup                  : {label}")
    print(f"  Scored at k            : {EVAL_K} for every setup (the chatbot answers from {answer_k()})")
    if "hit_rate" in s:
        print(f"  Hit@1 / Hit@3 (page)   : {s['hit@1']:.3f} / {s['hit@3']:.3f}")
        print(f"  Hit@{EVAL_K} (page)           : {s['hit_rate']:.3f}   ({s['answerable']} questions)")
        print(f"  MRR@{EVAL_K}                  : {s['mrr']:.3f}")
        print(f"  Right document @{EVAL_K}      : {s['doc_hit_rate']:.3f}")
        for t, (hits, total) in s["by_type"].items():
            print(f"     {t:<20}: {hits}/{total}")
    if "no_answer_acc" in s:
        print(f"  No-answer refused      : {s['no_answer_acc']:.3f}   ({s['no_answer']} questions)")
    if faithfulness != "":
        print(f"  Faithfulness           : {faithfulness:.3f}")
        print(f"  Answer relevance       : {relevance:.3f}")
        print(f"  (graded by {settings['chat_model']}, the same model that wrote the answers:"
              f" treat as rough)")
    advice = suggest_cutoff(details)
    if advice:
        print(advice + (f"   [now: {MIN_RELEVANCE}]" if MIN_RELEVANCE else ""))
    print("=" * 55)

    save_summary(label, s, settings, faithfulness, relevance)
    details_file = save_details(label, details, ragas_df)
    print(f"Saved: {RESULTS_FILE} (summary) and {details_file} (every question)")