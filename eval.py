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

Every run:
  - adds one row to eval_results.csv            (compare before / after)
  - writes eval_details_<setup>.csv             (every question: hit, rank, answer)
"""

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import app
from app import get_vector_db, llm, embeddings, CHUNK_STRATEGY

# Works with your app.py as it is: uses its functions when they exist,
# otherwise does the same simple search + prompt itself.
USE_HYBRID = getattr(app, "USE_HYBRID", False)
USE_RERANK = getattr(app, "USE_RERANK", False)
USE_REWRITE = getattr(app, "USE_REWRITE", False)
MIN_RELEVANCE = getattr(app, "MIN_RELEVANCE", 0.0)


def retrieve(vector_db, question):
    if hasattr(app, "retrieve"):
        return app.retrieve(vector_db, question)
    return vector_db.similarity_search(question, k=3, filter={"is_latest": True})


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
    return "+".join(parts)


# ---------------------------------------------------------------------------
# 3. Run every question
# ---------------------------------------------------------------------------

def run_questions(vector_db, cases):
    details = []
    for n, case in enumerate(cases, start=1):
        docs = retrieve(vector_db, case["question"])
        answer = generate_answer(case["question"], docs) if docs else "I could not find this in the documents."
        row = {"case": case, "docs": docs, "answer": answer}
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
    """MIN_RELEVANCE advice: real questions should score ABOVE it, no_answer ones BELOW."""
    real = [d["score"] for d in details if d["case"]["type"] != "no_answer" and d.get("score") is not None]
    none = [d["score"] for d in details if d["case"]["type"] == "no_answer" and d.get("score") is not None]
    if not real or not none:
        return None
    lowest_real, highest_none = min(real), max(none)
    if highest_none < lowest_real:
        return (f"  Suggested MIN_RELEVANCE  : {(highest_none + lowest_real) / 2:.2f}  "
                f"(no_answer max {highest_none:.2f} < real min {lowest_real:.2f}: clean cut)")
    return (f"  MIN_RELEVANCE advice     : no clean cut (no_answer max {highest_none:.2f}, "
            f"real min {lowest_real:.2f}). Keep 0 and let the LLM refuse.")


def score(details):
    answerable = [d for d in details if d["case"]["type"] != "no_answer"]
    no_answer = [d for d in details if d["case"]["type"] == "no_answer"]
    s = {"answerable": len(answerable), "no_answer": len(no_answer)}
    if answerable:
        s["hit_rate"] = sum(1 for d in answerable if d["rank"]) / len(answerable)
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
                "retrieved_contexts": [doc.page_content for doc in d["docs"]],
                "reference": d["case"]["answer"]}
               for d in details if d["case"]["type"] != "no_answer" and d["docs"]]
    result = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=LangchainLLMWrapper(llm),                       # your CHAT_MODEL grades the answers
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


def save_summary(label, s, faithfulness="", relevance=""):
    new_file = not RESULTS_FILE.exists()
    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else x
    with RESULTS_FILE.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time", "setup", "questions", "hit_rate", "mrr", "doc_hit_rate",
                        "no_answer_acc", "faithfulness", "answer_relevance"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), label,
                    s["answerable"] + s["no_answer"], fmt(s.get("hit_rate", "")),
                    fmt(s.get("mrr", "")), fmt(s.get("doc_hit_rate", "")),
                    fmt(s.get("no_answer_acc", "")), fmt(faithfulness), fmt(relevance)])


if __name__ == "__main__":
    cases = load_cases()
    label = run_label()
    print(f"Setup: {label}\n")

    vector_db = get_vector_db()
    details = run_questions(vector_db, cases)
    s = score(details)

    faithfulness = relevance = ""
    ragas_df = None
    if "--ragas" in sys.argv:
        print("\nRAGAS: grading answers (slow)...")
        faithfulness, relevance, ragas_df = score_ragas(details)

    print("\n" + "=" * 55)
    print(f"  Setup                  : {label}")
    if "hit_rate" in s:
        print(f"  Hit-rate @3 (page)     : {s['hit_rate']:.3f}   ({s['answerable']} questions)")
        print(f"  MRR                    : {s['mrr']:.3f}")
        print(f"  Right document found   : {s['doc_hit_rate']:.3f}")
        for t, (hits, total) in s["by_type"].items():
            print(f"     {t:<20}: {hits}/{total}")
    if "no_answer_acc" in s:
        print(f"  No-answer refused      : {s['no_answer_acc']:.3f}   ({s['no_answer']} questions)")
    if faithfulness != "":
        print(f"  Faithfulness           : {faithfulness:.3f}")
        print(f"  Answer relevance       : {relevance:.3f}")
    advice = suggest_cutoff(details)
    if advice:
        print(advice + (f"   [now: {MIN_RELEVANCE}]" if MIN_RELEVANCE else ""))
    print("=" * 55)

    save_summary(label, s, faithfulness, relevance)
    details_file = save_details(label, details, ragas_df)
    print(f"Saved: {RESULTS_FILE} (summary) and {details_file} (every question)")