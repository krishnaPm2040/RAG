"""
tests/test_logic.py - fast tests of the pure-Python logic (no Ollama, no PDFs needed).

    pip install pytest
    python -m pytest tests -q

Each test names the review item it checks.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0]]                       # pytest's own flags must not reach app.py

import app                                     # noqa: E402
import clean_pdfs                              # noqa: E402
from langchain_core.documents import Document  # noqa: E402


# ---------------------------------------------------------------------------
# H3: supersede matching must compare circular numbers EXACTLY
# ---------------------------------------------------------------------------

def _write(folder, name, **meta):
    base = {"doc_type": "rbi_circular", "issue_date": "2025-01-01", "supersedes": None,
            "circular_no": None}
    base.update(meta)
    (folder / f"{name}.json").write_text(json.dumps({"file_name": f"{name}.pdf", "metadata": base}),
                                         encoding="utf-8")


def _latest(folder):
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))["metadata"]["is_latest"]
            for p in folder.glob("*.json")}


def test_h3_rbi_2024_25_1_is_not_superseded_by_a_reference_to_12(tmp_path):
    _write(tmp_path, "c1", circular_no="RBI/2024-25/1")
    _write(tmp_path, "c12", circular_no="RBI/2024-25/12")
    _write(tmp_path, "new", circular_no="RBI/2025-26/99", issue_date="2025-06-01",
           supersedes="RBI/2024-25/12")
    clean_pdfs.link_superseded(tmp_path)
    assert _latest(tmp_path) == {"c1": True, "c12": False, "new": True}


def test_h3_department_numbers_are_exact_too(tmp_path):
    _write(tmp_path, "a", circular_no="DBR.BP.BC.No.4/21.04.048/2024-25")
    _write(tmp_path, "b", circular_no="DBR.BP.BC.No.45/21.04.048/2024-25")
    _write(tmp_path, "new", circular_no="RBI/2025-26/7", issue_date="2025-06-01",
           supersedes="DBR.BP.BC.No.45/21.04.048/2024-25")
    clean_pdfs.link_superseded(tmp_path)
    assert _latest(tmp_path) == {"a": True, "b": False, "new": True}


def test_h3_several_references_each_match(tmp_path):
    _write(tmp_path, "x", circular_no="RBI/2023-24/5")
    _write(tmp_path, "y", circular_no="RBI/2023-24/50")
    _write(tmp_path, "new", circular_no="RBI/2025-26/1", issue_date="2025-06-01",
           supersedes="RBI/2023-24/5; RBI/2023-24/50")
    clean_pdfs.link_superseded(tmp_path)
    assert _latest(tmp_path) == {"x": False, "y": False, "new": True}


# ---------------------------------------------------------------------------
# M4 and round 3 #16: when are old (withdrawn / superseded) circulars searched?
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "Is the Rs 2000 note withdrawn?",                                   # M4: banknote topic
    "Which Rs 2000 notes were withdrawn from circulation?",
    "What is the credit history requirement?",
    "What was the previous policy on KYC?",
    "Can I make a premature withdrawal of a term deposit?",
])
def test_m4_ordinary_questions_search_only_current_documents(question):
    assert not app.OLD_VERSION_HINT.search(question)


@pytest.mark.parametrize("question", [
    "What did the withdrawn circular on customer protection say?",
    "Under the withdrawn Master Direction on Interest Rate on Deposits, 2016, ...?",
    "Under the withdrawn KYC Amendment Directions, 2025, how many reminders ...?",
    "What does RBI/2024-25/12 say?",
    "What did the 2019 circular on KYC say?",
    "Show the superseded version of the KYC direction",
    "What did the earlier circular say?",
])
def test_old_documents_are_searched_when_the_question_asks_for_them(question):
    assert app.OLD_VERSION_HINT.search(question)


# ---------------------------------------------------------------------------
# Round 3 #4: follow-up gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "What is the limit that applies to small accounts?",
    "Is there a fee for exchanging Rs 2000 notes?",
    "What are the documents mentioned in the KYC direction?",
    "What is the due date for the CPIS survey?",
])
def test_complete_questions_are_not_rewritten(question):
    assert not app.looks_like_follow_up(question)


@pytest.mark.parametrize("question", [
    "and for rural accounts?", "what is its limit?", "Does it apply to NRIs?", "why?",
])
def test_real_follow_ups_are_rewritten(question):
    assert app.looks_like_follow_up(question)


# ---------------------------------------------------------------------------
# Citations, sources, per-file limit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("answer,valid,invalid", [
    ("A [1]. B [2][3].", {1, 2, 3}, []),
    ("A [1, 2].", {1, 2}, []),
    ("A [1-3].", {1, 2, 3}, []),
    ("A [6].", set(), [6]),
])
def test_citation_parser(answer, valid, invalid):
    assert app.check_citations(answer, 5) == (valid, invalid)


def _doc(source, page=1, text="x"):
    return Document(page_content=text, metadata={"source": source, "page": page, "page_end": page})


def test_l2_a_refusal_shows_no_sources():
    results = [_doc("a.pdf"), _doc("b.pdf")]
    assert app.sources_to_show("I could not find this in the documents.", results) == []


def test_sources_are_only_the_cited_ones():
    results = [_doc("a.pdf"), _doc("b.pdf"), _doc("c.pdf")]
    shown = app.sources_to_show("Rs 20,000 [2].", results)
    assert [n for n, _ in shown] == [2]


def test_uncited_answer_shows_all_sources():
    results = [_doc("a.pdf"), _doc("b.pdf")]
    assert [n for n, _ in app.sources_to_show("Rs 20,000.", results)] == [1, 2]


def test_per_file_limit_keeps_three_chunks():
    docs = [_doc("a.pdf", p) for p in range(5)] + [_doc("b.pdf")]
    kept = app.limit_per_source(docs, 3)
    assert [d.metadata["source"] for d in kept] == ["a.pdf"] * 3 + ["b.pdf"]


# ---------------------------------------------------------------------------
# BM25 tokenizer (round 1 #9, round 3 #20)
# ---------------------------------------------------------------------------

def test_tokenizer_keeps_circular_numbers_whole():
    tokens = app.tokenize("See RBI/2024-25/12 for details.")
    assert "rbi/2024-25/12" in tokens and "rbi" in tokens


def test_tokenizer_ignores_case_and_punctuation():
    assert app.tokenize("Passport?") == app.tokenize("passport")


# ---------------------------------------------------------------------------
# L6: chunk IDs depend on the content, not on the chunk's position
# ---------------------------------------------------------------------------

def _sections():
    meta = {"source": "a.pdf", "page": 1, "page_end": 1, "heading": "H", "doc_type": "rbi_circular"}
    return [Document(page_content=f"Section {i} " + "word " * 60, metadata=dict(meta, page=i))
            for i in range(1, 6)]


def test_l6_chunk_ids_do_not_shift_when_an_earlier_chunk_is_removed():
    before = {c.metadata["page"]: c.metadata["chunk_id"] for c in app.create_chunks(_sections(), "recursive")}
    after = {c.metadata["page"]: c.metadata["chunk_id"] for c in app.create_chunks(_sections()[1:], "recursive")}
    assert all(before[p] == after[p] for p in (2, 3, 4, 5))


def test_l6_chunk_ids_are_unique_even_for_repeated_text():
    meta = {"source": "a.pdf", "page": 1, "page_end": 1, "heading": "H", "doc_type": "rbi_circular"}
    same = [Document(page_content="Same text " + "word " * 60, metadata=dict(meta)) for _ in range(3)]
    ids = [c.metadata["chunk_id"] for c in app.create_chunks(same, "recursive")]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# H1: the tuned extras (synonyms, stemming, DOC_FOCUS) can be switched off for the ablation
# ---------------------------------------------------------------------------

def test_h1_synonyms_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(app, "USE_SYNONYMS", False)
    assert app.expand_query(["penalty"]) == ["penalty"]
    monkeypatch.setattr(app, "USE_SYNONYMS", True)
    assert "fine" in app.expand_query(["penalty"])


def test_h1_stemming_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(app, "USE_STEMMING", False)
    assert "categories" in app.tokenize("categories")
    monkeypatch.setattr(app, "USE_STEMMING", True)
    assert "category" in app.tokenize("categories")


# ---------------------------------------------------------------------------
# M3: an updated master direction is dated by its latest update
# ---------------------------------------------------------------------------

def test_m3_updated_on_is_found():
    head = ("RBI/FMRD/2016-17/31\nFMRD Master Direction No. 1/2016-17   July 5, 2016\n"
            "(Updated as on October 01, 2026)\n(Updated as on September 22, 2026)\n")
    meta = clean_pdfs.extract_version_metadata(head)
    assert meta["issue_date"] == "2016-07-05"
    assert meta["updated_on"] == "2026-10-01"


# ---------------------------------------------------------------------------
# L1: no "annual reports only" choice when annual reports are not in the index
# ---------------------------------------------------------------------------

def test_l1_ui_has_no_annual_report_option():
    ui = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Annual reports only" not in ui


# ---------------------------------------------------------------------------
# Final-review resubmission findings (N1, N3, N4, N6, N9)
# ---------------------------------------------------------------------------

import eval as ev                              # noqa: E402


def test_n6_number_words_count_as_key_facts():
    assert "3" in ev.key_facts("At least three advance intimations before the due date.")
    assert ev.fact_score("three reminders", "The RE must send 3 reminders [1].") == (1, 1)
    assert ev.fact_score("three reminders", "The RE must send three reminders [1].") == (1, 1)


def test_n4_include_old_drops_the_latest_filter():
    q = "How much is a customer liable for if a fraud is reported on the 5th working day?"
    assert app.search_filter(q) == {"is_latest": True}
    assert app.search_filter(q, include_old=True) is None


def _old_doc(latest=True, text="x"):
    return Document(page_content=text, metadata={"source": "a.pdf", "page": 1, "is_latest": latest})


def test_n4_fallback_uses_old_versions_only_after_a_refusal(monkeypatch):
    answers = iter(["I could not find this in the documents.", "Within 10 working days [1]."])
    monkeypatch.setattr(app, "generate_answer", lambda q, r, on_token=None: next(answers))
    monkeypatch.setattr(app, "retrieve", lambda db, q, include_old=False, **kw: [_old_doc(latest=False)])
    answer, used, old = app.answer_with_fallback(None, "within how many days must a bank credit back?",
                                                 [_old_doc()])
    assert old is True and "10 working days" in answer and used[0].metadata["is_latest"] is False


def test_n4_no_fallback_when_no_old_document_is_found(monkeypatch):
    monkeypatch.setattr(app, "generate_answer", lambda q, r, on_token=None: "I could not find this in the documents.")
    monkeypatch.setattr(app, "retrieve", lambda db, q, include_old=False, **kw: [_old_doc(latest=True)])
    answer, used, old = app.answer_with_fallback(None, "What is the capital of Australia?", [_old_doc()])
    assert old is False and app.is_refusal(answer)


def test_n4_no_fallback_for_a_real_answer(monkeypatch):
    monkeypatch.setattr(app, "generate_answer", lambda q, r, on_token=None: "Rs 20,000 [1].")
    monkeypatch.setattr(app, "retrieve", lambda *a, **k: pytest.fail("must not search again"))
    assert app.answer_with_fallback(None, "limit?", [_old_doc()])[2] is False


def test_n9_fingerprint_is_not_recomputed_when_nothing_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "JSON_FOLDER", tmp_path)
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    calls = []
    real = app._data_fingerprint
    monkeypatch.setattr(app, "_data_fingerprint", lambda s, f: calls.append(1) or real(s, f))
    app._fingerprints.clear()
    first = app.data_fingerprint("recursive")
    assert app.data_fingerprint("recursive") == first and len(calls) == 1
    (tmp_path / "a.json").write_text('{"changed": 1}', encoding="utf-8")
    assert app.data_fingerprint("recursive") != first and len(calls) == 2


def test_n1_floor_stops_when_pdfs_are_missing(tmp_path, monkeypatch):
    (tmp_path / "json").mkdir()
    (tmp_path / "pdfs").mkdir()
    for name in ("a.pdf", "b.pdf"):
        (tmp_path / "json" / f"{name}.json").write_text(json.dumps(
            {"file_name": name, "file_hash": "abc", "metadata": {"doc_type": "rbi_circular"}}), encoding="utf-8")
    monkeypatch.setattr(app, "JSON_FOLDER", tmp_path / "json")
    monkeypatch.setattr(ev, "PDF_FOLDER", tmp_path / "pdfs")
    with pytest.raises(SystemExit, match="2 of 2 PDFs"):
        ev.page_texts()


def test_results_file_gets_new_columns_without_losing_rows(tmp_path, monkeypatch):
    path = tmp_path / "eval_results.csv"
    path.write_text("time,setup,hit@1\n2026-11-23 10:00,recursive,0.725\n", encoding="utf-8-sig")
    monkeypatch.setattr(ev, "RESULTS_FILE", path)
    s = {"answerable": 1, "no_answer": 0, "hit@1": 1.0, "sec_per_q": 1.0, "answered_on_miss": "0/0"}
    ev.save_summary("recursive+hybrid", s, {"chat_model": "ollama:qwen3:8b", "test_file": "t.csv"})
    import csv as _csv
    rows = list(_csv.DictReader(path.open(encoding="utf-8-sig")))
    assert [r["setup"] for r in rows] == ["recursive", "recursive+hybrid"]
    assert rows[0]["hit@1"] == "0.725" and rows[1]["answered_on_miss"] == "0/0"