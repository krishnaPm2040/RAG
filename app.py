"""
app.py - the ENGINE of the RAG chatbot: builds and stores the chunks. No chat here.
The chat is the web UI:  streamlit run streamlit_app.py

  1. python clean_pdfs.py            -> output/json/*.json
  2. ollama pull nomic-embed-text    -> embedding model
     ollama pull qwen2.5:7b          -> chat model (runs on your PC)
  3. python app.py                   -> builds / checks ALL 3 databases (recursive, fixed, semantic)
     python app.py --strategy=fixed  -> only one strategy
     python app.py --rebuild         -> force a rebuild
  4. streamlit run streamlit_app.py  -> the chat in your browser

A database is rebuilt AUTOMATICALLY when the JSON files or the chunk settings
change, or when the last build did not finish.

Older (superseded) circulars are hidden by default. They are included when the
question names a circular number, a year with "circular/direction", or words
like "old", "previous", "superseded".
"""

import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


# ---------------------------------------------------------------------------
# Settings (everything you may want to change is here)
# ---------------------------------------------------------------------------

JSON_FOLDER = Path("output/json")
DB_FOLDER = "vector_db/chroma_db"
COLLECTION = "rbi_documents"

DOMAIN = "RBI circulars"                        # used in all prompts: change it in ONE place

CHUNK_SIZE = 1000                               # characters (about 200 words)
CHUNK_OVERLAP = 150
SEMANTIC_MAX_CHARS = 2000                       # semantic chunks bigger than this are split again
SEMANTIC_PERCENTILE = 95                        # cut where the topic change is in the top 5%
DB_VERSION = "v1"                               # raise (v2, v3...) when you change the chunking CODE

EMBED_MODEL = "nomic-embed-text"                # local; first run: ollama pull nomic-embed-text
CHAT_MODEL = "qwen2.5:7b"                       # on your PC, free. A cloud model also works here
                                                # (e.g. "gpt-oss:120b-cloud"), through local Ollama
BATCH_SIZE = 50                                 # chunks embedded at a time

FETCH_K = 20                                    # candidates fetched before merge / re-rank
MIN_RELEVANCE = 0.0                             # 0 = off. eval.py prints a suggested value
SYSTEM_PROMPT = f"""You are a helpful assistant that answers questions about {DOMAIN}.

RULES
1. Use ONLY the numbered context passages given with each question. Never use outside
   knowledge, and never guess rules, limits, dates, amounts or circular numbers.
2. If the context does not contain the answer, reply exactly:
   "I could not find this in the documents."
   Do not add anything else in that case.
3. If the user only greets you or says thanks (for example "hi", "hello", "thank you"),
   reply briefly and politely, and invite them to ask a question about {DOMAIN}.
   Do not say "I could not find this in the documents" for a greeting.
4. Cite every fact with the number of the passage it came from, in square brackets,
   right after the sentence, like [1] or [2]. Use only numbers that exist in the context.
   If a sentence uses two passages, cite both: [1][3].
5. Quote amounts, limits, dates, time periods and circular numbers exactly as written
   in the context (for example "Rs 20,000", "90 days", "RBI/2024-25/12").
6. If passages disagree, say so, and prefer the newest circular (the latest issue date).
   If a passage is from an older or superseded circular, say that it is old.
7. Keep answers short and clear: first the direct answer in one or two sentences,
   then any conditions or exceptions as short bullet points.
8. Answer in the same language as the question."""

HISTORY_TURNS = 3                               # earlier Q&A used to understand follow-ups

# Defaults from the command line (Streamlit / eval can pass their own per call)
CHUNK_STRATEGY = "recursive"
for arg in sys.argv:
    if arg.startswith("--strategy="):
        CHUNK_STRATEGY = arg.split("=", 1)[1]
USE_HYBRID = "--hybrid" in sys.argv
USE_RERANK = "--rerank" in sys.argv
USE_REWRITE = "--rewrite" in sys.argv
TOP_K = 5 if USE_RERANK else 3                  # chunks given to the LLM


def top_k_for(rerank):
    return 5 if rerank else 3


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class NomicEmbeddings(OllamaEmbeddings):
    """nomic-embed-text works best when told what each text is."""
    def embed_documents(self, texts):
        return super().embed_documents(["search_document: " + t for t in texts])

    def embed_query(self, text):
        return super().embed_query("search_query: " + text)


embeddings = NomicEmbeddings(model=EMBED_MODEL)            # local Ollama (localhost:11434)

llm = ChatOllama(
    model=CHAT_MODEL,
    base_url="http://localhost:11434",          # local Ollama (uses your device key for cloud models)
    temperature=0,
)


# ---------------------------------------------------------------------------
# 1. Load the JSON files: one Document per section
# ---------------------------------------------------------------------------

def load_documents():
    documents = []
    files = sorted(JSON_FOLDER.glob("*.json"))
    for json_file in files:
        data = json.loads(json_file.read_text(encoding="utf-8"))
        meta = data.get("metadata", {})

        for section in data["sections"]:
            if section["type"] == "heading":                 # already in heading_path
                continue
            if section["type"] == "table" and section.get("rows_text"):
                body = "\n".join(section["rows_text"])       # tables as sentences
            else:
                body = section["text"]

            # footnote text goes WITH the section; the bare [^1] markers go away
            body = re.sub(r"\[\^[^\]]+\]:?", "", body)
            notes = [f.get("text", "") for f in section.get("footnotes", []) if isinstance(f, dict)]
            if notes:
                body += "\nNotes: " + " ".join(n for n in notes if n)

            documents.append(Document(
                page_content=body,                           # heading is added to EVERY chunk later
                metadata={                                   # no None values for Chroma
                    "source": data["file_name"],
                    "title": data.get("title", "") or "",
                    "page": section["page"],
                    "page_end": section.get("page_end") or section["page"],
                    "heading": section.get("heading_path", "") or "",
                    "part": section.get("part") or "main",   # main / annex / note
                    "lang": section.get("lang") or "",
                    "section_type": section.get("type") or "",
                    "doc_type": meta.get("doc_type") or "",
                    "model": meta.get("model") or "",
                    "product": meta.get("product") or "",
                    "brand": meta.get("brand") or "",
                    "circular_no": meta.get("circular_no") or "",
                    "issue_date": meta.get("issue_date") or "",
                    "is_latest": meta.get("is_latest", True),
                },
            ))

    print(f"Sections loaded: {len(documents)} from {len(files)} files")
    return documents


# ---------------------------------------------------------------------------
# 2. Chunking: fixed / recursive / semantic (own semantic chunker, no
#    langchain-experimental needed)
# ---------------------------------------------------------------------------

SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+|\n+")


def cosine_distance(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 1 - dot / (na * nb) if na and nb else 1.0


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * pct / 100
    low, high = math.floor(pos), math.ceil(pos)
    return values[low] + (values[high] - values[low]) * (pos - low)


def semantic_chunks(documents):
    """Cut each section where the meaning changes most between two sentences.
    Never makes a chunk bigger than SEMANTIC_MAX_CHARS."""
    jobs = []                                                # (doc, sentences)
    for doc in documents:
        sentences = [s.strip() for s in SENTENCE_END.split(doc.page_content) if s.strip()]
        jobs.append((doc, sentences))

    to_embed = [s for _, sents in jobs if len(sents) > 2 for s in sents]
    vectors = []
    for i in range(0, len(to_embed), BATCH_SIZE):            # in batches, gently (Windows ports)
        vectors.extend(embeddings.embed_documents(to_embed[i:i + BATCH_SIZE]))
        print(f"  semantic: embedded {min(i + BATCH_SIZE, len(to_embed))}/{len(to_embed)} sentences",
              end="\r")
        time.sleep(0.2)
    print()

    chunks, v = [], 0
    for doc, sents in jobs:
        if len(sents) <= 2:                                  # too short to split by meaning
            chunks.append(doc)
            continue
        vecs = vectors[v:v + len(sents)]
        v += len(sents)
        dists = [cosine_distance(vecs[i], vecs[i + 1]) for i in range(len(sents) - 1)]
        cut = percentile(dists, SEMANTIC_PERCENTILE)

        group = [sents[0]]
        for i, sentence in enumerate(sents[1:]):
            too_big = len(" ".join(group)) + len(sentence) > SEMANTIC_MAX_CHARS
            if dists[i] > cut or too_big:                    # topic changed, or chunk full
                chunks.append(Document(page_content=" ".join(group), metadata=dict(doc.metadata)))
                group = []
            group.append(sentence)
        chunks.append(Document(page_content=" ".join(group), metadata=dict(doc.metadata)))

    backup = RecursiveCharacterTextSplitter(chunk_size=SEMANTIC_MAX_CHARS, chunk_overlap=CHUNK_OVERLAP)
    return [piece for c in chunks                            # a single huge sentence: split anyway
            for piece in (backup.split_documents([c]) if len(c.page_content) > SEMANTIC_MAX_CHARS else [c])]


def create_chunks(documents, strategy=None):
    strategy = strategy or CHUNK_STRATEGY
    if strategy == "fixed":
        chunks = CharacterTextSplitter(separator="", chunk_size=CHUNK_SIZE,
                                       chunk_overlap=CHUNK_OVERLAP).split_documents(documents)
    elif strategy == "recursive":
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]).split_documents(documents)
    elif strategy == "semantic":
        chunks = semantic_chunks(documents)
    else:
        sys.exit(f"Unknown strategy: {strategy}. Use fixed / recursive / semantic")

    for c in chunks:                            # EVERY chunk says which section it is from
        c.page_content = c.metadata["heading"] + "\n" + c.page_content

    print(f"Chunks created ({strategy}): {len(chunks)}")
    return chunks


# ---------------------------------------------------------------------------
# 3. Build the vector database (batches, retry, pause) + "build complete" marker
# ---------------------------------------------------------------------------

def collection_name(strategy=None):
    """Chunk settings are in the name: new settings -> new database, never a mix."""
    return f"{COLLECTION}_{strategy or CHUNK_STRATEGY}_{CHUNK_SIZE}_{DB_VERSION}"


def marker_file(strategy=None):
    """Written ONLY when a build finishes."""
    return Path(DB_FOLDER) / f"{collection_name(strategy)}.complete.json"


def data_fingerprint(strategy=None):
    """Changes when any JSON file or any chunk setting changes."""
    h = hashlib.sha256()
    settings = [strategy or CHUNK_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP,
                SEMANTIC_MAX_CHARS, DB_VERSION, EMBED_MODEL]
    h.update(json.dumps(settings).encode())
    for f in sorted(JSON_FOLDER.glob("*.json")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def build_vector_db(vector_db, chunks, strategy=None, fingerprint=None):
    marker_file(strategy).unlink(missing_ok=True)           # not complete until the end
    vector_db.reset_collection()                            # empty it first: no duplicates
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        for attempt in range(1, 6):                         # try each batch up to 5 times
            try:
                vector_db.add_documents(batch)
                break
            except Exception as e:
                print(f"\n  batch at {i} failed (try {attempt}/5): {e}")
                print("  waiting 60s for Windows to free ports...")
                time.sleep(60)
        else:
            sys.exit("Embedding keeps failing. Restart Ollama, then run again.")
        done = min(i + BATCH_SIZE, len(chunks))
        print(f"  embedded {done}/{len(chunks)} ({done * 100 // len(chunks)}%)", end="\r", flush=True)
        time.sleep(0.5)                                     # small pause: avoids port exhaustion

    Path(DB_FOLDER).mkdir(parents=True, exist_ok=True)
    marker_file(strategy).write_text(json.dumps({
        "chunks": len(chunks),
        "fingerprint": fingerprint or data_fingerprint(strategy),
        "built": time.strftime("%Y-%m-%d %H:%M"),
    }), encoding="utf-8")
    print("\nVector database created.")
    return vector_db


_open_dbs = {}


def is_built(strategy=None):
    """True if this strategy's database is complete and up to date (no build needed)."""
    marker = marker_file(strategy)
    if not marker.exists():
        return False
    info = json.loads(marker.read_text(encoding="utf-8"))
    return info.get("fingerprint") == data_fingerprint(strategy)


def get_vector_db(rebuild=False, strategy=None):
    """Open the saved database. Rebuild it when: --rebuild, the last build did
    not finish, or the documents / chunk settings changed since the last build."""
    strategy = strategy or CHUNK_STRATEGY
    vector_db = Chroma(collection_name=collection_name(strategy), embedding_function=embeddings,
                       persist_directory=DB_FOLDER,
                       collection_metadata={"hnsw:space": "cosine"})   # relevance score 0..1
    count = vector_db._collection.count()
    marker = marker_file(strategy)
    info = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
    fingerprint = data_fingerprint(strategy)

    if not rebuild and count and info.get("chunks") == count and info.get("fingerprint") == fingerprint:
        print(f"Vector database loaded: {count} chunks ({strategy})")
        _keyword_indexes.pop(collection_name(strategy), None)
        _open_dbs[collection_name(strategy)] = vector_db
        return vector_db

    if rebuild:
        print(f"--rebuild given - building '{collection_name(strategy)}'...")
    elif not count:
        print(f"No database named '{collection_name(strategy)}' yet - building it "
              f"(first time for these settings; takes a while, do NOT stop it)...")
    elif count:
        if info.get("chunks") != count:
            print(f"Vector database is INCOMPLETE ({count} chunks, last build did not finish) - rebuilding...")
        else:
            print("Documents or chunk settings changed since the last build - rebuilding...")

    documents = load_documents()
    if not documents:
        sys.exit(f"No data in {JSON_FOLDER}/. Run clean_pdfs.py first.")
    build_vector_db(vector_db, create_chunks(documents, strategy), strategy, fingerprint)
    _keyword_indexes.pop(collection_name(strategy), None)    # old BM25 index is out of date
    _open_dbs[collection_name(strategy)] = vector_db
    return vector_db


# ---------------------------------------------------------------------------
# 4. Retrieval pieces
# ---------------------------------------------------------------------------

# 4a. Old circulars: hidden unless the question asks for them
OLD_VERSION_HINT = re.compile(
    r"\bRBI/(?:[A-Za-z]{2,10}/)*\d{4}-\d{2}/\d+"                         # a circular number
    r"|\b(?:circular|direction|notification|guidelines?|master)\b[^.?]{0,40}\b(?:19|20)\d{2}\b"
    r"|\b(?:19|20)\d{2}\b[^.?]{0,20}\b(?:circular|direction|notification|guidelines?|master)\b"
    r"|\b(?:old|older|previous|earlier|superseded|withdrawn|repealed|history)\b", re.I)


def search_filter(question):
    return None if OLD_VERSION_HINT.search(question) else {"is_latest": True}


# 4b. Follow-up questions -> one standalone question
def make_standalone(question, history):
    """'and for rural accounts?' + earlier turns -> 'What is the KYC limit for rural accounts?'"""
    if not history:
        return question
    convo = "\n".join(f"User: {q}\nAssistant: {a[:400]}" for q, a in history[-HISTORY_TURNS:])
    messages = [
        ("system", "Rewrite the user's LAST question as one standalone question, using the "
                   "conversation for missing details. Keep numbers, circular numbers and names "
                   "exactly. If it is already standalone, return it unchanged. "
                   "Return ONLY the question."),
        ("human", f"Conversation:\n{convo}\n\nLast question: {question}"),
    ]
    standalone = llm.invoke(messages).content.strip()
    return standalone or question


# 4c. Query rewriting (search only; the answer uses the real question)
def rewrite_query(question):
    messages = [
        ("system", f"Rewrite the user's question as one clear, complete search query "
                   f"for {DOMAIN}. Keep circular numbers, section numbers and "
                   f"amounts exactly as written. Return ONLY the rewritten query."),
        ("human", question),
    ]
    rewritten = llm.invoke(messages).content.strip()
    return rewritten or question


# 4d. BM25 keyword search (own small index on rank_bm25; no langchain-community)
def tokenize(text):
    return re.findall(r"\w+", text.lower())                 # "Passport?" == "passport"


class KeywordIndex:
    def __init__(self, docs):
        from rank_bm25 import BM25Okapi
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in docs])

    def search(self, query, k, only_latest=True):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        hits = []
        for i in order:
            if scores[i] <= 0 or len(hits) >= k:
                break
            if only_latest and not self.docs[i].metadata.get("is_latest", True):
                continue
            hits.append(self.docs[i])
        return hits


_keyword_indexes = {}                                       # one per collection


def get_keyword_index(vector_db):
    name = vector_db._collection.name
    if name not in _keyword_indexes:
        data = vector_db.get(include=["documents", "metadatas"])
        docs = [Document(page_content=t, metadata=m)
                for t, m in zip(data["documents"], data["metadatas"])]
        _keyword_indexes[name] = KeywordIndex(docs)
        print(f"BM25 index built: {len(docs)} chunks")
    return _keyword_indexes[name]


# 4e. Merge vector + BM25 results (Reciprocal Rank Fusion)
def rrf_merge(result_lists, c=60):
    scores, docs = {}, {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:200])
            scores[key] = scores.get(key, 0) + 1 / (c + rank)
            docs[key] = doc
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


# 4f. Re-ranker
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:                                  # downloads once (~90 MB)
        from sentence_transformers import CrossEncoder
        print("Loading re-ranker...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(question, docs):
    scores = get_reranker().predict([(question, d.page_content) for d in docs])
    ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in ranked]


# ---------------------------------------------------------------------------
# 5. Retrieval: all settings can be passed per call (Streamlit users don't
#    change each other's settings)
# ---------------------------------------------------------------------------

def retrieve(vector_db, question, hybrid=None, rerank_on=None, rewrite=None):
    hybrid = USE_HYBRID if hybrid is None else hybrid
    rerank_on = USE_RERANK if rerank_on is None else rerank_on
    rewrite = USE_REWRITE if rewrite is None else rewrite
    top_k = top_k_for(rerank_on)

    query = rewrite_query(question) if rewrite else question
    if rewrite:
        print(f"  (searching for: {query})")

    where = search_filter(question)
    k = FETCH_K if (hybrid or rerank_on) else top_k
    if MIN_RELEVANCE:                                      # cut-off on: keep only close matches
        scored = vector_db.similarity_search_with_relevance_scores(query, k=k, filter=where)
        if not scored or scored[0][1] < MIN_RELEVANCE:
            return []                                      # nothing close enough: "not found"
        results = [doc for doc, _ in scored]
    else:
        results = vector_db.similarity_search(query, k=k, filter=where)

    if hybrid:
        keyword = get_keyword_index(vector_db).search(query, FETCH_K, only_latest=where is not None)
        results = rrf_merge([results, keyword])[:FETCH_K]
    if rerank_on and results:
        results = rerank(query, results)
    return results[:top_k]


def top_relevance(vector_db, question):
    """Best relevance score for a question (eval.py uses it to suggest MIN_RELEVANCE)."""
    scored = vector_db.similarity_search_with_relevance_scores(question, k=1,
                                                               filter=search_filter(question))
    return scored[0][1] if scored else 0.0


# ---------------------------------------------------------------------------
# 6. Generation with citations
# ---------------------------------------------------------------------------

def generate_answer(question, results):
    context = "\n\n".join(
        f"[{i}] ({doc.metadata['source']}, page {doc.metadata['page']})\n{doc.page_content}"
        for i, doc in enumerate(results, start=1)
    )
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    return llm.invoke(messages).content


def check_citations(answer, n_sources):
    """-> (cited numbers that exist, numbers that do not exist)"""
    used = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    valid = {n for n in used if 1 <= n <= n_sources}
    return valid, sorted(used - valid)


GREETING = re.compile(r"\s*(hi|hello|hey|hai|good (morning|afternoon|evening)|thanks?|thank you)[\s!.]*",
                      re.I)


def answer_question(vector_db, question, history=None, hybrid=None, rerank_on=None, rewrite=None):
    """Full pipeline for one chat turn (used by Streamlit and eval)."""
    if GREETING.fullmatch(question):                           # "hi", "thanks": no search needed
        return {"question_used": question, "results": [], "cited": set(), "invalid_citations": [],
                "answer": f"Hello! Ask me a question about {DOMAIN}, for example: "
                          "'What is the Rs 2000 note exchange limit?'"}
    standalone = make_standalone(question, history)
    results = retrieve(vector_db, standalone, hybrid, rerank_on, rewrite)
    if not results:
        answer = "I could not find this in the documents."
    else:
        answer = generate_answer(standalone, results)
    cited, invalid = check_citations(answer, len(results))
    return {"question_used": standalone, "results": results, "answer": answer,
            "cited": cited, "invalid_citations": invalid}


# ---------------------------------------------------------------------------
# 7. Run from the terminal: ONLY builds / checks the database (chat = Streamlit)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    chosen = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--strategy=")]
    strategies = chosen or ["recursive", "fixed", "semantic"]
    for name in strategies:
        print(f"\n=== {name} ===")
        db = get_vector_db(rebuild="--rebuild" in sys.argv, strategy=name)
        print(f"Ready: {db._collection.count()} chunks in '{collection_name(name)}'.")
    print("\nAll done. Start the chat:  streamlit run streamlit_app.py")