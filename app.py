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
ANNUAL_REPORT = re.compile(r"an(?:nu|v)al[\s_-]*report|financial[\s_-]*results", re.I)
                                                # annual reports + results ("anval" = typo in a file name)
MIN_SECTION_CHARS = 60                          # smaller sections (text boxes, "38%") join a neighbour
MAX_HEADING_PATH = 150                          # a heading path never becomes a whole paragraph
SMALL_SECTION_CHARS = 300                       # sibling sections smaller than this are merged (Day 2 #2)
MIN_CHUNK_CHARS = 100                           # a chunk with less real text joins its neighbour
INCLUDE_ANNUAL_REPORTS = False                  # False = annual reports are NOT put in the database
ANNUAL_CHUNK_SIZE = 2500                        # annual reports: bigger chunks -> far fewer chunks
MAX_PER_SOURCE = 2                              # at most 2 chunks from one file in an answer
SEMANTIC_PERCENTILE = 95                        # cut where the topic change is in the top 5% of a section
SEMANTIC_OUTLIER = 1.5                          # ...AND an outlier among ALL gaps: > Q3 + 1.5 x IQR (Day 2 #1)
SEMANTIC_MIN_GAP = 0.0                          # optional fixed floor for the gap (cosine distance)
DB_VERSION = "v4"                               # raise (v2, v3...) when you change the chunking CODE

EMBED_MODEL = "nomic-embed-text"                # local; first run: ollama pull nomic-embed-text
CHAT_MODEL = "qwen2.5:7b"                       # local chat model: ollama pull qwen2.5:7b
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
6. Each passage header shows its circular number, issue date, and "OLD VERSION" if it was
   withdrawn or superseded. If passages disagree, say so, and prefer the newest circular
   (the latest issue date).
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
SEARCH_SCOPE = "all"                            # eval.py: --scope=circulars / annual_reports
for arg in sys.argv:
    if arg.startswith("--scope="):
        SEARCH_SCOPE = arg.split("=", 1)[1]
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

llm = ChatOllama(                                # local Ollama chat model (no key, no internet)
    model=CHAT_MODEL,
    base_url="http://localhost:11434",
    temperature=0,
    keep_alive="60m",                           # keep the model in memory: no reload wait
)

THINK = re.compile(r"<think>.*?(</think>|$)", re.S)  # "thinking" text of reasoning models: never shown


def clean(text):
    """Remove the model's hidden thinking (<think>...</think>) from a reply."""
    return THINK.sub("", text or "").strip()


def ask(messages):
    """One call to the chat model -> clean text."""
    return clean(llm.invoke(messages).content)


# ---------------------------------------------------------------------------
# 1. Load the JSON files: one Document per section
# ---------------------------------------------------------------------------

NUMBER_ONLY = re.compile(r"^[\W\d_]*(?:%|crores?|cr|lakhs?|bn|mn|x|times|bps)?[\W\d_]*$", re.I)


def clean_heading(path):
    """'~14% > 36% > In 2021, under the oversight...' -> only real, short headings."""
    parts = [p.strip() for p in (path or "").split(" > ")]
    parts = [p for p in parts if p and re.search(r"[A-Za-z]{3,}", p) and not NUMBER_ONLY.match(p)]
    parts = [p if len(p) <= 150 else p[:150].rsplit(" ", 1)[0] + "..." for p in parts]
    text = " > ".join(parts)
    while len(text) > MAX_HEADING_PATH and len(parts) > 1:  # too long: keep the closest headings
        parts = parts[1:]
        text = " > ".join(parts)
    return text[:MAX_HEADING_PATH]


def join_tiny_sections(documents):
    """Sections with fewer than MIN_SECTION_CHARS characters (text boxes, big numbers)
    say nothing on their own: add them to the section before (same file), or the next one."""
    out, waiting = [], []
    for doc in documents:
        if len(doc.page_content.strip()) < MIN_SECTION_CHARS:
            prev = out[-1] if out and out[-1].metadata["source"] == doc.metadata["source"] else None
            if prev is not None:
                prev.page_content += "\n" + doc.page_content.strip()
                prev.metadata["page_end"] = max(prev.metadata["page_end"], doc.metadata["page_end"])
            else:
                waiting.append(doc)                       # first section of a file: wait for the next
            continue
        if waiting:
            same = [w for w in waiting if w.metadata["source"] == doc.metadata["source"]]
            out.extend(w for w in waiting if w.metadata["source"] != doc.metadata["source"])
            if same:
                doc.page_content = "\n".join(w.page_content.strip() for w in same) + "\n" + doc.page_content
                doc.metadata["page"] = min(doc.metadata["page"], same[0].metadata["page"])
            waiting = []
        out.append(doc)
    out.extend(waiting)                                   # a file with only tiny text: keep it
    return out


def load_documents():
    documents = []
    files = sorted(JSON_FOLDER.glob("*.json"))
    for json_file in files:
        data = json.loads(json_file.read_text(encoding="utf-8"))
        meta = data.get("metadata", {})
        is_annual = (meta.get("doc_type") == "annual_report"            # found by clean_pdfs.py
                     or bool(ANNUAL_REPORT.search(data["file_name"] + " " + (data.get("title") or ""))))
        if is_annual and not INCLUDE_ANNUAL_REPORTS:
            print(f"  skipped (annual report): {data['file_name']}")
            continue

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
                    "heading": clean_heading(section.get("heading_path", "")) or data.get("title", "") or "",
                    "part": section.get("part") or "main",   # main / annex / note
                    "lang": section.get("lang") or "",
                    "section_type": section.get("type") or "",
                    "doc_type": "annual_report" if is_annual else (meta.get("doc_type") or "rbi_circular"),
                    "model": meta.get("model") or "",
                    "product": meta.get("product") or "",
                    "brand": meta.get("brand") or "",
                    "circular_no": meta.get("circular_no") or "",
                    "issue_date": meta.get("issue_date") or "",
                    "is_latest": meta.get("is_latest", True),
                    "superseded_by": meta.get("superseded_by") or "",
                },
            ))

    before = len(documents)
    documents = join_tiny_sections(documents)
    print(f"Sections loaded: {len(documents)} from {len(files)} files "
          f"({before - len(documents)} tiny sections joined to a neighbour)")
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


def semantic_chunks(documents, max_chars=SEMANTIC_MAX_CHARS):
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

    # gaps between neighbouring sentences, per section
    all_dists, v = [], 0
    for doc, sents in jobs:
        if len(sents) <= 2:
            all_dists.append(None)
            continue
        vecs = vectors[v:v + len(sents)]
        v += len(sents)
        all_dists.append([cosine_distance(vecs[i], vecs[i + 1]) for i in range(len(sents) - 1)])

    # Minimum gap: a section's biggest gap is ALWAYS above its own 95th percentile, so on
    # its own that rule splits every section. The gap must ALSO be unusually big for the
    # whole corpus (an outlier: above Q3 + 1.5 x IQR of all gaps). A section about ONE
    # topic has only ordinary gaps, so it is not split. (Day 2 #1)
    flat = [d for ds in all_dists if ds for d in ds]
    q1, q3 = percentile(flat, 25), percentile(flat, 75)
    min_gap = max(SEMANTIC_MIN_GAP, q3 + SEMANTIC_OUTLIER * (q3 - q1))
    print(f"  semantic: split only where the gap > {min_gap:.3f} (corpus median gap {percentile(flat, 50):.3f})")

    chunks = []
    for (doc, sents), dists in zip(jobs, all_dists):
        if dists is None:                                    # too short to split by meaning
            chunks.append(doc)
            continue
        cut = max(percentile(dists, SEMANTIC_PERCENTILE), min_gap)

        group = [sents[0]]
        for i, sentence in enumerate(sents[1:]):
            too_big = len(" ".join(group)) + len(sentence) > max_chars
            if dists[i] > cut or too_big:                    # topic changed, or chunk full
                chunks.append(Document(page_content=" ".join(group), metadata=dict(doc.metadata)))
                group = []
            group.append(sentence)
        chunks.append(Document(page_content=" ".join(group), metadata=dict(doc.metadata)))

    backup = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=CHUNK_OVERLAP)
    return [piece for c in chunks                            # a single huge sentence: split anyway
            for piece in (backup.split_documents([c]) if len(c.page_content) > max_chars else [c])]


def split_group(documents, strategy, size):
    if strategy == "fixed":
        return CharacterTextSplitter(separator="", chunk_size=size,
                                     chunk_overlap=CHUNK_OVERLAP).split_documents(documents)
    if strategy == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]).split_documents(documents)
    if strategy == "semantic":
        return semantic_chunks(documents, max_chars=max(size, SEMANTIC_MAX_CHARS))
    sys.exit(f"Unknown strategy: {strategy}. Use fixed / recursive / semantic")


def merge_sections(documents, size):
    """Annual reports have many short sections: join neighbours (same file) up to `size`
    characters, so one report makes far fewer chunks. Each part keeps its own heading."""
    merged, group, length = [], [], 0
    for d in documents + [None]:
        same_file = d is not None and group and d.metadata["source"] == group[0].metadata["source"]
        if group and (d is None or not same_file or length + len(d.page_content) > size):
            first, last = group[0].metadata, group[-1].metadata
            text = "\n\n".join(g.metadata["heading"] + "\n" + g.page_content for g in group)
            meta = dict(first, heading=first["heading"] if len(group) == 1 else first["heading"].split(" > ")[0],
                        page_end=last.get("page_end", last["page"]))
            merged.append(Document(page_content=text, metadata=meta))
            group, length = [], 0
        if d is not None:
            group.append(d)
            length += len(d.page_content)
    return merged


def parent_heading(heading):
    """'Chapter 3 > 3.1 Eligibility' -> 'Chapter 3' (siblings share the parent)."""
    return heading.rsplit(" > ", 1)[0] if " > " in heading else heading


def merge_small_siblings(documents, size=None):
    """Day 2 #2: many sections are only a line or two. Join neighbouring SMALL sections of the
    same file and the same parent heading, up to `size` characters. Each part keeps its own
    sub-heading inside the text, so nothing is lost."""
    size = size or CHUNK_SIZE
    merged = []
    for d in documents:
        prev = merged[-1] if merged else None
        if (prev is not None
                and prev.metadata["source"] == d.metadata["source"]
                and parent_heading(prev.metadata["heading"]) == parent_heading(d.metadata["heading"])
                and (len(prev.page_content) < SMALL_SECTION_CHARS or len(d.page_content) < SMALL_SECTION_CHARS)
                and len(prev.page_content) + len(d.page_content) <= size):
            if not prev.metadata.get("_merged"):             # first merge: name the first part too
                own = prev.metadata["heading"].rsplit(" > ", 1)[-1]
                if own != parent_heading(prev.metadata["heading"]):
                    prev.page_content = own + "\n" + prev.page_content
                prev.metadata["heading"] = parent_heading(prev.metadata["heading"])
                prev.metadata["_merged"] = True
            own = d.metadata["heading"].rsplit(" > ", 1)[-1]
            prefix = own + "\n" if own != prev.metadata["heading"] else ""
            prev.page_content += "\n\n" + prefix + d.page_content
            prev.metadata["page_end"] = max(prev.metadata["page_end"], d.metadata["page_end"])
            continue
        merged.append(Document(page_content=d.page_content, metadata=dict(d.metadata)))
    for d in merged:
        d.metadata.pop("_merged", None)
    return merged


def merge_tiny_chunks(chunks, size):
    """A chunk with less than MIN_CHUNK_CHARS of text (often the tail of a split) joins the chunk
    before it from the same section, if that stays within 1.25 x size."""
    out = []
    for c in chunks:
        prev = out[-1] if out else None
        if (len(c.page_content.strip()) < MIN_CHUNK_CHARS and prev is not None
                and prev.metadata["source"] == c.metadata["source"]
                and parent_heading(prev.metadata["heading"]) == parent_heading(c.metadata["heading"])
                and len(prev.page_content) + len(c.page_content) <= size * 1.25):
            if c.page_content.strip() not in prev.page_content:    # overlap may already hold it
                own = c.metadata["heading"].rsplit(" > ", 1)[-1]
                prefix = "\n\n" + own + "\n" if c.metadata["heading"] != prev.metadata["heading"] else " "
                prev.page_content += prefix + c.page_content.strip()
            prev.metadata["page_end"] = max(prev.metadata.get("page_end", 0), c.metadata.get("page_end", 0))
            continue
        out.append(c)
    return out


def create_chunks(documents, strategy=None):
    """Circulars: normal chunks. Annual reports: bigger chunks, so they do not flood the database."""
    strategy = strategy or CHUNK_STRATEGY
    circulars = [d for d in documents if d.metadata["doc_type"] != "annual_report"]
    annual = [d for d in documents if d.metadata["doc_type"] == "annual_report"]
    circulars = merge_small_siblings(circulars, CHUNK_SIZE)
    chunks = merge_tiny_chunks(split_group(circulars, strategy, CHUNK_SIZE), CHUNK_SIZE) if circulars else []
    if annual:
        chunks += merge_tiny_chunks(split_group(merge_sections(annual, ANNUAL_CHUNK_SIZE), strategy,
                                                ANNUAL_CHUNK_SIZE), ANNUAL_CHUNK_SIZE)

    for i, c in enumerate(chunks):              # EVERY chunk says which section it is from
        c.metadata["heading"] = c.metadata["heading"][:MAX_HEADING_PATH]
        c.page_content = c.metadata["heading"] + "\n" + c.page_content
        # Day 3 #1: a unique ID per chunk (hybrid search merges on it, Chroma stores it)
        c.metadata["chunk_id"] = hashlib.sha1(
            f"{c.metadata['source']}|{c.metadata['page']}|{i}|{c.page_content}".encode()).hexdigest()[:16]

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
                SEMANTIC_MAX_CHARS, DB_VERSION, EMBED_MODEL, ANNUAL_CHUNK_SIZE,
                ANNUAL_REPORT.pattern, INCLUDE_ANNUAL_REPORTS,
                MIN_SECTION_CHARS, MAX_HEADING_PATH, SMALL_SECTION_CHARS, MIN_CHUNK_CHARS,
                SEMANTIC_PERCENTILE, SEMANTIC_OUTLIER, SEMANTIC_MIN_GAP]
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
                vector_db.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
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
# Day 3 #2: ONLY circular numbers, "circular/direction + year" and words that are about
# versions. Ordinary words ("credit history", "the previous policy", "an earlier statement")
# no longer let withdrawn circulars in.
OLD_VERSION_HINT = re.compile(
    r"\bRBI/(?:[A-Za-z]{2,10}/)*\d{4}-\d{2}/\d+"                         # a circular number
    r"|\b[A-Z]{2,6}(?:\.[A-Za-z]{1,10}){1,6}\.?\s?(?:No\.?\s?)?\d+/[\d.]+/\d{4}-\d{2,4}"   # DBR.BP.BC.No.45/...
    r"|\b(?:circular|direction|notification|guidelines?|master\s+(?:circular|direction))\b[^.?]{0,40}"
    r"\b(?:19|20)\d{2}\b"
    r"|\b(?:19|20)\d{2}\b[^.?]{0,20}\b(?:circular|direction|notification|guidelines?)\b"
    r"|\b(?:superseded|supersedes|withdrawn|repealed|rescinded|no\s+longer\s+in\s+force"
    r"|old(?:er)?\s+(?:circular|direction|version|rule|instruction|guideline)s?"
    r"|(?:previous|earlier)\s+(?:circular|direction|version|master\s+direction)s?)\b", re.I)


SCOPES = {"all": None,                                     # sidebar "Search in"
          "circulars": {"doc_type": {"$ne": "annual_report"}},
          "annual_reports": {"doc_type": "annual_report"}}


def search_filter(question, scope="all"):
    """Chroma filter: latest versions only (unless the question asks for old ones) + document type."""
    parts = []
    if not OLD_VERSION_HINT.search(question):
        parts.append({"is_latest": True})
    if SCOPES.get(scope):
        parts.append(SCOPES[scope])
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {"$and": parts}


def in_scope(meta, scope):
    kind = meta.get("doc_type", "")
    return (scope == "all" or (scope == "annual_reports") == (kind == "annual_report"))


def limit_per_source(docs, limit=MAX_PER_SOURCE):
    """Keep at most `limit` chunks per file, so one big report cannot fill every slot."""
    seen, kept = {}, []
    for d in docs:
        src = d.metadata.get("source")
        if seen.get(src, 0) < limit:
            kept.append(d)
            seen[src] = seen.get(src, 0) + 1
    return kept


# 4b. Follow-up questions -> one standalone question
FOLLOW_UP = re.compile(
    r"^\s*(?:and|or|but|also|so|then|what about|how about|same|why|why not|is that|are they|does it|"
    r"can it|what if|in that case)\b"
    r"|\b(?:it|its|it's|they|them|their|this|that|these|those|the same|above|such|there|he|she|"
    r"former|latter|mentioned)\b", re.I)
TOPIC_WORD = re.compile(r"\b[A-Za-z]{4,}\b")


def looks_like_follow_up(question):
    """Day 3 #4: only rewrite questions that need the conversation:
    'and for rural accounts?', 'what is its limit?', 'why?' -> yes
    'What is the KYC limit for small accounts?'               -> no (complete on its own)"""
    if FOLLOW_UP.search(question):
        return True
    return len(TOPIC_WORD.findall(question)) <= 1           # "limit?", "and rural?"


def make_standalone(question, history):
    """'and for rural accounts?' + earlier turns -> 'What is the KYC limit for rural accounts?'"""
    if not history or not looks_like_follow_up(question):
        return question
    convo = "\n".join(f"User: {q}\nAssistant: {a[:400]}" for q, a in history[-HISTORY_TURNS:])
    messages = [
        ("system", "Rewrite the user's LAST question as one standalone question, using the "
                   "conversation for missing details. Keep numbers, circular numbers and names "
                   "exactly. If it is already standalone, return it unchanged. "
                   "Return ONLY the question."),
        ("human", f"Conversation:\n{convo}\n\nLast question: {question}"),
    ]
    standalone = ask(messages)
    return standalone or question


# 4c. Query rewriting (search only; the answer uses the real question)
def rewrite_query(question):
    messages = [
        ("system", f"Rewrite the user's question as one clear, complete search query "
                   f"for {DOMAIN}. Keep circular numbers, section numbers and "
                   f"amounts exactly as written. Return ONLY the rewritten query."),
        ("human", question),
    ]
    rewritten = ask(messages)
    return rewritten or question


# 4d. BM25 keyword search (own small index on rank_bm25; no langchain-community)
REF_TOKEN = re.compile(r"\b[A-Za-z]{2,10}(?:[./][A-Za-z0-9\-]+)*?[./]\d[\w.\-]*(?:/[\w.\-]+)+")


def tokenize(text):
    """Words, plus every circular number as ONE token (Day 3 #6):
    'RBI/2024-25/12' -> rbi, 2024, 25, 12  AND  rbi/2024-25/12"""
    words = re.findall(r"\w+", text.lower())                # "Passport?" == "passport"
    refs = [r.lower().rstrip(".-") for r in REF_TOKEN.findall(text)]
    return words + refs


class KeywordIndex:
    def __init__(self, docs):
        from rank_bm25 import BM25Okapi
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in docs])

    def search(self, query, k, only_latest=True, scope="all"):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        hits = []
        for i in order:
            if scores[i] <= 0 or len(hits) >= k:
                break
            if only_latest and not self.docs[i].metadata.get("is_latest", True):
                continue
            if not in_scope(self.docs[i].metadata, scope):
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
            # Day 3 #1: the chunk's own ID (old databases: file + page + FULL text)
            key = doc.metadata.get("chunk_id") or (doc.metadata.get("source"), doc.metadata.get("page"),
                                                   doc.page_content)
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

def retrieve(vector_db, question, hybrid=None, rerank_on=None, rewrite=None, scope=None, top_k=None):
    scope = scope or SEARCH_SCOPE
    hybrid = USE_HYBRID if hybrid is None else hybrid
    rerank_on = USE_RERANK if rerank_on is None else rerank_on
    rewrite = USE_REWRITE if rewrite is None else rewrite
    top_k = top_k or top_k_for(rerank_on)                  # eval.py asks for the same k everywhere

    query = rewrite_query(question) if rewrite else question
    if rewrite:
        print(f"  (searching for: {query})")

    where = search_filter(question, scope)
    k = FETCH_K                                            # fetch more, then keep max 2 per file
    if MIN_RELEVANCE:                                      # cut-off on: keep only close matches
        scored = vector_db.similarity_search_with_relevance_scores(query, k=k, filter=where)
        if not scored or scored[0][1] < MIN_RELEVANCE:
            return []                                      # nothing close enough: "not found"
        results = [doc for doc, _ in scored]
    else:
        results = vector_db.similarity_search(query, k=k, filter=where)

    if hybrid:
        keyword = get_keyword_index(vector_db).search(query, FETCH_K, scope=scope,
                                                      only_latest=not OLD_VERSION_HINT.search(question))
        results = rrf_merge([results, keyword])[:FETCH_K]
    if rerank_on and results:
        results = rerank(query, results)
    return limit_per_source(results)[:top_k]


def top_relevance(vector_db, question, scope=None):
    scope = scope or SEARCH_SCOPE
    """Best relevance score for a question (eval.py uses it to suggest MIN_RELEVANCE)."""
    scored = vector_db.similarity_search_with_relevance_scores(question, k=1,
                                                               filter=search_filter(question, scope))
    return scored[0][1] if scored else 0.0


# ---------------------------------------------------------------------------
# 6. Generation with citations
# ---------------------------------------------------------------------------

def passage_header(i, doc):
    """Day 3 #3: give the model what rule 6 needs: number, date, old or current.
    [2] (RBI (KYC) Direction.pdf, page 4 | RBI/2016-17/11 | issued 2016-02-25 | OLD VERSION, replaced by ...)"""
    m = doc.metadata
    parts = [f"{m.get('source')}, page {m.get('page')}"]
    if m.get("circular_no"):
        parts.append(m["circular_no"])
    if m.get("issue_date"):
        parts.append(f"issued {m['issue_date']}")
    if m.get("is_latest") is False:
        parts.append("OLD VERSION (withdrawn or superseded"
                     + (f", replaced by {m['superseded_by']})" if m.get("superseded_by") else ")"))
    return f"[{i}] (" + " | ".join(parts) + ")"


def generate_answer(question, results, on_token=None):
    context = "\n\n".join(f"{passage_header(i, doc)}\n{doc.page_content}"
                            for i, doc in enumerate(results, start=1))
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    if on_token is None:
        return ask(messages)
    answer = ""
    for part in llm.stream(messages):           # words arrive one by one: show them at once
        answer += part.content or ""
        shown = clean(answer)                   # thinking part is hidden
        if shown:
            on_token(shown)
    return clean(answer)


def warm_up():
    """Load the chat model into memory once, so the first real question is not slow."""
    try:
        llm.invoke("hi")
    except Exception:
        pass


def check_citations(answer, n_sources):
    """-> (cited numbers that exist, numbers that do not exist)"""
    used = set()                                # [1], [1][3], [1, 2], [1-3], [1–3], [1,2,4]
    for inside in re.findall(r"\[(\s*\d+(?:\s*[,;\-–]\s*\d+)*\s*)\]", answer):
        for a, b in re.findall(r"(\d+)\s*(?:[\-–]\s*(\d+))?", inside):
            lo, hi = int(a), int(b or a)
            used.update(range(lo, min(hi, lo + 20) + 1))
    valid = {n for n in used if 1 <= n <= n_sources}
    return valid, sorted(used - valid)


GREETING = re.compile(r"\s*(hi|hello|hey|hai|good (morning|afternoon|evening)|thanks?|thank you)[\s!.]*",
                      re.I)


def answer_question(vector_db, question, history=None, hybrid=None, rerank_on=None, rewrite=None,
                    on_token=None, scope=None):
    """Full pipeline for one chat turn (used by Streamlit and eval)."""
    if GREETING.fullmatch(question):                           # "hi", "thanks": no search needed
        return {"question_used": question, "results": [], "cited": set(), "invalid_citations": [],
                "answer": f"Hello! Ask me a question about {DOMAIN}, for example: "
                          "'What is the Rs 2000 note exchange limit?'"}
    standalone = make_standalone(question, history)
    results = retrieve(vector_db, standalone, hybrid, rerank_on, rewrite, scope)
    if not results:
        answer = "I could not find this in the documents."
    else:
        answer = generate_answer(standalone, results, on_token)
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