"""RAG-style matcher between JD signals and resume library entries.

Purpose:
- Turn structured JD signals into search queries.
- Retrieve relevant resume items from the local resume library.
- Output ranked matches for experience and project sections.

Supported retrieval strategies:
1) `keyword`
   Deterministic lexical matching with phrase and token fallback.
2) `embedding`
   Semantic matching using vector embeddings + cosine similarity.

The output is written to `rag_match_output.json` and consumed by the final
JSON builder (`build_final_json.py`).
"""

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Default input/output paths used by CLI flags.
DEFAULT_JD_ANALYSIS_PATH = "jd_analysis.json"
DEFAULT_LIBRARY_PATH = "resume_library.json"
DEFAULT_DB_PATH = "resume_embeddings.db"
DEFAULT_OUTPUT_PATH = "rag_match_output.json"

# Default embedding model when semantic retrieval is enabled.
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def load_json(path: Path) -> dict:
    """Load UTF-8 JSON as dict.

    Raises:
    - FileNotFoundError: if file does not exist.
    - json.JSONDecodeError: if file content is not valid JSON.
    """

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def to_text(value: object) -> str:
    """Normalize arbitrary values into strings.

    This keeps chunk-building robust when some fields are null/non-string.
    """

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def library_to_chunks(library: dict) -> List[dict]:
    """Flatten resume library sections into searchable text chunks.

    Why this exists:
    - Upstream library data is structured JSON.
    - Retrieval works on searchable text plus stable IDs.

    ID format:
    - experience -> "experience:<index>"
    - projects   -> "project:<index>"
    - skills     -> "skill:<index>"
    - education  -> "education:<index>"

    These IDs allow downstream steps to map a match back to the original item.
    """

    chunks: List[dict] = []

    # Experience section -> one chunk per experience row.
    for i, item in enumerate(library.get("experience", [])):
        title = to_text(item.get("title"))
        company = to_text(item.get("company"))
        location = to_text(item.get("location"))
        start_date = to_text(item.get("start_date"))
        end_date = to_text(item.get("end_date"))
        details = " ".join(item.get("details", []))

        # Keep text human-readable so both keyword and embedding retrieval benefit.
        text = (
            f"Experience: {title}. Company: {company}. Location: {location}. "
            f"Date: {start_date} - {end_date}. Details: {details}"
        ).strip()

        chunks.append(
            {
                "doc_id": f"experience:{i}",
                "source_type": "experience",
                "title": title,
                "text": text,
            }
        )

    # Projects section -> one chunk per project row.
    for i, item in enumerate(library.get("projects", [])):
        title = to_text(item.get("title"))
        start_date = to_text(item.get("start_date"))
        end_date = to_text(item.get("end_date"))
        details = " ".join(item.get("details", []))
        text = f"Project: {title}. Date: {start_date} - {end_date}. Details: {details}".strip()
        chunks.append(
            {
                "doc_id": f"project:{i}",
                "source_type": "project",
                "title": title,
                "text": text,
            }
        )

    # Skills section is included for completeness, even though final selection
    # currently focuses on experience/project.
    for i, item in enumerate(library.get("skills", [])):
        title = to_text(item.get("title"))
        skills = ", ".join(item.get("skills", []))
        text = f"SkillCategory: {title}. Skills: {skills}".strip()
        chunks.append(
            {
                "doc_id": f"skill:{i}",
                "source_type": "skill",
                "title": title,
                "text": text,
            }
        )

    # Education section is also chunked, useful for future ranking extensions.
    for i, item in enumerate(library.get("education", [])):
        school = to_text(item.get("school"))
        major = to_text(item.get("major"))
        start_date = to_text(item.get("start_date"))
        end_date = to_text(item.get("end_date"))
        text = f"Education: {school}. Major: {major}. Date: {start_date} - {end_date}".strip()
        chunks.append(
            {
                "doc_id": f"education:{i}",
                "source_type": "education",
                "title": school or f"education_{i}",
                "text": text,
            }
        )

    return chunks


def library_hash(chunks: List[dict]) -> str:
    """Create deterministic hash of the searchable library view.

    Used by embedding mode to decide whether cached vectors are still valid.
    """

    joined = "\n".join([f"{c['doc_id']}|{c['text']}" for c in chunks])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def ensure_db(conn: sqlite3.Connection) -> None:
    """Create SQLite schema for embedding cache + metadata."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            doc_id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            vector_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str:
    """Read one metadata value, returning empty string when missing."""

    row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return row[0] if row else ""


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Upsert one metadata key/value pair."""

    conn.execute(
        "INSERT INTO metadata(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def rebuild_embeddings_if_needed(
    conn: sqlite3.Connection,
    chunks: List[dict],
    embeddings: OpenAIEmbeddings,
    embed_model: str,
) -> None:
    """Rebuild local vector cache only when required.

    Rebuild conditions:
    - Library hash changed.
    - Embedding model changed.
    - Embedding table is empty.

    This avoids re-embedding unchanged data on repeated runs.
    """

    current_hash = library_hash(chunks)
    existing_hash = get_meta(conn, "library_hash")
    existing_model = get_meta(conn, "embedding_model")
    existing_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    # Early exit when cache is already valid.
    if existing_hash == current_hash and existing_model == embed_model and existing_count > 0:
        return

    # Full refresh strategy keeps implementation simple and deterministic.
    conn.execute("DELETE FROM embeddings")
    if chunks:
        vectors = embeddings.embed_documents([c["text"] for c in chunks])
        for chunk, vector in zip(chunks, vectors):
            conn.execute(
                "INSERT INTO embeddings(doc_id, source_type, title, text, vector_json) VALUES(?, ?, ?, ?, ?)",
                (
                    chunk["doc_id"],
                    chunk["source_type"],
                    chunk["title"],
                    chunk["text"],
                    json.dumps(vector),
                ),
            )

    set_meta(conn, "library_hash", current_hash)
    set_meta(conn, "embedding_model", embed_model)
    conn.commit()


def load_all_vectors(conn: sqlite3.Connection) -> List[dict]:
    """Load all embedding rows into memory for scoring."""

    rows = conn.execute(
        "SELECT doc_id, source_type, title, text, vector_json FROM embeddings"
    ).fetchall()

    out: List[dict] = []
    for row in rows:
        out.append(
            {
                "doc_id": row[0],
                "source_type": row[1],
                "title": row[2],
                "text": row[3],
                "vector": json.loads(row[4]),
            }
        )
    return out


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between equal-length vectors.

    Returns -1.0 for invalid/empty/zero-norm vectors to signal "not usable".
    """

    if not v1 or not v2 or len(v1) != len(v2):
        return -1.0

    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return -1.0
    return dot / (n1 * n2)


def jd_queries(jd: dict) -> List[Tuple[str, str]]:
    """Build deduplicated query list from selected JD fields.

    Current fields intentionally used:
    - must_have_skills
    - keywords
    - soft_skills

    Output tuple format: (query_type, query_text)
    """

    out: List[Tuple[str, str]] = []
    for field in ["must_have_skills", "keywords", "soft_skills"]:
        for text in jd.get(field, []):
            t = to_text(text).strip()
            if t:
                out.append((field, t))

    # Deduplicate by lowercase text to avoid duplicate scoring pressure.
    seen = set()
    deduped = []
    for item in out:
        key = item[1].lower()
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def normalize_text(text: str) -> str:
    """Normalize text for robust keyword matching.

    We preserve technical symbols (+, #, /, .) to retain terms like:
    - C++
    - C#
    - .NET
    - path-ish strings
    """

    text = text.lower()
    text = re.sub(r"[^a-z0-9\+\#\/\.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_token_set(text: str) -> set:
    """Tokenize normalized text into set form for fast overlap checks."""

    normalized = normalize_text(text)
    if not normalized:
        return set()
    return set(normalized.split(" "))


def phrase_match(phrase: str, doc_text_norm: str, doc_tokens: set) -> bool:
    """Match a query phrase against one document using two-stage logic.

    Stage 1 (strict): exact normalized substring exists in doc text.
    Stage 2 (fallback): all normalized phrase tokens exist in doc token set.

    This balances precision with tolerance for minor wording differences.
    """

    phrase_norm = normalize_text(phrase)
    if not phrase_norm:
        return False

    # Fast exact phrase match.
    if phrase_norm in doc_text_norm:
        return True

    # Token fallback: every query token must appear in document tokens.
    phrase_tokens = [t for t in phrase_norm.split(" ") if t]
    if not phrase_tokens:
        return False
    return all(t in doc_tokens for t in phrase_tokens)


def jaccard_similarity(tokens_a: set, tokens_b: set) -> float:
    """Compute Jaccard similarity for near-duplicate filtering."""

    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    if union == 0:
        return 0.0
    return inter / union


def retrieve_source_matches_keyword(
    queries: List[Tuple[str, str]],
    docs: List[dict],
    source_type: str,
    top_k: int,
    dedup_similarity: float,
) -> List[dict]:
    """Keyword retrieval path for one source type (experience/project/etc.).

    Ranking:
    1) keyword_score (matched query count)
    2) coverage (matched_count / total_query_count)

    De-duplication:
    - Remove candidates with token Jaccard similarity above threshold.
    """

    source_docs = [d for d in docs if d["source_type"] == source_type]
    if not source_docs or not queries:
        return []

    # Pre-deduplicate query texts before scoring docs.
    query_items = []
    seen = set()
    for query_type, query_text in queries:
        key = normalize_text(query_text)
        if key and key not in seen:
            seen.add(key)
            query_items.append((query_type, query_text, key))

    scored = []
    for doc in source_docs:
        doc_text = doc["text"]
        doc_text_norm = normalize_text(doc_text)
        doc_tokens = text_token_set(doc_text)

        matched = []
        for query_type, query_text, _ in query_items:
            if phrase_match(query_text, doc_text_norm, doc_tokens):
                matched.append({"query_type": query_type, "query": query_text})

        score = len(matched)
        if score == 0:
            continue

        coverage = score / len(query_items) if query_items else 0.0
        scored.append(
            {
                "doc_id": doc["doc_id"],
                "source_type": doc["source_type"],
                "title": doc["title"],
                "keyword_score": score,
                "coverage": round(coverage, 4),
                "matched_keywords": matched,
                "text": doc["text"][:400],
                # Internal helper field used only for dedup stage.
                "_doc_tokens": doc_tokens,
            }
        )

    # Sort highest relevance first.
    scored.sort(key=lambda x: (x["keyword_score"], x["coverage"]), reverse=True)

    # Keep top unique results based on token-set similarity.
    kept = []
    for candidate in scored:
        is_duplicate = False
        for existing in kept:
            sim = jaccard_similarity(candidate["_doc_tokens"], existing["_doc_tokens"])
            if sim > dedup_similarity:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(candidate)
        if len(kept) >= top_k:
            break

    # Remove helper field before final payload.
    for item in kept:
        item.pop("_doc_tokens", None)
    return kept


def retrieve_source_matches_embedding(
    queries: List[Tuple[str, str]],
    docs: List[dict],
    embeddings: OpenAIEmbeddings,
    source_type: str,
    top_k: int,
    threshold: float,
    dedup_similarity: float,
) -> List[dict]:
    """Embedding retrieval path for one source type.

    Flow:
    1) Embed all query texts.
    2) Score each query against each source doc vector.
    3) Keep query-doc matches with score >= threshold.
    4) Rank by match_count then avg_score.
    5) Remove near-duplicate docs by vector similarity.
    """

    source_docs = [d for d in docs if d["source_type"] == source_type]
    if not source_docs or not queries:
        return []

    # One embedding call per query text; reused across all documents.
    query_vectors = []
    for query_type, query_text in queries:
        query_vectors.append((query_type, query_text, embeddings.embed_query(query_text)))

    scored_experiences = []
    for doc in source_docs:
        matched_queries = []
        for query_type, query_text, query_vec in query_vectors:
            score = cosine_similarity(query_vec, doc["vector"])
            if score >= threshold:
                matched_queries.append(
                    {
                        "query_type": query_type,
                        "query": query_text,
                        "score": round(score, 4),
                    }
                )

        match_count = len(matched_queries)
        if match_count == 0:
            continue

        matched_queries.sort(key=lambda x: x["score"], reverse=True)
        avg_score = sum(m["score"] for m in matched_queries) / match_count
        coverage = match_count / len(queries)

        scored_experiences.append(
            {
                "doc_id": doc["doc_id"],
                "source_type": doc["source_type"],
                "title": doc["title"],
                "match_count": match_count,
                "avg_score": round(avg_score, 4),
                "coverage": round(coverage, 4),
                "matched_queries": matched_queries,
                "text": doc["text"][:400],
            }
        )

    # Higher match_count wins; avg_score breaks ties.
    scored_experiences.sort(
        key=lambda x: (x["match_count"], x["avg_score"]),
        reverse=True,
    )

    # Deduplicate by cosine similarity between document vectors.
    kept = []
    for candidate in scored_experiences:
        candidate_vec = next(d["vector"] for d in source_docs if d["doc_id"] == candidate["doc_id"])
        is_duplicate = False
        for existing in kept:
            existing_vec = next(d["vector"] for d in source_docs if d["doc_id"] == existing["doc_id"])
            if cosine_similarity(candidate_vec, existing_vec) > dedup_similarity:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(candidate)
        if len(kept) >= top_k:
            break

    return kept


def main() -> None:
    """CLI entrypoint for JD-to-library retrieval + ranking."""

    # Load API keys and optional embedding-model override from `.env`.
    load_dotenv()

    parser = argparse.ArgumentParser(description="RAG match JD analysis to resume library.")
    parser.add_argument("--jd-analysis", default=DEFAULT_JD_ANALYSIS_PATH)
    parser.add_argument("--library", default=DEFAULT_LIBRARY_PATH)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.38)
    parser.add_argument(
        "--retrieval",
        choices=["keyword", "embedding"],
        default="keyword",
        help="Retrieval strategy. default=keyword",
    )
    parser.add_argument(
        "--dedup-similarity",
        type=float,
        default=0.90,
        help="Drop near-duplicate items if similarity is above this value.",
    )
    args = parser.parse_args()

    # Read inputs and prepare searchable corpus.
    jd = load_json(Path(args.jd_analysis))
    library = load_json(Path(args.library))
    chunks = library_to_chunks(library)

    # Build JD queries once; both retrieval modes use the same query list.
    queries = jd_queries(jd)
    embed_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)

    if args.retrieval == "keyword":
        # Keyword mode does not need vector DB; it works directly on chunk text.
        docs = chunks
        experience_matches = retrieve_source_matches_keyword(
            queries=queries,
            docs=docs,
            source_type="experience",
            top_k=args.top_k,
            dedup_similarity=args.dedup_similarity,
        )
        project_matches = retrieve_source_matches_keyword(
            queries=queries,
            docs=docs,
            source_type="project",
            top_k=args.top_k,
            dedup_similarity=args.dedup_similarity,
        )
    else:
        # Embedding mode reads/writes SQLite cache for faster reruns.
        embeddings = OpenAIEmbeddings(model=embed_model)
        db_path = Path(args.db)
        conn = sqlite3.connect(db_path)
        try:
            ensure_db(conn)
            rebuild_embeddings_if_needed(conn, chunks, embeddings, embed_model)
            docs = load_all_vectors(conn)
        finally:
            conn.close()

        experience_matches = retrieve_source_matches_embedding(
            queries=queries,
            docs=docs,
            embeddings=embeddings,
            source_type="experience",
            top_k=args.top_k,
            threshold=args.threshold,
            dedup_similarity=args.dedup_similarity,
        )
        project_matches = retrieve_source_matches_embedding(
            queries=queries,
            docs=docs,
            embeddings=embeddings,
            source_type="project",
            top_k=args.top_k,
            threshold=args.threshold,
            dedup_similarity=args.dedup_similarity,
        )

    # Final payload is intentionally verbose to aid debugging/tuning.
    payload = {
        "retrieval_mode": args.retrieval,
        "embedding_model": embed_model,
        "db_path": str(Path(args.db).resolve()),
        "library_items": len(chunks),
        "query_count": len(queries),
        "query_fields_used": ["must_have_skills", "keywords", "soft_skills"],
        "threshold": args.threshold,
        "dedup_similarity": args.dedup_similarity,
        "top_k": args.top_k,
        "experience_results": experience_matches,
        "project_results": project_matches,
    }

    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. RAG match output saved to: {Path(args.output).resolve()}")
    print(f"Embedding DB: {Path(args.db).resolve()}")


if __name__ == "__main__":
    main()
