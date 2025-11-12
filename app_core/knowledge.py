"""Knowledge base helpers and query heuristics."""
from __future__ import annotations

import json
import logging
import re
import string
from typing import Dict, List, Optional, Sequence, Set

import streamlit as st

log = logging.getLogger(__name__)

COURSE_HINTS = [
    "course",
    "class",
    "prereq",
    "prerequisite",
    "credit",
    "credits",
    "catalog",
    "syllabus",
    "covers",
    "topic",
    "learn",
    "teaches",
    "semester",
    "enroll",
    "registration",
    "requirement",
    "requirements",
    "what is",
    "describe",
    "explain",
    "about",
]
COURSE_CODE_RE = re.compile(r"\b([A-Za-z]{2,4})\s*-?\s*(\d{3})\b")
NAME_RE = re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z0-9_\- ]{1,40})\b", re.IGNORECASE)
_CODE_RE = re.compile(r"^[A-Za-z]{2,5}\s?\d{3}$")

KNOWN_COLLEGES: Set[str] = {"CAS", "COP", "COD"}
LANG_OPTIONS = {"English": "English", "Arabic": "Arabic"}

DEGREE_TOTAL = {"CS": 126, "Pharmacy": 180, "Dentistry": 189}
MAJOR_MAP = {
    "CS": {"college": "CAS", "prefixes": ("CSC", "MAT", "STA")},
    "Pharmacy": {"college": "COP", "prefixes": ("PHA", "CHE", "BIO")},
    "Dentistry": {"college": "COD", "prefixes": ("BDS", "BIO", "CHE")},
}

CHUNK_CHAR_CAP = 900
HISTORY_TURNS = 6


def load_general_kb(path: str = "general_academic_kb.json") -> str:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            items = json.load(handle)
        lines = []
        for item in items:
            topic = (item.get("topic") or "").strip()
            content = (item.get("content") or "").strip()
            if topic and content:
                lines.append(f"- {topic}: {content}")
        return "\n".join(lines[:20])
    except Exception:
        return ""


def needs_prep_tips(question: str) -> bool:
    text = (question or "").lower()
    keywords = [
        "prepare",
        "preparation",
        "study tips",
        "how to study",
        "how do i prepare",
        "how can i prepare",
        "succeed",
        "revision",
        "exam",
        "practice",
        "resources",
        "what to expect",
        "how to get ready",
        "how to revise",
    ]
    return any(key in text for key in keywords)


def web_enrichment_snippet(question: str, max_chars: int = 800) -> str:
    try:
        from duckduckgo_search import DDGS  # type: ignore

        with DDGS() as ddgs:
            results = list(ddgs.text(question, region="wt-wt", safesearch="moderate", max_results=3))
        snippets: List[str] = []
        for result in results:
            body = (result.get("body") or "").strip()
            if body:
                snippets.append(body)
        combined = "\n\n".join(snippets)
        return combined[:max_chars]
    except Exception as exc:  # pragma: no cover - optional dependency
        log.debug("duckduckgo enrichment failed: %s", exc)
        return ""


def _trim(value: str, limit: int) -> str:
    value = (value or "").strip()
    return f"{value[:limit]}…" if len(value) > limit else value


def prepare_history_text(messages, n_pairs: int = HISTORY_TURNS) -> str:
    out, user_seen = [], 0
    for msg in reversed(messages or []):
        out.append(msg)
        if msg.get("role") == "user":
            user_seen += 1
            if user_seen >= n_pairs:
                break
    out.reverse()
    return "\n".join(f"{m['role']}: {m.get('content','')}" for m in out if m.get("content"))


def build_history_text(max_turns: int = HISTORY_TURNS) -> str:
    history = prepare_history_text(st.session_state.get("messages", []), n_pairs=max_turns)
    return history


def fastpath_course_code(query: str, rows) -> Optional[str]:
    if not query or not _CODE_RE.match(query.strip()):
        return None
    key = re.sub(r"\s+", "", query).upper()
    for row in rows or []:
        code = re.sub(r"\s+", "", str(row.get("code", ""))).upper()
        if code == key:
            title = (row.get("title") or "").strip()
            description = (row.get("description") or "").strip()
            return f"**{row.get('code','')} — {title}**\n\n{description}"
    return None


def _norm_text(value: str) -> str:
    return (value or "").lower().translate(str.maketrans("", "", string.punctuation)).strip()


def expand_synonyms(text: str) -> str:
    if not text:
        return text
    replacements = {
        r"\bds\b": "data structures",
        r"\bdata struct(ure)?s?\b": "data structures",
        r"\balgo(rithms)?\b": "algorithms",
        r"\bai\b": "artificial intelligence",
        r"\bml\b": "machine learning",
        r"\bprog\b": "programming",
        r"\bpharm\b": "pharmacy",
        r"\bdent(al)?\b": "dentistry",
    }
    out = " " + text.lower() + " "
    for pattern, replacement in replacements.items():
        out = re.sub(pattern, replacement, out)
    return out.strip()


def simple_token_overlap(a: str, b: str) -> int:
    tokens_a, tokens_b = set(_norm_text(a).split()), set(_norm_text(b).split())
    return len(tokens_a & tokens_b)


def rows_to_kb(rows_subset: List[Dict]) -> str:
    return "\n---\n".join(
        f"College: {(row.get('college') or 'Unknown')}\n"
        f"Course Code: {row['code']}\n"
        f"Title: {row['title']}\n"
        f"Description: {row['description']}\n"
        f"Prerequisites: {(row.get('prereqs') or 'Unknown')}\n"
        f"Credits: {(row.get('credits') or 'Unknown')}\n"
        f"[source: courses.csv | code: {row['code']}]"
        for row in rows_subset
    )


def is_coursey(question: str) -> bool:
    if not question:
        return False
    if COURSE_CODE_RE.search(question):
        return True
    text = question.lower()
    return any(hint in text for hint in COURSE_HINTS + ["csc", "mat", "mth", "sta", "pha", "che", "bio", "bds", "pharm", "dent"])


def maybe_capture_name(question: str) -> None:
    if not question:
        return
    match = NAME_RE.search(question)
    if match:
        st.session_state.user_name = match.group(1).strip()


def friendly_prefix() -> str:
    name = st.session_state.get("user_name")
    return f"{name}, " if name else ""


def find_rows_by_code(rows: List[Dict], question: str) -> List[Dict]:
    if not rows or not question:
        return []
    index = {(row.get("code", "") or "").replace(" ", "").upper(): row for row in rows if "code" in row}
    hits, seen = [], set()
    for dept, num in COURSE_CODE_RE.findall(question):
        key = f"{dept.upper()}{num}"
        row = index.get(key)
        if row and row["code"] not in seen:
            hits.append(row)
            seen.add(row["code"])
    return hits


def find_rows_by_title(rows: List[Dict], question: str) -> List[Dict]:
    if not rows or not question:
        return []
    question_tokens = set(_norm_text(question).split())
    best, best_score = None, 0
    for row in rows:
        title_norm = _norm_text(row.get("title", ""))
        if not title_norm:
            continue
        title_tokens = set(title_norm.split())
        overlap = len(question_tokens & title_tokens)
        if overlap > best_score and overlap >= 2:
            best, best_score = row, overlap
    return [best] if best else []


def infer_scopes(question: str) -> Dict[str, List[str]]:
    text = (question or "").lower()
    dept_scopes: List[str] = []
    college_scopes: List[str] = []
    if any(word in text for word in ["computer science", " comp sci", " cs ", " c.s.", "csc", "programming"]):
        dept_scopes.append("cs")
    if any(word in text for word in ["math", "mathematics", " mat ", "mth", "algebra", "calculus"]):
        dept_scopes.append("math")
    if any(word in text for word in ["statistics", " sta ", "probability"]):
        dept_scopes.append("stats")
    if any(word in text for word in ["chemistry", "organic", "inorganic", "che "]):
        dept_scopes.append("chem")
    if any(word in text for word in ["biology", "bio", "genetics"]):
        dept_scopes.append("bio")
    if any(word in text for word in ["pharmacy", "pharm", "pha"]):
        dept_scopes.append("pharm")
    if any(word in text for word in ["dentistry", "dent", "bds"]):
        dept_scopes.append("dent")
    for tag in {"CAS", "COP", "COD"}:
        if tag.lower() in text:
            college_scopes.append(tag)
    return {"dept": dept_scopes or ["all"], "college": college_scopes or ["all"]}


def filter_rows_by_college(rows: List[Dict], college_tag: str) -> List[Dict]:
    if college_tag == "All":
        return rows
    return [row for row in rows if (row.get("college", "").upper() == college_tag)]


def reorder_docs_by_scopes(docs, scopes: Dict[str, List[str]], college_filter: str):
    if not docs:
        return docs
    colleges = {c.upper() for c in scopes.get("college", []) if c != "all"}
    if college_filter != "All":
        colleges.add(college_filter.upper())
    dept_prefixes: Sequence[str] = tuple(
        prefix
        for scope in scopes.get("dept", [])
        if scope != "all"
        for prefix in {
            "cs": ("CSC", "CSE"),
            "math": ("MAT", "MTH"),
            "stats": ("STA",),
            "chem": ("CHE", "CHEM"),
            "bio": ("BIO", "BIOL"),
            "pharm": ("PHA",),
            "dent": ("BDS",),
        }.get(scope, ())
    )

    def score(doc):
        result = 0
        meta = getattr(doc, "metadata", {}) or {}
        code = meta.get("code", "").upper()
        college = meta.get("college", "").upper()
        if colleges and college in colleges:
            result -= 2
        if dept_prefixes and code.startswith(dept_prefixes):
            result -= 1
        return result

    return sorted(docs, key=score)


def parse_catalog_intent(question: str) -> Optional[Dict]:
    if not question:
        return None
    text = question.lower()
    scopes = infer_scopes(question)
    if "how many" in text and "course" in text:
        return {"type": "count", "scopes": scopes}
    if any(
        keyword in text
        for keyword in ["list all courses", "show all courses", "list courses", "all courses"]
    ):
        return {"type": "list", "limit": 150, "scopes": scopes}
    return None


def hybrid_retrieve(query: str, retriever, vector_store, top_k: int, bm25=None):
    if not query:
        return []
    expanded = expand_synonyms(query)
    vector_docs = []
    try:
        vector_docs = retriever.invoke(expanded) or []
    except AttributeError:
        vector_docs = retriever.get_relevant_documents(expanded) or []
    except Exception as exc:
        log.error("Vector retrieve error: %s", exc)
    bm_docs = []
    if bm25 is not None:
        try:
            all_docs = list(vector_store.docstore._dict.values())
            scores = bm25.get_scores(expanded.split())
            best_ids = sorted(range(len(all_docs)), key=lambda i: -scores[i])[:top_k]
            bm_docs = [all_docs[i] for i in best_ids]
        except Exception as exc:
            log.warning("BM25 failed: %s", exc)
    keyed = {}
    for doc in vector_docs + bm_docs:
        meta = getattr(doc, "metadata", {}) or {}
        keyed[(meta.get("code"), getattr(doc, "page_content", ""))] = doc
    merged = list(keyed.values())
    merged.sort(key=lambda doc: -simple_token_overlap(expanded, getattr(doc, "page_content", "")))
    return merged[: max(top_k * 2, top_k)]


def build_kb_from_docs(semantic_docs, bm25_docs, *, top_k: int, cap: int = CHUNK_CHAR_CAP) -> str:
    docs = []
    if semantic_docs:
        docs.extend(semantic_docs)
    if bm25_docs:
        docs.extend(bm25_docs)
    docs = docs[:top_k]
    return "\n\n".join(
        f"[{idx + 1}] {_trim(getattr(doc, 'page_content', str(doc)), cap)}"
        for idx, doc in enumerate(docs)
    )


def prepare_kb_from_docs(docs) -> str:
    if not docs:
        return ""
    blocks = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        text = getattr(doc, "page_content", "")
        blocks.append(text + f"\n[source: {meta.get('source', '?')} | code: {meta.get('code', '?')}]")
    return "\n---\n".join(blocks).strip()


def get_current_major_key() -> str:
    options = sorted(MAJOR_MAP.keys())
    if "schedule_major_key" not in st.session_state or st.session_state.schedule_major_key not in options:
        st.session_state.schedule_major_key = options[0]
    return st.session_state.schedule_major_key


__all__ = [
    "LANG_OPTIONS",
    "KNOWN_COLLEGES",
    "MAJOR_MAP",
    "DEGREE_TOTAL",
    "CHUNK_CHAR_CAP",
    "HISTORY_TURNS",
    "load_general_kb",
    "needs_prep_tips",
    "web_enrichment_snippet",
    "prepare_history_text",
    "build_history_text",
    "fastpath_course_code",
    "expand_synonyms",
    "simple_token_overlap",
    "rows_to_kb",
    "is_coursey",
    "maybe_capture_name",
    "friendly_prefix",
    "find_rows_by_code",
    "find_rows_by_title",
    "infer_scopes",
    "filter_rows_by_college",
    "reorder_docs_by_scopes",
    "parse_catalog_intent",
    "hybrid_retrieve",
    "build_kb_from_docs",
    "prepare_kb_from_docs",
    "get_current_major_key",
]
