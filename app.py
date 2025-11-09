# app.py
# AUIBPT — Course, Liberal-Arts & Schedule Builder (Ryunix build)
# - College-aware CSV (college,code,title,credits,description,prereqs)
# - Hybrid retrieval (FAISS + optional BM25 via lazy import)
# - Splash & footer (Ryunix Productions)
# - Schedule Builder: LA rules + major prereqs + per-course swap + auto top-up + undo + lock
# - Picker persists selections across filter changes (Major / LA / Both)
# - Export schedule (CSV) + Save/Load (JSON)
# - Student profile context — user “Completed Courses” feed the chat answers
# - Institutional KB (AUIB + CAS/COP/COD + faculty) with dedicated prompt & routing
# - Profile picture avatar + GUI color customizer
# - OpenAI-first LLM (Streamlit Cloud ready) with Groq/Ollama fallback for local

from __future__ import annotations

import os
import re
import csv
import io
import json
import time
import string
import logging
import importlib.util
from typing import List, Dict, Tuple, Optional, Set
from univkb import UNIV_KB, is_university_query, univ_kb_blocks_for
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from ui import apply_theme, render_appearance_controls
from data_io import build_or_load_index  
from settings import MODEL_NAME, CSV_PATH, INDEX_DIR, TOP_K, TEMPERATURE, NUM_PREDICT, USE_OPENAI
from collections import OrderedDict
import os
try:
    from settings import USE_GROQ_ONLY as _USE_GROQ_ONLY  
except Exception:
    _USE_GROQ_ONLY = None

_USE_GROQ_ONLY_ENV = os.getenv("USE_GROQ_ONLY", "").strip().lower()
if _USE_GROQ_ONLY_ENV in {"1", "true", "yes", "on"}:
    USE_GROQ_ONLY = True
elif _USE_GROQ_ONLY_ENV in {"0", "false", "no", "off"}:
    USE_GROQ_ONLY = False
elif isinstance(_USE_GROQ_ONLY, bool):
    USE_GROQ_ONLY = _USE_GROQ_ONLY
else:
    USE_GROQ_ONLY = False  

def get_current_major_key() -> str:
    """Return a valid major key from session or the first available one."""
    opts = sorted(MAJOR_MAP.keys()) if 'MAJOR_MAP' in globals() else []
    if not opts:
        return "GENERAL"
    if "schedule_major_key" not in st.session_state or st.session_state.schedule_major_key not in opts:
        st.session_state.schedule_major_key = opts[0]
    return st.session_state.schedule_major_key
def _qa_cache_get(q: str):
    key = (q or "").strip().lower()
    cache = st.session_state.setdefault("_qa_cache", OrderedDict())
    if key in cache:
        # move to end (LRU)
        cache.move_to_end(key)
        return cache[key]
    return None

def _qa_cache_put(q: str, a: str, cap: int = 64):
    key = (q or "").strip().lower()
    cache = st.session_state.setdefault("_qa_cache", OrderedDict())
    cache[key] = a
    cache.move_to_end(key)
    while len(cache) > cap:
        cache.popitem(last=False)

TOP_K = 3                 
CHUNK_CHAR_CAP = 900     
HISTORY_TURNS = 6  
_CODE_RE = re.compile(r"^[A-Za-z]{2,5}\s?\d{3}$")  # e.g., CSC101 or CSC 101

def _trim(s, n):
    s = (s or "").strip()
    return (s[:n] + "…") if len(s) > n else s

def prepare_history_text(messages, n_pairs=HISTORY_TURNS):
    """Keep only the last n_pairs (user) turns when building history text."""
    out, user_seen = [], 0
    for m in reversed(messages or []):
        out.append(m)
        if m.get("role") == "user":
            user_seen += 1
            if user_seen >= n_pairs:
                break
    out.reverse()
    return "\n".join(
        f"{m['role']}: {m.get('content','')}"
        for m in out if m.get("content")
    )

def fastpath_course_code(q, rows):
    """If prompt looks like a course code, answer directly from CSV (skip model)."""
    if not q or not _CODE_RE.match(q.strip()):
        return None
    key = re.sub(r"\s+", "", q).upper()
    for r in rows or []:
        code = re.sub(r"\s+", "", str(r.get("code",""))).upper()
        if code == key:
            title = (r.get("title") or "").strip()
            desc  = (r.get("description") or "").strip()
            return f"**{r.get('code','')} — {title}**\n\n{desc}"
    return None

def build_kb_from_docs(semantic_docs, bm25_docs, top_k=TOP_K, cap=CHUNK_CHAR_CAP):
    """Combine top docs from both retrievers and trim."""
    docs = []
    if semantic_docs: docs.extend(semantic_docs)
    if bm25_docs:     docs.extend(bm25_docs)
    docs = docs[:top_k]
    return "\n\n".join(
        f"[{i+1}] {_trim(getattr(d,'page_content',str(d)), cap)}"
        for i, d in enumerate(docs)
    )
# --- Secrets / keys (env + streamlit secrets) ---
def _get_secret(key: str) -> str | None:
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets[key]
    except Exception:
        return None

OPENAI_KEY = _get_secret("OPENAI_API_KEY")
GROQ_KEY   = _get_secret("GROQ_API_KEY")

if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY
if GROQ_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_KEY

try:
    import openai
    OpenAIRateLimitError = openai.RateLimitError
except Exception:
    class OpenAIRateLimitError(Exception): ...

# ---------------------- UI CONFIG ----------------------
page_icon = "RP.png" if os.path.exists("RP.png") else None
st.set_page_config(
    page_title="AUIBPT",
    page_icon=page_icon,
    layout="wide",
    menu_items={
        "About": "AUIBPT v1.5 — Course & Schedule assistant for AUIB.",
        "Get Help": "mailto:ali.1241375@auib.edu.iq",
        "Report a bug": "mailto:ali.1241375@auib.edu.iq",
    }
)

# Hide sidebar; widen main
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    .main .block-container { max-width: 1200px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- Global polish CSS ----------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Ubuntu, Arial, "Noto Sans", sans-serif;
        line-height: 1.55;
    }}
    .main .block-container {{
        max-width: 1100px;
        padding-top: 1.25rem;
    }}
    [data-testid="stChatMessage"] {{
        padding: 0.6rem 0.75rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.02);
        margin-bottom: 0.35rem;
    }}
    [data-testid="stChatMessage"] .stMarkdown p {{ margin-bottom: 0.35rem; }}
    .stButton>button, .stDownloadButton>button {{
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.15);
        transition: transform .02s ease-in;
    }}
    .stButton>button:active {{ transform: translateY(1px); }}
    pre, code {{ font-size: 0.93rem; }}
    hr {{ border-color: rgba(255,255,255,0.12) !important; }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- LOGGING ----------------------
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("app")

# ---------------------- CONSTANTS / REGEX ----------------------
COURSE_HINTS = [
    "course", "class", "prereq", "prerequisite", "credit", "credits",
    "catalog", "syllabus", "covers", "topic", "learn", "teaches",
    "semester", "enroll", "registration", "requirement", "requirements",
    "what is", "describe", "explain", "about"
]
COURSE_CODE_RE = re.compile(r"\b([A-Za-z]{2,4})\s*-?\s*(\d{3})\b")
NAME_RE = re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z0-9_\- ]{1,40})\b", re.IGNORECASE)

DEPT_PREFIXES = {
    "cs": ("CSC", "CSE"),
    "math": ("MAT", "MTH"),
    "stats": ("STA",),
    "physics": ("PHY", "PHYS"),
    "chem": ("CHE", "CHEM"),
    "bio": ("BIO", "BIOL"),
    "pharm": ("PHA",),
    "dent": ("BDS",),
}
KNOWN_COLLEGES = {"CAS", "COP", "COD"}
LANG_OPTIONS = {"English": "English", "Arabic": "Arabic"}

DEGREE_TOTAL = {"CS": 126, "Pharmacy": 180, "Dentistry": 189}
MAJOR_MAP = {
    "CS": {"college": "CAS", "prefixes": ("CSC", "MAT", "STA")},
    "Pharmacy": {"college": "COP", "prefixes": ("PHA", "CHE", "BIO")},
    "Dentistry": {"college": "COD", "prefixes": ("BDS", "BIO", "CHE")},
}




# ---------------------- PROMPTS ----------------------
COURSE_PROMPT = """
You are AUIBPT, a sharp and friendly university course assistant. Your vibe is upbeat, helpful, and concise.
Use ONLY the provided course knowledge base (kb). If an item is missing in kb, write "Unknown"—do not invent.
Keep it student-friendly and practical.

When relevant, factor in the student's completed courses from the section "student_profile" (e.g., prerequisites already satisfied, avoid recommending repeats).

Format your final answer exactly as:
- College: <college tag or 'Unknown'>
- Summary: <one lively sentence, ~15–25 words>
- Key topics: <comma-separated>
- Prerequisites: <text or 'Unknown'>
- Credits: <text or 'Unknown'>
- Source: <course code(s)>

Respond in {answer_lang}. Put ONLY your final answer inside <final>...</final>.

student_profile:
{student_context}

kb:
{kb}

history:
{history}

question:
{question}
"""

CHAT_PROMPT = """
You are AUIBPT, a friendly campus assistant and digital friend.
Keep your tone upbeat and conversational (≤3 sentences). 
Answer any type of question — about life, hobbies, events, news, or studies — not only courses.
Avoid offering or enrolling the user in courses unless they *explicitly* ask about classes or majors.
Be natural, concise, and supportive.

Respond in {answer_lang}. Put ONLY your final answer inside <final>...</final>.

student_profile:
{student_context}

history:
{history}

question:
{question}
"""


UNIV_PROMPT = """
You are AUIBPT, a friendly, factual assistant for AUIB. Use ONLY the supplied 'univ_kb' block.
If something is unknown, say "Unknown" briefly. Be concise, useful, and student-facing.

Personalize if relevant using 'student_profile' (e.g., suggest how a professor’s area aligns with the student’s path).

Format the final answer EXACTLY as:
Topic: <short topic>
Highlights: <1–2 sentences>
Details:
- <bullet 1>
- <bullet 2>
- <bullet 3>
Source: AUIB institutional KB

Respond in {answer_lang}. Put ONLY your final answer inside <final>...</final>.

student_profile:
{student_context}

univ_kb:
{univ_kb}

history:
{history}

question:
{question}
"""

# ---------------------- OUTPUT NORMALIZATION ----------------------
def _to_str(x) -> str:
    try:
        from langchain_core.messages import AIMessage
        if isinstance(x, AIMessage):
            return x.content or ""
    except Exception:
        pass
    return x if isinstance(x, str) else str(x)

def _clean_output(text: str) -> str:
    if not text:
        return "I don't know from the provided data."
    finals = re.findall(r"<final>(.*?)</final>", text, flags=re.DOTALL | re.IGNORECASE)
    if finals:
        text = next((blk.strip() for blk in reversed(finals) if blk.strip()), finals[-1].strip())
    else:
        m_open = re.search(r"<final>(.*)$", text, flags=re.DOTALL | re.IGNORECASE)
        if m_open:
            text = m_open.group(1).strip()
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?final\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r'(?is)\brules:\b.*', '', text)
    text = re.sub(r'(?is)\b(knowledge base|kb|instructions)\b.*', '', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "I don't know from the provided data."

# ---------------------- HELPERS ----------------------
def _norm_text(s: str) -> str:
    return (s or "").lower().translate(str.maketrans("", "", string.punctuation)).strip()

def _cap_history(n: int = 80):
    msgs = st.session_state.get("messages", [])
    if len(msgs) > n:
        st.session_state.messages = msgs[-n:]



def expand_synonyms(q: str) -> str:
    if not q:
        return q
    repl = {
        r"\bds\b": "data structures",
        r"\bdata struct(ure)?s?\b": "data structures",
        r"\balgo(rithms)?\b": "algorithms",
        r"\bai\b": "artificial intelligence",
        r"\bml\b": "machine learning",
        r"\bprog\b": "programming",
        r"\bpharm\b": "pharmacy",
        r"\bdent(al)?\b": "dentistry",
    }
    out = " " + q.lower() + " "
    for pat, sub in repl.items():
        out = re.sub(pat, sub, out)
    return out.strip()

def simple_token_overlap(a: str, b: str) -> int:
    A, B = set(_norm_text(a).split()), set(_norm_text(b).split())
    return len(A & B)

def rows_to_kb(rows_subset: List[Dict]) -> str:
    return "\n---\n".join(
        f"College: {(r.get('college') or 'Unknown')}\n"
        f"Course Code: {r['code']}\n"
        f"Title: {r['title']}\n"
        f"Description: {r['description']}\n"
        f"Prerequisites: {(r.get('prereqs') or 'Unknown')}\n"
        f"Credits: {(r.get('credits') or 'Unknown')}\n"
        f"[source: courses.csv | code: {r['code']}]"
        for r in rows_subset
    )

def is_coursey(q: str) -> bool:
    if not q:
        return False
    ql = q.lower()
    if COURSE_CODE_RE.search(q):
        return True
    return any(h in ql for h in COURSE_HINTS + ["csc", "mat", "mth", "sta", "pha", "che", "bio", "bds", "pharm", "dent"])

def maybe_capture_name(q: str) -> None:
    if not q:
        return
    m = NAME_RE.search(q)
    if m:
        st.session_state.user_name = m.group(1).strip()

def friendly_prefix() -> str:
    n = st.session_state.get("user_name")
    return f"{n}, " if n else ""

def build_history_text(max_turns: int = 10) -> str:
    msgs = st.session_state.messages[-2 * max_turns:] if st.session_state.messages else []
    lines = []
    for m in msgs:
        role = "User" if m["role"] == "user" else "AI"
        lines.append(f"{role}: {m['content']}")
    hist_msgs_text = prepare_history_text(st.session_state.get("messages", []), n_pairs=HISTORY_TURNS)
    history_text = hist_msgs_text
    return "\n".join(lines)

def find_rows_by_code(rows: List[Dict], q: str) -> List[Dict]:
    if not rows or not q:
        return []
    idx = {(r.get("code", "") or "").replace(" ", "").upper(): r for r in rows if "code" in r}
    hits, seen = [], set()
    for dept, num in COURSE_CODE_RE.findall(q):
        key = f"{dept.upper()}{num}"
        row = idx.get(key)
        if row and row["code"] not in seen:
            hits.append(row); seen.add(row["code"])
    return hits

def find_rows_by_title(rows: List[Dict], q: str) -> List[Dict]:
    if not rows or not q:
        return []
    q_tokens = set(_norm_text(q).split())
    best, best_score = None, 0
    for r in rows:
        tn = _norm_text(r.get("title", ""))
        if not tn:
            continue
        t_tokens = set(tn.split())
        overlap = len(q_tokens & t_tokens)
        if overlap > best_score and overlap >= 2:
            best, best_score = r, overlap
    return [best] if best else []

def infer_scopes(q: str) -> Dict[str, List[str]]:
    ql = (q or "").lower()
    dept_scopes: List[str] = []
    college_scopes: List[str] = []
    if any(w in ql for w in ["computer science", " comp sci", " cs ", " c.s.", "csc", "programming"]): dept_scopes.append("cs")
    if any(w in ql for w in ["math", "mathematics", " mat ", "mth", "algebra", "calculus"]): dept_scopes.append("math")
    if any(w in ql for w in ["statistics", " sta ", "probability"]): dept_scopes.append("stats")
    if any(w in ql for w in ["physics", "mechanics", "optics"]): dept_scopes.append("physics")
    if any(w in ql for w in ["chemistry", "organic", "inorganic", "che "]): dept_scopes.append("chem")
    if any(w in ql for w in ["biology", "bio", "genetics"]): dept_scopes.append("bio")
    if any(w in ql for w in ["pharmacy", "pharm", "pha"]): dept_scopes.append("pharm")
    if any(w in ql for w in ["dentistry", "dent", "bds"]): dept_scopes.append("dent")
    for tag in {"CAS","COP","COD"}:
        if tag.lower() in ql:
            college_scopes.append(tag)
    return {"dept": dept_scopes or ["all"], "college": college_scopes or ["all"]}

def filter_rows_by_college(rows: List[Dict], college_tag: str) -> List[Dict]:
    if college_tag == "All":
        return rows
    return [r for r in rows if (r.get("college","").upper() == college_tag)]

def reorder_docs_by_scopes(docs: List, scopes: Dict[str, List[str]], college_filter: str) -> List:
    if not docs:
        return docs
    colleges = set([c.upper() for c in scopes.get("college", []) if c != "all"])
    if college_filter != "All":
        colleges.add(college_filter.upper())
    dept_prefixes = tuple(p for s in scopes.get("dept", []) if s != "all" for p in DEPT_PREFIXES.get(s, ()))
    def score(doc):
        s = 0
        code = (doc.metadata.get("code") or "").upper()
        college = (doc.metadata.get("college") or "").upper()
        if colleges and college in colleges: s -= 2
        if dept_prefixes and code.startswith(dept_prefixes): s -= 1
        return s
    return sorted(docs, key=score)

def parse_catalog_intent(q: str) -> Dict | None:
    if not q:
        return None
    ql = q.lower()
    scopes = infer_scopes(q)
    if "how many" in ql and "course" in ql:
        return {"type": "count", "scopes": scopes}
    if any(kw in ql for kw in ["list all courses", "show all courses", "list courses", "all courses"]):
        return {"type": "list", "limit": 150, "scopes": scopes}
    return None



def hybrid_retrieve(q: str, retriever, vs, top_k: int, bm25=None) -> List:
    if not q:
        return []
    qx = expand_synonyms(q)
    vec_docs = []
    try:
        vec_docs = retriever.invoke(qx) or []
    except AttributeError:
        vec_docs = retriever.get_relevant_documents(qx) or []
    except Exception as e:
        log.error(f"Vector retrieve error: {e}")
    bm_docs = []
    if bm25 is not None:
        try:
            all_docs = list(vs.docstore._dict.values())
            scores = bm25.get_scores(qx.split())
            best_ids = sorted(range(len(all_docs)), key=lambda i: -scores[i])[:top_k]
            bm_docs = [all_docs[i] for i in best_ids]
        except Exception as e:
            log.warning(f"BM25 failed, continuing with vectors only: {e}")
    keyed = {}
    for d in (vec_docs + bm_docs):
        keyed[(d.metadata.get("code"), d.page_content)] = d
    merged = list(keyed.values())
    merged.sort(key=lambda d: -simple_token_overlap(qx, d.page_content))
    return merged[: max(top_k * 2, top_k)]

def prepare_kb_from_docs(docs) -> str:
    if not docs:
        return ""
    blocks = []
    for d in docs:
        meta = d.metadata or {}
        text = d.page_content
        blocks.append(text + f"\n[source: {meta.get('source','?')} | code: {meta.get('code','?')}]")
    return "\n---\n".join(blocks).strip()

# ---------------------- LLM & STREAMING ----------------------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder): self.placeholder = placeholder; self.text = ""
    def on_llm_new_token(self, token, **_): self.text += token; self.placeholder.markdown(self.text)

def _make_openai_llm(model_name: str, temperature: float, num_predict: int, callbacks=None):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=num_predict,
        streaming=True,
        callbacks=callbacks or [],
        max_retries=8,
        timeout=60.0,
    )

def _make_groq_llm(model_name: str, temperature: float, num_predict: int, callbacks=None):
    from langchain_groq import ChatGroq
    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
        max_tokens=num_predict,
        streaming=True,
        callbacks=callbacks or [],
        max_retries=8,
        timeout=60.0,
    )

def _make_ollama_llm(model_name: str, temperature: float, num_predict: int, callbacks=None):
    from langchain_ollama import OllamaLLM
    return OllamaLLM(
        model=model_name,
        temperature=temperature,
        num_predict=num_predict,
        stop=["</final>"],
        callbacks=callbacks or [],
    )

def make_llm(model_name: str, temperature: float, num_predict: int, callbacks=None):
    effective_use_openai = bool(OPENAI_KEY) and USE_OPENAI
    if effective_use_openai:
        try:
            return _make_openai_llm(model_name, temperature, num_predict, callbacks)
        except Exception as e:
            st.warning(f"OpenAI init failed: {e}. Trying Groq…")
    if GROQ_KEY:
        try:
            groq_model = model_name
            if "gpt-" in model_name.lower():
                groq_model = "llama-3.1-8b-instant"
            return _make_groq_llm(groq_model, temperature, num_predict, callbacks)
        except Exception as e:
            st.warning(f"Groq init failed: {e}. Trying Ollama…")
    try:
        return _make_ollama_llm(model_name, temperature, num_predict, callbacks)
    except Exception:
        from langchain_openai import ChatOpenAI
        st.warning("No Groq key and Ollama not available. Using OpenAI mini as last resort.")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            max_tokens=num_predict,
            streaming=True,
            callbacks=callbacks or [],
            max_retries=8,
            timeout=60.0,
        )

def ask_llm_stream(chain, kb: str, history_text: str, q: str, answer_lang: str, student_context: str, placeholder, univ_kb: str = "", groq_fallback_chain=None) -> str:
    handler = StreamHandler(placeholder)
    payload = {
        "kb": kb,
        "univ_kb": univ_kb,
        "history": history_text,
        "question": q,
        "answer_lang": answer_lang,
        "student_context": student_context
    }
    if USE_GROQ_ONLY and groq_fallback_chain is not None:
        chain, groq_fallback_chain = groq_fallback_chain, None

    def _invoke(c, stream=True):
        if stream:
            return c.invoke(payload, config={"callbacks": [handler]})
        else:
            return c.invoke(payload)

    try:
        raw = _invoke(chain, stream=True)
        final_text = _clean_output(_to_str(raw).strip())
        placeholder.markdown(final_text)
        return final_text
    except OpenAIRateLimitError as e:
        if groq_fallback_chain is not None:
            st.info("Switching to Groq due to OpenAI rate limit.")
            try:
                raw = _invoke(groq_fallback_chain, stream=True)
                final_text = _clean_output(_to_str(raw).strip())
                placeholder.markdown(final_text)
                return final_text
            except Exception as ee:
                last_err = ee
        else:
            last_err = e
    except Exception as e:
        try:
            raw = _invoke(chain, stream=False)
            final_text = _clean_output(_to_str(raw).strip())
            placeholder.markdown(final_text)
            return final_text
        except Exception as ee:
            last_err = ee

    if groq_fallback_chain is not None:
        try:
            raw = _invoke(groq_fallback_chain, stream=False)
            final_text = _clean_output(_to_str(raw).strip())
            placeholder.markdown(final_text)
            return final_text
        except Exception as ee:
            last_err = ee

    log.error(f"LLM failure (after Groq fallback if any): {last_err}")
    msg = "We’re a bit busy right now. Please try again in ~30–60s."
    placeholder.warning(msg)
    return msg

# ---------------------- SPLASH ----------------------
def show_splash():
    st.markdown("""
        <style>
        @keyframes fadeInOut { 0%{opacity:0} 10%{opacity:1} 80%{opacity:1} 100%{opacity:0} }
        .splash-container{
            position:fixed; z-index:9999; inset:0; display:flex; align-items:center; justify-content:center;
            background: radial-gradient(circle at 50% 50%, #0b1220 0%, #050812 60%, #000 100%);
            animation: fadeInOut 1.6s ease-in-out forwards;
        }
        .splash-title{ font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial; font-size:40px; letter-spacing:2px; color:#e2e8f0; text-shadow:0 2px 12px rgba(255,255,255,0.25); }
        </style>
        <div class="splash-container"><div class="splash-title">Ryunix Productions</div></div>
    """, unsafe_allow_html=True)
    time.sleep(1.65)
    st.session_state._splash_shown = True
    st.rerun()


if "theme_primary" not in st.session_state: st.session_state.theme_primary = "#4d1212"
if "theme_bg" not in st.session_state:      st.session_state.theme_bg = "#000000"
if "theme_text" not in st.session_state:    st.session_state.theme_text = "#e2e8f0"
apply_theme(st.session_state.theme_primary, st.session_state.theme_bg, st.session_state.theme_text)

if "theme_primary" not in st.session_state: st.session_state.theme_primary = "#4d1212"
if "theme_bg" not in st.session_state:      st.session_state.theme_bg = "#000000"
if "theme_text" not in st.session_state:    st.session_state.theme_text = "#FFFFFF"




# ---------------------- SESSION ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "_splash_shown" not in st.session_state:
    st.session_state._splash_shown = False
if "profile_avatar_path" not in st.session_state:
    st.session_state.profile_avatar_path = None



if "schedule_slots" not in st.session_state:
    st.session_state.schedule_slots = []
if "schedule_planned_credits" not in st.session_state:
    st.session_state.schedule_planned_credits = 0
if "schedule_target_credits" not in st.session_state:
    st.session_state.schedule_target_credits = 15
if "schedule_major_key" not in st.session_state:
    st.session_state.schedule_major_key = "CS"
if "completed_codes_all" not in st.session_state:
    st.session_state.completed_codes_all = set()
if "swap_history" not in st.session_state:
    st.session_state.swap_history = []
if "show_schedule" not in st.session_state:
    st.session_state.show_schedule = False
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

if not st.session_state._splash_shown:
    show_splash()


status_col1, status_col2, status_col3 = st.columns([1.3, 1, 1])
with status_col1:
    st.markdown("AUIBPT")
with status_col1:
    st.caption(f"Model: `{MODEL_NAME}` • k={TOP_K} • T={TEMPERATURE} • max={NUM_PREDICT}")
with status_col3:
    with st.expander("Appearance", expanded=False):
        render_appearance_controls()
    with st.expander("Settings", expanded=False):
        if "answer_lang" not in st.session_state:
            st.session_state.answer_lang = "English"
        if "debug" not in st.session_state:
            st.session_state.debug = False
        st.session_state.answer_lang = st.selectbox("Answer language", ["English","Arabic"],
        index=["English","Arabic"].index(st.session_state.answer_lang))
        st.session_state.debug = st.toggle("Debug", value=st.session_state.debug,
        help="Show knowledge base and timing details")
        answer_lang = st.session_state.answer_lang
        debug = st.session_state.debug




        # College filter
        try:
            colleges = sorted(list(KNOWN_COLLEGES))
        except Exception:
            colleges = []
        options = ["All"] + colleges
        if "college_filter" not in st.session_state:
            st.session_state.college_filter = "All"
        try:
            _idx = options.index(st.session_state.college_filter)
        except ValueError:
            _idx = 0
        college_filter = st.selectbox("College filter", options, index=_idx)
        st.session_state.college_filter = college_filter

    toggle_label = "Close Schedule Builder" if st.session_state.get("show_schedule", False) else "Open Schedule Builder"
    cols_hdr = st.columns(2)
    with cols_hdr[0]:
        if st.button(toggle_label, key="toggle_schedule_hdr"):
            st.session_state.show_schedule = not st.session_state.get("show_schedule", False)
            st.rerun()
    with cols_hdr[1]:
        clear_clicked = st.button("Clear chat")

    exists = os.path.exists(CSV_PATH)
    st.caption(f"CSV: {'found' if exists else 'missing'}")

with status_col1:
    st.caption("BETA — AUIBPT (Ryunix Build)")
with status_col3:
    force_rebuild = st.button("Reset")
    if 'force_rebuild' in locals() and force_rebuild:
        build_or_load_index(CSV_PATH, INDEX_DIR, force=True)
        st.success("Index reset.")
        st.session_state.pop("index_ready", None)
        st.rerun()
try:
    rows_all, vs, bm25 = build_or_load_index(CSV_PATH, INDEX_DIR, force=False)
    st.caption(f"Loaded {len(rows_all)} courses • Vector index ready ✓") #healthy or not (delete later)

    college_filter = st.session_state.get("college_filter", "All")
    rows = filter_rows_by_college(rows_all, college_filter)
    retriever = vs.as_retriever(search_kwargs={"k": int(TOP_K)})
except Exception as e:
    st.error(f"Failed to prepare index or load catalog: {e}")
    st.stop()

llm = make_llm(MODEL_NAME, TEMPERATURE, NUM_PREDICT)

course_chain = ChatPromptTemplate.from_template(COURSE_PROMPT) | llm
chat_chain   = ChatPromptTemplate.from_template(CHAT_PROMPT)   | llm
univ_chain   = ChatPromptTemplate.from_template(UNIV_PROMPT)   | llm

groq_fallback_course = None
groq_fallback_chat   = None
groq_fallback_univ   = None
if GROQ_KEY:
    try:
        groq_llm = _make_groq_llm("llama-3.1-8b-instant", TEMPERATURE, NUM_PREDICT)
        groq_fallback_course = ChatPromptTemplate.from_template(COURSE_PROMPT) | groq_llm
        groq_fallback_chat   = ChatPromptTemplate.from_template(CHAT_PROMPT)   | groq_llm
        groq_fallback_univ   = ChatPromptTemplate.from_template(UNIV_PROMPT)   | groq_llm
    except Exception as e:
        st.warning(f"Groq fallback unavailable: {e}")

# ---------------------- LIBERAL ARTS (rules) ----------------------
LA_REQUIREMENTS = {
    "General": 1,
    "Communication": 3,
    "Quantitative": 2,   # must be CSC101 and MAT101
    "Humanities": 4,
    "SocialScience": 2,
    "NaturalScience": 2,
}
LA_CATEGORY = {
    "UNI101": "General",
    "ENL101": "Communication", "ENL201": "Communication", "ENL210": "Communication",
    "CSC101": "Quantitative", "MAT101": "Quantitative",
    "HIS101": "Humanities", "HIS102": "Humanities", "HIS105": "Humanities",
    "HUM101": "Humanities", "LIT101": "Humanities", "PHA210": "Humanities",
    "PHI101": "Humanities", "POL125": "Humanities",
    "TLD100": "Humanities", "TLD101": "Humanities", "TLD102": "Humanities", "TLD103": "Humanities",
    "COM101": "SocialScience", "ECO101": "SocialScience", "FIN101": "SocialScience",
    "HCT108": "SocialScience", "MIS101": "SocialScience", "POL101": "SocialScience",
    "POL112": "SocialScience", "POL191": "SocialScience", "PSY101": "SocialScience",
    "SOC101": "SocialScience",
    "CHE100": "NaturalScience", "ENV201": "NaturalScience", "GEO101": "NaturalScience",
    "PHY100": "NaturalScience", "PHY105": "NaturalScience",
}
LA_QUANT_BOTH = {"CSC101", "MAT101"}
MAJOR_WEIGHT = 3
LA_WEIGHT = 1

DIFFICULTY_WEIGHT_MAP = {
    "Easy":   (2.0, 1.0),
    "Medium": (3.0, 1.0),
    "Hard":   (4.0, 1.0),
}

def get_semester_weights(difficulty: str) -> tuple[float, float]:
    return DIFFICULTY_WEIGHT_MAP.get(difficulty, DIFFICULTY_WEIGHT_MAP["Medium"])

# ---------------------- PREREQS / CREDITS ----------------------
def _parse_prereq_codes(prereq_text: str) -> List[str]:
    if not prereq_text:
        return []
    codes = []
    parts = re.split(r"[;,/]+", prereq_text)
    for p in parts:
        for m in re.finditer(r"\b([A-Za-z]{2,4})\s*-?\s*(\d{3})\b", p):
            codes.append((m.group(1) + m.group(2)).upper())
    return sorted(set(codes))

def _credits_from_str(x: str) -> int:
    if x is None: return 3
    s = str(x).strip()
    if not s: return 3
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else 3

# ---------------------- LA COUNTS ----------------------
def la_completed_counts(taken_codes: Set[str]) -> Dict[str, int]:
    counts = {k: 0 for k in LA_REQUIREMENTS}
    for code in taken_codes:
        cat = LA_CATEGORY.get(code.upper())
        if cat:
            counts[cat] += 1
    return counts

def la_remaining(counts: Dict[str, int], taken_codes: Set[str]) -> Dict[str, int]:
    remain = {}
    for cat, need in LA_REQUIREMENTS.items():
        have = counts.get(cat, 0)
        remain[cat] = max(0, need - have)
    have_both = LA_QUANT_BOTH.issubset({c.upper() for c in taken_codes})
    if not have_both:
        missing = len(LA_QUANT_BOTH - {c.upper() for c in taken_codes})
        remain["Quantitative"] = missing
    else:
        remain["Quantitative"] = 0
    return remain

def la_recommend_pool(taken_codes: Set[str], rows_scope: List[Dict], remain: Dict[str, int]) -> Dict[str, List[Dict]]:
    taken_codes = {c.upper() for c in taken_codes}
    code_to_row = {r["code"].upper(): r for r in rows_scope}
    by_cat: Dict[str, List[Dict]] = {k: [] for k in LA_REQUIREMENTS}
    for code, cat in LA_CATEGORY.items():
        r = code_to_row.get(code)
        if not r: continue
        if remain.get(cat, 0) <= 0: continue
        if code in taken_codes: continue
        req_codes = _parse_prereq_codes(r.get("prereqs",""))
        if not all(rc in taken_codes for rc in req_codes): continue
        by_cat[cat].append(r)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda r: (int(re.search(r"(\d{3})$", r["code"]).group(1)) if re.search(r"(\d{3})$", r["code"]) else 999, r.get("title","")))
    return by_cat

def _eligible_major_rows(taken_codes: Set[str], rows_scope: List[Dict], prefixes: Tuple[str,...]) -> List[Dict]:
    taken_codes = {c.upper() for c in taken_codes}
    pool = []
    for r in rows_scope:
        code = r["code"].upper()
        if code in taken_codes: continue
        if not code.startswith(prefixes): continue
        req_codes = _parse_prereq_codes(r.get("prereqs",""))
        if req_codes and not all(rc in taken_codes for rc in req_codes): continue
        pool.append(r)
    def score(r):
        req = _parse_prereq_codes(r.get("prereqs",""))
        lvl_m = re.search(r"(\d{3})$", r["code"].upper())
        lvl = int(lvl_m.group(1)) if lvl_m else 0
        return (-len(req), lvl, r.get("title",""))
    return sorted(pool, key=score)

def _credits_completed(taken_codes: Set[str], rows_all: List[Dict]) -> int:
    idx = {r["code"].upper(): r for r in rows_all}
    total = 0
    for c in taken_codes:
        r = idx.get(c.upper())
        if r:
            total += _credits_from_str(r.get("credits"))
    return total

# ---------------------- STUDENT PROFILE CONTEXT ----------------------
def student_context_from_taken(rows_all: List[Dict], taken_codes: Set[str]) -> str:
    if not taken_codes:
        return "Completed: (none)"
    idx = {r["code"].upper(): r for r in rows_all}
    items = []
    for c in sorted({x.upper() for x in taken_codes}):
        r = idx.get(c)
        if not r:
            continue
        cr = r.get("credits") or ""
        title = r.get("title") or ""
        items.append(f"{c} ({title}; {cr} cr)")
    completed_credits = _credits_completed(taken_codes, rows_all)
    return "Completed (" + str(completed_credits) + " credits): " + "; ".join(items)

# ----------------------  SCHEDULE BUILDER (+ swap) ----------------------
# <<<AUIBPT:MAJ_LA_COUNTS>>>
def build_semester_schedule(
    major_key: str,
    target_credits: int,
    taken_codes: Set[str],
    rows_all: List[Dict],
    difficulty: str,
    desired_major: Optional[int] = None,
    desired_la: Optional[int] = None,
) -> Tuple[List[Dict], Dict[str,int], Dict[str,int], int]:
    """
    Build a semester plan subject to:
      - target_credits (hard cap),
      - explicit desired_major / desired_la counts (if provided),
      - otherwise fall back to a difficulty-based ratio.

    Always respects prerequisites and LA category availability,
    and auto-includes CSC101/MAT101 if 'Quantitative' is still missing.
    """
    used_codes: Set[str] = set(c.upper() for c in taken_codes)
    code_to_row = {r["code"].upper(): r for r in rows_all}

    # LA state & pools
    la_counts = la_completed_counts(used_codes)
    la_remain = la_remaining(la_counts, used_codes)
    la_pool_by_cat = la_recommend_pool(used_codes, rows_all, la_remain)

    la_flat: List[Tuple[str, Dict]] = []
    for cat, lst in la_pool_by_cat.items():
        for r in lst:
            la_flat.append((cat, r))

    major_info = MAJOR_MAP[major_key]
    major_pool = _eligible_major_rows(used_codes, rows_all, major_info["prefixes"])

    def cr_of(r: Dict) -> int:
        return _credits_from_str(r.get("credits"))

    schedule: List[Dict] = []
    cur_credits = 0

    def try_add_row(r: Dict, origin: str) -> bool:
        nonlocal cur_credits
        cu = r["code"].upper()
        if cu in used_codes:
            return False
        # prereqs gate
        reqs = _parse_prereq_codes(r.get("prereqs",""))
        if any(rc not in used_codes for rc in reqs):
            return False
        c = cr_of(r)
        if cur_credits + c > target_credits:
            return False
        schedule.append(r)
        used_codes.add(cu)
        cur_credits += c
        return True

    # Ensure Quantitative pair if still missing (CSC101, MAT101).
    for q_code in ["CSC101", "MAT101"]:
        if la_remain.get("Quantitative", 0) > 0 and q_code not in used_codes:
            rq = code_to_row.get(q_code)
            if rq and try_add_row(rq, origin="LA:Quantitative"):
                # refresh LA pools after adding
                la_counts = la_completed_counts(used_codes)
                la_remain = la_remaining(la_counts, used_codes)
                la_pool_by_cat = la_recommend_pool(used_codes, rows_all, la_remain)
                la_flat = []
                for cat, lst in la_pool_by_cat.items():
                    for r in lst:
                        la_flat.append((cat, r))

    # --- Decide major/LA goals ---
    # If explicit counts provided, honor them; otherwise compute from difficulty.
    if desired_major is not None or desired_la is not None:
        # If only one side provided, infer the other from a rough slots estimate.
        avg_cr = 3  # typical
        est_slots = max(1, min(7, target_credits // avg_cr))
        if desired_major is None and desired_la is not None:
            desired_major = max(0, est_slots - int(desired_la))
        if desired_la is None and desired_major is not None:
            desired_la = max(0, est_slots - int(desired_major))
        desired_major = max(0, int(desired_major or 0))
        desired_la    = max(0, int(desired_la or 0))
    else:
        # Difficulty → ratio → approximate counts
        major_weight, la_weight = get_semester_weights(difficulty)
        avg_cr = 3
        total_slots = max(1, min(7, target_credits // avg_cr))
        # Split by weights
        if major_weight + la_weight <= 0:
            major_goal = total_slots
            la_goal = 0
        else:
            major_goal = round(total_slots * (major_weight / (major_weight + la_weight)))
            la_goal = total_slots - major_goal
        desired_major, desired_la = max(0, major_goal), max(0, la_goal)

    major_remaining = int(desired_major)
    la_remaining_goal = int(desired_la)

    # Main fill loop
    guard = 0
    MAX_ITERS = 1000
    while cur_credits < target_credits and guard < MAX_ITERS:
        guard += 1

        # refresh pools each iteration
        major_pool = [r for r in _eligible_major_rows(used_codes, rows_all, major_info["prefixes"])
                      if r["code"].upper() not in used_codes]

        la_counts = la_completed_counts(used_codes)
        la_remain = la_remaining(la_counts, used_codes)
        la_pool_by_cat = la_recommend_pool(used_codes, rows_all, la_remain)
        la_flat = [(cat, r) for cat, lst in la_pool_by_cat.items() for r in lst
                   if r["code"].upper() not in used_codes]

        # If user asked for X LA courses but there are none needed left, we can still add LA as free electives.
        la_any_available = bool(la_flat)

        picked = False


        def pick_major_first() -> bool:
            if major_remaining > 0 and la_remaining_goal > 0:
                # If specific LA categories still needed, prefer LA.
                if any(v > 0 for v in la_remain.values()):
                    return False
                return True
            if major_remaining > 0:
                return True
            if la_remaining_goal > 0:
                return False
            # both goals satisfied → pick by availability
            return bool(major_pool)  # True if major_pool has items, else LA

        try_major = pick_major_first()

        for bucket in (["major", "la"] if try_major else ["la", "major"]):
            if bucket == "major":
                if major_remaining <= 0 or not major_pool:
                    continue
                for r in major_pool:
                    if try_add_row(r, origin="Major"):
                        major_remaining -= 1
                        picked = True
                        break
                if picked:
                    break
            else:
                # LA bucket
                if la_remaining_goal <= 0 or not la_any_available:
                    continue
                for (cat, r) in la_flat:
                    
                    if la_remain.get(cat, 0) <= 0 and any(v > 0 for v in la_remain.values()):
                        continue
                    if try_add_row(r, origin=f"LA:{cat}"):
                        la_remaining_goal -= 1
                        picked = True
                        break
                if picked:
                    break

        if not picked:
        
            for r in major_pool:
                if try_add_row(r, origin="Major"):
                    picked = True
                    break
            if not picked:
                for (cat, r) in la_flat:
                    if try_add_row(r, origin=f"LA:{cat}"):
                        picked = True
                        break

        if not picked:
            
            break

        
        if major_remaining <= 0 and la_remaining_goal <= 0:
            
            if cur_credits >= target_credits - 2:
                break

    return schedule, la_counts, la_remain, cur_credits


def _rebuild_pools(major_key: str, taken_codes: Set[str], rows_all: List[Dict]) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    la_counts = la_completed_counts(taken_codes)
    la_remain = la_remaining(la_counts, taken_codes)
    la_pool = la_recommend_pool(taken_codes, rows_all, la_remain)
    major_pool = _eligible_major_rows(taken_codes, rows_all, MAJOR_MAP[major_key]["prefixes"])
    return la_pool, major_pool

def _export_schedule_csv(slots: List[Dict]) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["code", "title", "credits", "category", "prereqs"])
    for s in slots:
        r = s["candidates"][s["current_idx"]]
        writer.writerow([r["code"], r["title"], r.get("credits") or "", s["origin"], r.get("prereqs") or ""])
    return output.getvalue().encode("utf-8")

def export_schedule_json(slots):
    data = [{
        "origin": s["origin"],
        "current_idx": s["current_idx"],
        "candidates": [{
            "code": r["code"], "title": r["title"],
            "credits": r.get("credits"), "prereqs": r.get("prereqs")
        } for r in s["candidates"]]
    } for s in slots]
    return json.dumps({"version":"1.0","slots":data}, ensure_ascii=False, indent=2).encode("utf-8")

def import_schedule_json(payload_bytes):
    obj = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(obj, dict) or "slots" not in obj:
        raise ValueError("Invalid schedule file.")
    new_slots = []
    for s in obj["slots"]:
        if not {"origin","current_idx","candidates"} <= set(s):
            continue
        cand_rows = []
        for r in s["candidates"]:
            if "code" in r and "title" in r:
                cand_rows.append({"code":r["code"],"title":r["title"],"credits":r.get("credits"),"prereqs":r.get("prereqs")})
        if cand_rows:
            new_slots.append({
                "id": f"import-{len(new_slots)}",
                "origin": s["origin"],
                "candidates": cand_rows,
                "current_idx": int(s["current_idx"]),
                "locked": bool(s.get("locked", False)),
            })
    return new_slots

def _auto_top_up(major_key: str, target_credits: int, taken_codes: Set[str], slots: List[Dict], rows_all: List[Dict]) -> None:
    current_total = sum(_credits_from_str(s["candidates"][s["current_idx"]].get("credits")) for s in slots)
    if current_total >= target_credits:
        return
    used = set(taken_codes) | {s["candidates"][s["current_idx"]]["code"].upper() for s in slots}
    la_pool, major_pool = _rebuild_pools(major_key, used, rows_all)

    la_candidates = [r for pool in la_pool.values() for r in pool if r["code"].upper() not in used]
    major_candidates = [r for r in major_pool if r["code"].upper() not in used]
    for origin, cand_list in [("LA:Any", la_candidates), ("Major", major_candidates)]:
        for r in cand_list:
            cr = _credits_from_str(r.get("credits"))
            if current_total + cr <= target_credits:
                slots.append({
                    "id": f"extra-{origin}-{len(slots)}",
                    "origin": origin,
                    "candidates": [r],
                    "current_idx": 0,
                    "locked": False,
                })
                used.add(r["code"].upper())
                current_total += cr
                if current_total >= target_credits:
                    return

def _toggle_lock(slot):
    if "locked" not in slot:
        slot["locked"] = True
    else:
        slot["locked"] = not slot["locked"]

def _push_swap(slot_idx, prev_idx):
    st.session_state.swap_history.append((slot_idx, prev_idx))

def _undo_swap():
    if not st.session_state.swap_history:
        return False
    slot_idx, prev_idx = st.session_state.swap_history.pop()
    if 0 <= slot_idx < len(st.session_state.schedule_slots):
        st.session_state.schedule_slots[slot_idx]["current_idx"] = prev_idx
        return True
    return False

# ---------------------- SCHEDULE BUILDER (function) ----------------------
def render_schedule_builder(rows_all, vs, bm25):
    st.markdown("### Schedule Builder")

    # --- difficulty incl. CUSTOM ---
    if "schedule_difficulty" not in st.session_state:
        st.session_state.schedule_difficulty = "Medium"  # used when NOT in Custom
    if "schedule_custom_mode" not in st.session_state:
        st.session_state.schedule_custom_mode = False

    diff_opts = ["Easy", "Medium", "Hard", "Custom"]
    current_choice = "Custom" if st.session_state.schedule_custom_mode else st.session_state.schedule_difficulty
    difficulty_choice = st.radio(
        "Semester difficulty",
        options=diff_opts,
        index=diff_opts.index(current_choice),
        horizontal=True,
        help=("Easy increases Liberal Arts weight; Medium balanced; "
              "Hard increases Major weight. Choose Custom to specify exact counts.")
    )
    # Persist selection
    if difficulty_choice == "Custom":
        st.session_state.schedule_custom_mode = True
        # keep previous non-custom choice in st.session_state.schedule_difficulty
    else:
        st.session_state.schedule_custom_mode = False
        st.session_state.schedule_difficulty = difficulty_choice

    # --- major / program select (always visible) ---
    options = sorted(MAJOR_MAP.keys())
    current = get_current_major_key()
    major_key = st.selectbox(
        "Major / program",
        options,
        index=options.index(current) if current in options else 0,
        help="Choose which major to plan for."
    )
    st.session_state.schedule_major_key = major_key
    _major_key = get_current_major_key()  # safe use below

    # --- target credits (always visible) ---
    if "schedule_target_credits" not in st.session_state:
        st.session_state.schedule_target_credits = 15
    target_credits = st.slider(
        "Target credits",
        min_value=9, max_value=21, value=int(st.session_state.schedule_target_credits), step=1,
        help="Maximum credits to take this term."
    )
    st.session_state.schedule_target_credits = target_credits

    # --- Major/Liberal counts (visible ONLY in Custom) ---
    if st.session_state.schedule_custom_mode:
        cols_counts = st.columns(2)
        with cols_counts[0]:
            if "schedule_major_count" not in st.session_state:
                st.session_state.schedule_major_count = 3
            st.session_state.schedule_major_count = st.number_input(
                "Major courses this term", min_value=0, max_value=7, step=1,
                value=int(st.session_state.schedule_major_count),
                help="Exact number of major courses you want to take."
            )
        with cols_counts[1]:
            if "schedule_la_count" not in st.session_state:
                st.session_state.schedule_la_count = 2
            st.session_state.schedule_la_count = st.number_input(
                "Liberal Arts courses this term", min_value=0, max_value=7, step=1,
                value=int(st.session_state.schedule_la_count),
                help="Exact number of liberal arts courses you want to take."
            )
    else:
        # ensure keys exist even when hidden
        if "schedule_major_count" not in st.session_state:
            st.session_state.schedule_major_count = 3
        if "schedule_la_count" not in st.session_state:
            st.session_state.schedule_la_count = 2

    # --- Completed-courses picker (always visible) ---
    picker_scope = st.radio(
        "Completed-course picker scope:",
        ["Major only", "Liberal Arts only", "Both"],
        horizontal=True,
    )

    major_prefixes = MAJOR_MAP[_major_key]["prefixes"]
    major_only_rows = [r for r in rows_all if r["code"].upper().startswith(major_prefixes)]
    la_only_rows    = [r for r in rows_all if r["code"].upper() in LA_CATEGORY]

    if picker_scope == "Major only":
        picker_rows = major_only_rows
    elif picker_scope == "Liberal Arts only":
        picker_rows = la_only_rows
    else:
        seen = set()
        picker_rows = []
        for r in major_only_rows + la_only_rows:
            cu = r["code"].upper()
            if cu not in seen:
                picker_rows.append(r); seen.add(cu)

    labels = [f"{r['code']} — {r['title']}" for r in picker_rows]
    label_to_code = {f"{r['code']} — {r['title']}": r["code"].upper() for r in picker_rows}
    visible_codes = set(label_to_code.values())

    if "completed_codes_all" not in st.session_state:
        st.session_state.completed_codes_all = set()

    preselected_labels = [lbl for lbl, code in label_to_code.items() if code in st.session_state.completed_codes_all]
    picked_labels = st.multiselect("I have completed:", labels, default=preselected_labels, key="completed_picker")
    picked_visible_codes = {label_to_code[lbl] for lbl in picked_labels}

    hidden_kept = st.session_state.completed_codes_all - visible_codes
    st.session_state.completed_codes_all = hidden_kept | picked_visible_codes
    taken_codes_all = set(st.session_state.completed_codes_all)

    completed_credits = _credits_completed(taken_codes_all, rows_all)
    degree_total = DEGREE_TOTAL.get(_major_key, 126)
    st.caption(f"Progress: {completed_credits} / {degree_total} credits • Target this term: {target_credits}")

    # --- action buttons (always visible) ---
    col_build_a, col_build_b, col_build_c, col_build_d, col_build_e = st.columns([0.35,0.2,0.2,0.15,0.1])
    with col_build_a:
        build_btn = st.button("Build schedule", use_container_width=True)
    with col_build_b:
        reset_btn = st.button("Reset", use_container_width=True)
    with col_build_c:
        topup_btn = st.button("Auto top-up", use_container_width=True, disabled=not st.session_state.get("schedule_slots"))
    with col_build_d:
        undo_btn = st.button("Undo swap", use_container_width=True, disabled=not st.session_state.get("swap_history"))
    with col_build_e:
        if st.button("Close", use_container_width=True):
            st.session_state.show_schedule = False
            st.rerun()

    if reset_btn:
        st.session_state.schedule_slots = []
        st.session_state.schedule_planned_credits = 0
        st.rerun()

    # --- build schedule (uses difficulty unless Custom; respects credit cap) ---
    if build_btn:
        desired_major = st.session_state.schedule_major_count if st.session_state.schedule_custom_mode else None
        desired_la    = st.session_state.schedule_la_count    if st.session_state.schedule_custom_mode else None

        schedule, la_counts, la_remain, planned_credits = build_semester_schedule(
            major_key=_major_key,
            target_credits=target_credits,
            taken_codes=taken_codes_all,
            rows_all=rows_all,
            difficulty=st.session_state.get("schedule_difficulty", "Medium"),  # used when not Custom
            desired_major=desired_major,  # used when Custom
            desired_la=desired_la,        # used when Custom
        )

        # rebuild pools for slot candidates
        la_pool, major_pool = _rebuild_pools(_major_key, taken_codes_all, rows_all)

        slots = []
        used = set(taken_codes_all)
        for idx, c in enumerate(schedule):
            if c["code"].upper() in LA_CATEGORY:
                origin = f"LA:{LA_CATEGORY[c['code'].upper()]}"
                pool = la_pool.get(LA_CATEGORY[c["code"].upper()], [])
            else:
                origin = "Major"
                pool = major_pool

            candidates = []
            seen_codes = set()
            candidates.append(c); seen_codes.add(c["code"].upper())
            for r in pool:
                cu = r["code"].upper()
                if cu not in seen_codes and cu not in used:
                    candidates.append(r); seen_codes.add(cu)

            slots.append({
                "id": f"{origin}-{idx}",
                "origin": origin,
                "candidates": candidates,
                "current_idx": 0,
                "locked": False,
            })
            used.add(c["code"].upper())

        st.session_state.schedule_slots = slots
        st.session_state.schedule_planned_credits = sum(_credits_from_str(s["candidates"][0].get("credits")) for s in slots)

    # --- top-up & undo ---
    if topup_btn and st.session_state.get("schedule_slots"):
        _auto_top_up(_major_key, target_credits, taken_codes_all, st.session_state.schedule_slots, rows_all)
        st.session_state.schedule_planned_credits = sum(_credits_from_str(s["candidates"][s["current_idx"]].get("credits")) for s in st.session_state.schedule_slots)
        st.rerun()

    if undo_btn:
        if _undo_swap():
            st.rerun()
        else:
            st.info("Nothing to undo.")

    # --- render current schedule ---
    if st.session_state.get("schedule_slots"):
        st.markdown("### Suggested schedule")
        new_total = 0

        for i, slot in enumerate(st.session_state.schedule_slots):
            cur = slot["candidates"][slot["current_idx"]]
            pr = cur.get("prereqs") or "None/Unknown"
            cr = cur.get("credits") or "Unknown"

            cols = st.columns([0.64, 0.12, 0.12, 0.12])
            with cols[0]:
                st.markdown(
                    f"**{cur['code']} — {cur['title']}**  \n"
                    f"Category: {slot['origin']} • Credits: {cr} • Prereqs: {pr}  \n"
                    f"Status: {'Locked' if slot.get('locked') else 'Unlocked'}"
                )
                why_bits = []
                if cur["code"].upper() in LA_CATEGORY:
                    why_bits.append(f"meets {LA_CATEGORY[cur['code'].upper()]}")
                else:
                    why_bits.append("major requirement/elective")
                reqs = _parse_prereq_codes(cur.get("prereqs",""))
                if not reqs:
                    why_bits.append("no explicit prerequisites")
                else:
                    if all(rc in taken_codes_all for rc in reqs):
                        why_bits.append("prerequisites satisfied")
                    else:
                        why_bits.append("prerequisites satisfied during planning")
                st.caption("Why this? " + " • ".join(why_bits))

            with cols[1]:
                if st.button(
                    "Swap",
                    help="Replace with the next eligible option",
                    key=f"swap_{slot['id']}",
                    disabled=bool(slot.get("locked", False)),
                ):
                    current_used = {c["candidates"][c["current_idx"]]["code"].upper() for c in st.session_state.schedule_slots}
                    current_used.discard(cur["code"].upper())

                    replaced = False
                    for j in range(slot["current_idx"] + 1, len(slot["candidates"])):  # noqa: E741
                        cand = slot["candidates"][j]
                        code_u = cand["code"].upper()
                        if code_u in current_used or code_u in taken_codes_all:
                            continue
                        old_cr = _credits_from_str(cur.get("credits"))
                        new_cr = _credits_from_str(cand.get("credits"))
                        current_total = sum(_credits_from_str(s["candidates"][s["current_idx"]].get("credits")) for s in st.session_state.schedule_slots)
                        if current_total - old_cr + new_cr <= st.session_state.schedule_target_credits:
                            _push_swap(i, slot["current_idx"])
                            slot["current_idx"] = j
                            replaced = True
                            break
                    if not replaced:
                        st.info(f"No more eligible options left for {cur['code']} — {cur['title']} in {slot['origin']}.")
                    st.rerun()

            with cols[2]:
                if st.button("Lock" if not slot.get("locked") else "Unlock", key=f"lock_{slot['id']}"):
                    _toggle_lock(slot)
                    st.rerun()
            with cols[3]:
                pass

            new_total += _credits_from_str(cur.get("credits"))

        st.session_state.schedule_planned_credits = new_total
        st.success(f"Planned credits: {new_total} / Target {st.session_state.schedule_target_credits}")

        csv_bytes = _export_schedule_csv(st.session_state.schedule_slots)
        st.download_button("Export schedule as CSV", data=csv_bytes, file_name="schedule.csv", mime="text/csv")

        json_bytes = export_schedule_json(st.session_state.schedule_slots)
        st.download_button("Save schedule (JSON)", data=json_bytes, file_name="schedule.json", mime="application/json")
        up = st.file_uploader("Load a saved schedule (JSON)", type=["json"], key="sched_loader")
        if up is not None:
            try:
                st.session_state.schedule_slots = import_schedule_json(up.read())
                st.session_state.schedule_planned_credits = sum(_credits_from_str(s["candidates"][s["current_idx"]].get("credits")) for s in st.session_state.schedule_slots)
                st.rerun()
            except Exception as e:
                st.warning(f"Could not load schedule: {e}")


# ---------------------- CHAT HISTORY RENDER ----------------------
def render_history():
    assistant_avatar = "RP.png" if os.path.exists("RP.png") else None
    for m in st.session_state.messages:
        if m["role"] == "user":
            with st.chat_message("user", avatar=st.session_state.get("profile_avatar_path")):
                st.markdown(m["content"])
        else:
            with st.chat_message("assistant", avatar=assistant_avatar or ""):
                st.markdown(m["content"])

render_history()

# ---------------------- CHAT ----------------------
def _student_profile_for_prompt() -> str:
    return student_context_from_taken(rows_all, st.session_state.completed_codes_all)

input_disabled = st.session_state.get("is_generating", False)
if st.session_state.get("is_generating", False):
    st.info("Assistant is generating a response...")
q = st.chat_input(
    "Ask anything",
    disabled=input_disabled
)
hit = fastpath_course_code(q, rows_all)  
if hit:
    with st.chat_message("assistant"):
        st.markdown(hit)
    st.session_state.messages.append({"role": "assistant", "content": hit})
    st.stop()  
if q is None:
    pass
elif not q.strip():
    st.warning("Please type a message first.")
else:
    st.session_state.is_generating = True
    start_ts = time.time()

    with st.chat_message("user", avatar=st.session_state.get("profile_avatar_path")):
        st.markdown(q)
    maybe_capture_name(q)
    st.session_state.messages.append({"role": "user", "content": q})
    _cap_history()
    scopes = infer_scopes(q)
    direct_rows = find_rows_by_code(rows_all, q)
    title_rows = find_rows_by_title(rows, q) if not direct_rows else []
    intent = parse_catalog_intent(q)

    kb = ""
    ans = None
    history_text = build_history_text()
    student_context = _student_profile_for_prompt()
    answer_lang_str = LANG_OPTIONS.get(answer_lang, "English")

    assistant_avatar = "RP.png" if os.path.exists("RP.png") else None
    with st.chat_message("assistant", avatar=assistant_avatar or None):
        ans_placeholder = st.empty()

        if is_university_query(q):
            univ_kb_text = univ_kb_blocks_for(q) or "University facts: (none)\nFaculty: (none)"
            ans = ask_llm_stream(
                univ_chain,
                kb="",
                history_text=history_text,
                q=q,
                answer_lang=answer_lang_str,
                student_context=student_context,
                placeholder=ans_placeholder,
                univ_kb=univ_kb_text,
                groq_fallback_chain=groq_fallback_univ,
            )

        elif direct_rows:
            if college_filter != "All":
                filtered = [r for r in direct_rows if (r.get("college","").upper() == college_filter)]
                if filtered:
                    direct_rows = filtered
            kb = rows_to_kb(direct_rows)
            if st.session_state.completed_codes_all:
                kb += "\n---\n" + rows_to_kb([r for r in rows_all if r["code"].upper() in st.session_state.completed_codes_all])
            ans = ask_llm_stream(
                course_chain, kb, history_text, q, answer_lang_str, student_context,
                ans_placeholder, groq_fallback_chain=groq_fallback_course
            )

        elif title_rows:
            kb = rows_to_kb(title_rows)
            if st.session_state.completed_codes_all:
                kb += "\n---\n" + rows_to_kb([r for r in rows_all if r["code"].upper() in st.session_state.completed_codes_all])
            ans = ask_llm_stream(
                course_chain, kb, history_text, q, answer_lang_str, student_context,
                ans_placeholder, groq_fallback_chain=groq_fallback_course
            )

        elif intent:
            if intent["type"] == "count":
                scoped_rows = rows
                depts = intent["scopes"].get("dept", ["all"])
                if "all" not in depts:
                    prefixes = tuple(p for s in depts for p in DEPT_PREFIXES.get(s, ()))
                    scoped_rows = [r for r in scoped_rows if r["code"].upper().startswith(prefixes)]
                ans_text = f"I currently know {len(scoped_rows)} courses from courses.csv."
                ans_placeholder.markdown(ans_text); ans = ans_text
            elif intent["type"] == "list":
                limit = intent.get("limit", 150)
                scoped_rows = rows
                depts = intent["scopes"].get("dept", ["all"])
                if "all" not in depts:
                    prefixes = tuple(p for s in depts for p in DEPT_PREFIXES.get(s, ()))
                    scoped_rows = [r for r in scoped_rows if r["code"].upper().startswith(prefixes)]
                lines = [f"{(r.get('college') or 'UNK')} • {r['code']} — {r['title']}" for r in scoped_rows]
                if len(lines) > limit:
                    more = len(lines) - limit
                    lines = lines[:limit] + [f"...and {more} more."]
                ans_text = "\n".join(lines)
                ans_placeholder.code(ans_text, language="markdown"); ans = ans_text

        else:
            qx = expand_synonyms(q)
            if is_coursey(qx):
                try:
                    docs = hybrid_retrieve(qx, retriever, vs, int(TOP_K), bm25=bm25)
                except Exception as e:
                    log.error(f"hybrid_retrieve error: {e}"); docs = []
                    docs = reorder_docs_by_scopes(docs, scopes, college_filter)


                    bm25_docs = []  


                    tokens = (q or "").strip().split()
                    use_bm25 = len(tokens) >= 4 and any(len(t) >= 4 for t in tokens) 
                    if use_bm25 and bm25:
                        try:
                            bm25_docs = bm25.get_top_n(tokens, n=int(TOP_K))
                        except Exception:
                            bm25_docs = []  


                    kb = build_kb_from_docs(docs, bm25_docs, top_k=TOP_K, cap=CHUNK_CHAR_CAP)
                if st.session_state.completed_codes_all:
                    kb += ("\n---\n" if kb else "") + rows_to_kb([r for r in rows_all if r["code"].upper() in st.session_state.completed_codes_all])
                if kb and kb != "(no relevant context found)":
                    ans = ask_llm_stream(course_chain, kb, history_text, q, answer_lang_str, student_context, ans_placeholder)
                else:
                    ans_placeholder.markdown("I don't know from the provided data."); ans = "I don't know from the provided data."
            else:
                ans = ask_llm_stream(
                    chat_chain, "", history_text, q, answer_lang_str, student_context,
                    ans_placeholder, groq_fallback_chain=groq_fallback_chat
                )
                if st.session_state.get("user_name") and ans and not ans.lower().startswith(st.session_state["user_name"].lower()):
                    prefixed = friendly_prefix() + ans
                    ans_placeholder.markdown(prefixed); ans = prefixed

        if debug:
            elapsed = f"{(time.time() - start_ts):.2f}s"
            with st.expander(f"Debug: retrieved KB • {elapsed}"):
                st.code(kb or "(none)")
            if is_university_query(q):
                with st.expander("Debug: UNIV_KB view"):
                    st.code(univ_kb_blocks_for(q), language="markdown")
            st.caption(f"Answered in {elapsed} • Model: {MODEL_NAME} • k={TOP_K} • T={TEMPERATURE}")

    st.session_state.messages.append({"role": "assistant", "content": ans})
    _cap_history()
    st.session_state.is_generating = False

# Clear chat action (after header button definition)
if 'clear_clicked' in locals() and clear_clicked:
    st.session_state.messages = []
    st.rerun()

# ---------------------- SCHEDULE BUILDER (invoke below chat) ----------------------
if st.session_state.get("show_schedule", False):
    render_schedule_builder(rows_all, vs, bm25)


st.markdown("""
<style>
/* No extra gap below the chat input */
div[data-testid="stChatInput"] { margin-bottom: 0 !important; }

/* No extra bottom padding on the page */
.main .block-container { padding-bottom: 0 !important; }

/* Hide the custom footer if it still exists in the DOM */
.custom-chat-footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

