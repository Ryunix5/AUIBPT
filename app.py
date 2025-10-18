# app.py
# AUIBPT — Course, Liberal-Arts & Schedule Builder (Ryunix build)
# - College-aware CSV (college,code,title,credits,description,prereqs)
# - Hybrid retrieval (FAISS + optional BM25 via lazy import)
# - Splash & footer (Ryunix Productions)
# - 🗓️ Schedule Builder: LA rules + major prereqs + per-course swap + auto top-up + undo + lock
# - Picker persists selections across filter changes (Major / LA / Both)
# - Export schedule (CSV) + Save/Load (JSON)
# - Student profile context — user “Completed Courses” feed the chat answers
# - Institutional KB (AUIB + CAS/COP/COD + faculty) with dedicated prompt & routing
# - Profile picture avatar + GUI color customizer
# - OpenAI-first LLM (Streamlit Cloud ready) with Ollama fallback for local

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

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler  # fixed import (no langchain.callbacks)

from settings import MODEL_NAME, CSV_PATH, INDEX_DIR, TOP_K, TEMPERATURE, NUM_PREDICT, USE_OPENAI
from data_loader import load_catalog_rows, rows_to_documents
from indexer import ensure_index, load_index, rebuild_index

# --- OpenAI key resolution (safe locally + Streamlit Cloud) ---
def _get_openai_key() -> str | None:
    # 1) environment variable
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    # 2) Streamlit Cloud / local secrets.toml (optional)
    try:
        return st.secrets["OPENAI_API_KEY"]  # will raise if secrets missing
    except Exception:
        return None

OPENAI_KEY = _get_openai_key()
if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY  # for SDKs that read from env


# ---------------------- UI CONFIG ----------------------
page_icon = "RP.png" if os.path.exists("RP.png") else "🎓"
st.set_page_config(page_title="AUIBPT • Course Chatbot", page_icon=page_icon, layout="wide")

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
KNOWN_COLLEGES = {"All", "CAS", "COP", "COD"}
LANG_OPTIONS = {"English": "English", "Arabic": "Arabic"}

# Degree totals by program
DEGREE_TOTAL = {"CS": 126, "Pharmacy": 180, "Dentistry": 189}
MAJOR_MAP = {
    "CS": {"college": "CAS", "prefixes": ("CSC", "MAT", "STA")},
    "Pharmacy": {"college": "COP", "prefixes": ("PHA", "CHE", "BIO")},
    "Dentistry": {"college": "COD", "prefixes": ("BDS", "BIO", "CHE")},
}

# ---------------------- INSTITUTIONAL KB SEED & LOADER ----------------------
UNIV_KB_SEED = {
    "university": {
        "facts": [
            "American University of Iraq–Baghdad (AUIB), private non-profit; established 2018; first classes Feb 2021.",
            "Campus located at the Al-Faw Palace complex near Baghdad International Airport; historic site modernized for learning.",
            "English-language, American-style curriculum with liberal arts core for undergraduates."
        ],
        "leadership": [
            "President: Brad (John Bradley) Cook.",
            "VP Academic Affairs: Zouhair K. Atieh.",
            "Deans: CAS – Monica Hanna; COP – Achraf Al Faraj; COD – Nada Naaman."
        ],
        "accreditations": [
            "Recognized by Iraq’s Ministry of Higher Education and Scientific Research.",
            "College of Pharmacy (B.Pharm): ACPE International Pre-accreditation.",
            "College of Dentistry: Member of Association for Dental Education in Europe (ADEE)."
        ],
        "programs": [
            "CAS: Biology, Chemistry, Physics, Computer Science, English, History/Archaeology, Psychology, more.",
            "COP: B.Pharm 5-year program (clinical & industrial tracks).",
            "COD: BDS 5-year, ~189 credits (preclinical to clinical training)."
        ],
        "partnerships": [
            "Partners/links include Vanderbilt (Education), Lawrence Tech/Temple (Engineering), Exeter (UK), Sapienza (Italy)."
        ]
    },
    "faculty": [
        {"college":"CAS","name":"Monica Hanna","title":"Dean, College of Arts & Sciences; Egyptologist","areas":"Archaeology, cultural heritage","prior":"American Univ. in Cairo; Univ. of Pisa; Aswan Univ. (Egypt)","notes":""},
        {"college":"CAS","name":"Doris Jaalouk","title":"Professor of Biology/Public Health","areas":"Public health, health education, nutrition","prior":"Notre Dame Univ.–Louaize (Lebanon)","notes":""},
        {"college":"CAS","name":"Robert David Putnam","title":"Assistant Professor of History","areas":"Modern Iraqi/Middle East history","prior":"Seattle Pacific Univ.; research posts","notes":""},
        {"college":"CAS","name":"Mutasem Sinnokrot","title":"Associate Professor of Chemistry/Physics","areas":"Materials & chemical engineering, physical chemistry","prior":"Khalifa Univ. (UAE); other posts","notes":"~35 publications; >5k citations (approx.)"},
        {"college":"CAS","name":"Dhrgam Al Kafaf","title":"Assistant Professor of Computer Science","areas":"Artificial intelligence, autonomous systems, computer vision","prior":"Research posts in AI/computer vision","notes":"Faculty Senate (Educational Resources Committee)"},
        {"college":"CAS","name":"Ahmed Elshewy","title":"Assistant Professor of Chemistry","areas":"Medicinal & organic chemistry, catalysis","prior":"Cairo Univ.; postdoctoral work","notes":""},
        {"college":"CAS","name":"Christos Kokorelis","title":"Associate Professor of Mathematical Physics","areas":"String theory, mathematical physics","prior":"University of Sussex (UK)","notes":""},
        {"college":"CAS","name":"Ioannis Haranas","title":"Associate Professor of Astrophysics","areas":"Astrophysics, planetary & space physics","prior":"Wilfrid Laurier Univ. (Canada)","notes":""},
        {"college":"CAS","name":"John Wall","title":"Associate Professor of English & Linguistics","areas":"Literature, linguistics, philosophy","prior":"Positions in UK & Middle East","notes":""},
        {"college":"CAS","name":"Haidar Sabbagh","title":"Assistant Professor of Physics","areas":"Applied physics, electronics, renewables","prior":"Industry & academia in Iraq","notes":""},
        {"college":"CAS","name":"Mohammad F. Kazan","title":"Associate Professor; Head of Natural & Applied Sciences","areas":"Molecular immunology, cytokines","prior":"Faculty roles in Lebanon","notes":""},
        {"college":"COP","name":"Achraf Al Faraj","title":"Dean, College of Pharmacy; Professor","areas":"Nanomedicine, drug delivery","prior":"Univ. Lyon 1 (France); King Saud Univ. (Saudi Arabia); Lebanon","notes":"Founding Dean; 50+ publications (approx.)"},
        {"college":"COP","name":"Baher S. Daihom","title":"Assistant Professor of Pharmaceutics","areas":"Pharmaceutics, 3D-printed drug delivery","prior":"Pharmaceutics PhD; international conference presentations","notes":"Research on 3D-printed implants"},
        {"college":"COP","name":"Rana Alaaeddine","title":"Assistant Professor of Pharmacology","areas":"Cardiovascular pharmacology, therapeutics","prior":"American Univ. of Beirut (AUB)","notes":""},
        {"college":"COP","name":"Adib Charafeddine","title":"Assistant Professor of Medicinal Chemistry","areas":"Medicinal/organic chemistry, drug design","prior":"Academic roles in Lebanon","notes":""},
        {"college":"COD","name":"Nada Naaman","title":"Dean, College of Dentistry; Periodontist","areas":"Periodontology, implant dentistry","prior":"Saint Joseph Univ. (Beirut); Paris Diderot Univ. (France)","notes":"Former dean; Secretary General of Arab Dental Faculties"}
    ]
}

def load_university_kb() -> dict:
    """Load institutional KB from JSON if present, else fall back to seed."""
    path = "auib_university_kb.json"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict): raise ValueError("Root must be object")
                if "university" not in data or "faculty" not in data: raise ValueError("Missing keys")
                return data
        except Exception as e:
            log.warning(f"Failed reading auib_university_kb.json, using seed: {e}")
    return UNIV_KB_SEED

UNIV_KB = load_university_kb()

_UNIV_HOOK_WORDS = {
    "auib","american university of iraq","baghdad","cas","cop","cod","college of arts","college of pharmacy",
    "college of dentistry","dean","professor","faculty","mission","vision","accreditation","acpe","adee",
    "campus","al-faw","al faw","partnership","vanderbilt","temple","exeter","sapienza"
}

def is_university_query(q: str) -> bool:
    if not q: return False
    ql = q.lower()
    has_name = any(f["name"].lower() in ql for f in UNIV_KB.get("faculty", []))
    hook = any(w in ql for w in _UNIV_HOOK_WORDS)
    return has_name or hook

def univ_kb_blocks_for(q: str, limit: int = 24) -> str:
    """Return a compact textual KB block drawn from UNIV_KB for the LLM."""
    ql = (q or "").lower()
    fac = UNIV_KB.get("faculty", [])
    scored = []
    for f in fac:
        blob = " ".join([f.get("name",""), f.get("title",""), f.get("areas",""), f.get("prior",""), f.get("college","")]).lower()
        score = 0
        A = set(re.sub(r"[^\w\s]", " ", ql).split())
        B = set(re.sub(r"[^\w\s]", " ", blob).split())
        score = len(A & B)
        if f.get("college","").lower() in ql: score += 1
        scored.append((score, f))
    scored.sort(key=lambda x: (-x[0], x[1].get("college",""), x[1].get("name","")))
    fac_pick = [f for s,f in scored[:limit] if s > 0] or fac[: min(limit, 16)]

    lines = []
    uni = UNIV_KB.get("university", {})
    if uni:
        facts = uni.get("facts", [])
        accs = uni.get("accreditations", [])
        leader = uni.get("leadership", [])
        progs = uni.get("programs", [])
        parts = uni.get("partnerships", [])
        if facts:      lines.append("University facts: " + " | ".join(facts))
        if accs:       lines.append("Accreditations: " + " | ".join(accs))
        if leader:     lines.append("Leadership: " + " | ".join(leader))
        if progs:      lines.append("Programs: " + " | ".join(progs))
        if parts:      lines.append("Partnerships: " + " | ".join(parts))
    if fac_pick:
        lines.append("Faculty:")
        for f in fac_pick:
            lines.append(
                f"- [{f.get('college','?')}] {f.get('name','?')} — {f.get('title','?')}; "
                f"Areas: {f.get('areas','?')}; Prior: {f.get('prior','?')}. {f.get('notes','')}".strip()
            )
    return "\n".join(lines) if lines else ""

# ---------------------- PROMPTS (friendly / interactive) ----------------------
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
You are AUIBPT, a helpful, upbeat campus buddy. Be concise (≤3 sentences), conversational, and encouraging.
Use plain language, optionally offer a short follow-up nudge.

Personalize advice using the student's completed courses from "student_profile" (e.g., what they could take next or what they've unlocked).

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

# ---------------------- HYBRID RETRIEVAL ----------------------
def _try_init_bm25(corpus_texts: List[str]):
    """Optional BM25 via lazy import. Returns None if unavailable."""
    try:
        spec = importlib.util.find_spec("rank_bm25")
        if spec is None: return None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        BM25Okapi = getattr(mod, "BM25Okapi", None)
        if BM25Okapi is None: return None
        tokenized = [t.split() for t in corpus_texts]
        return BM25Okapi(tokenized)
    except Exception:
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

def make_llm(model_name: str, temperature: float, num_predict: int, callbacks=None):
    """OpenAI-first with streaming; Ollama fallback for local dev (no key)."""
    effective_use_openai = bool(OPENAI_KEY) and USE_OPENAI
    if effective_use_openai:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=num_predict,
            streaming=True,
            callbacks=callbacks or []
        )
    else:
        # local fallback so devs without a key can still run the app
        from langchain_ollama import OllamaLLM  # lazy import to avoid Cloud issues
        return OllamaLLM(
            model=model_name,
            temperature=temperature,
            num_predict=num_predict,
            stop=["</final>"],
            callbacks=callbacks or []
        )


def ask_llm_stream(chain, kb: str, history_text: str, q: str, answer_lang: str, student_context: str, placeholder, univ_kb: str = "") -> str:
    handler = StreamHandler(placeholder)
    payload = {
        "kb": kb,
        "univ_kb": univ_kb,
        "history": history_text,
        "question": q,
        "answer_lang": answer_lang,
        "student_context": student_context
    }
    try:
        raw = chain.invoke(payload, config={"callbacks": [handler]})
        final_text = _clean_output(_to_str(raw).strip())
        placeholder.markdown(final_text)
        return final_text
    except Exception as e:
        log.error(f"Streaming failed, fallback to non-streamed invoke: {e}")
        raw = chain.invoke(payload)
        final_text = _clean_output(_to_str(raw).strip())
        placeholder.markdown(final_text)
        return final_text

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

# ---------------------- THEME / APPEARANCE ----------------------
def apply_theme(primary: str, bg: str, text: str):
    css = f"""
    <style>
    .stApp {{
        background: {bg} !important;
        color: {text} !important;
    }}
    .stButton>button, .stDownloadButton>button {{
        background: {primary} !important;
        color: white !important;
        border: 0 !important;
        border-radius: 8px !important;
    }}
    .stChatMessage .stMarkdown, .stMarkdown p {{
        color: {text} !important;
    }}
    a, .stMarkdown a {{ color: {primary} !important; }}
    .stSidebar {{ color: {text} !important; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    if os.path.exists("RP.png"):
        st.image("RP.png", width=72)
    st.markdown("## ⚙️ Settings")
    st.caption("System-managed settings shown for reference.")
    answer_lang = st.selectbox("Answer language", list({"English","Arabic"}), index=0)
    college_filter = st.selectbox("College filter", sorted(KNOWN_COLLEGES), index=0)
    st.text_input("Model", value=MODEL_NAME, disabled=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.number_input("Retriever k", 1, 10, value=int(TOP_K), step=1, disabled=True)
        debug = st.toggle("Debug mode (show KB & stats)", value=False)
    with col_b:
        st.slider("Temperature", 0.0, 1.5, value=float(TEMPERATURE), step=0.1, disabled=True)
        st.number_input("Max tokens", 64, 4096, value=int(NUM_PREDICT), step=32, disabled=True)

    # Appearance & avatar
    st.divider()
    st.markdown("## 🎨 Appearance")

    if "profile_avatar_path" not in st.session_state:
        st.session_state.profile_avatar_path = None
    if "theme_primary" not in st.session_state:
        st.session_state.theme_primary = "#4f46e5"
    if "theme_bg" not in st.session_state:
        st.session_state.theme_bg = "#0b1220"
    if "theme_text" not in st.session_state:
        st.session_state.theme_text = "#e2e8f0"

    c1, c2, c3 = st.columns(3)
    with c1:
        primary = st.color_picker("Accent", st.session_state.theme_primary, key="pick_primary")
    with c2:
        bg = st.color_picker("Background", st.session_state.theme_bg, key="pick_bg")
    with c3:
        textc = st.color_picker("Text", st.session_state.theme_text, key="pick_text")

    if primary != st.session_state.theme_primary or bg != st.session_state.theme_bg or textc != st.session_state.theme_text:
        st.session_state.theme_primary = primary
        st.session_state.theme_bg = bg
        st.session_state.theme_text = textc

    # Profile picture
    pp = st.file_uploader("Profile picture", type=["png","jpg","jpeg","gif"], key="profile_pic_up")
    if pp is not None:
        try:
            avatar_path = "user_avatar.png"
            with open(avatar_path, "wb") as f:
                f.write(pp.read())
            st.session_state.profile_avatar_path = avatar_path
            st.image(avatar_path, width=72, caption="Current profile")
        except Exception as e:
            st.warning(f"Could not save avatar: {e}")
    elif st.session_state.profile_avatar_path and os.path.exists(st.session_state.profile_avatar_path):
        st.image(st.session_state.profile_avatar_path, width=72, caption="Current profile")

    apply_theme(st.session_state.theme_primary, st.session_state.theme_bg, st.session_state.theme_text)

    st.divider()
    clear = st.button("🧹 Clear chat", use_container_width=True)

# ---------------------- SESSION ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "_splash_shown" not in st.session_state:
    st.session_state._splash_shown = False
# schedule state
if "schedule_slots" not in st.session_state:
    st.session_state.schedule_slots = []
if "schedule_planned_credits" not in st.session_state:
    st.session_state.schedule_planned_credits = 0
if "schedule_target_credits" not in st.session_state:
    st.session_state.schedule_target_credits = 15
if "schedule_major_key" not in st.session_state:
    st.session_state.schedule_major_key = "CS"
# persistent completed-course selections across filter changes
if "completed_codes_all" not in st.session_state:
    st.session_state.completed_codes_all = set()
# swap history (undo)
if "swap_history" not in st.session_state:
    st.session_state.swap_history = []

if not st.session_state._splash_shown:
    show_splash()
if clear:
    st.session_state.messages = []

# ---------------------- DATA & INDEX ----------------------
@st.cache_data(show_spinner=True, ttl=60)
def _load_rows(csv_path: str) -> List[Dict]:
    return load_catalog_rows(csv_path)

@st.cache_resource(show_spinner=True)
def _build_or_load_index(csv_path: str, index_dir: str, force: bool) -> Tuple[List[Dict], object, object]:
    rows = _load_rows(csv_path)
    docs = rows_to_documents(rows)
    if force:
        rebuild_index(docs, index_dir)
    else:
        ensure_index(docs, index_dir)
    vs = load_index(index_dir)
    try:
        corpus_texts = [d.page_content for d in vs.docstore._dict.values()]
    except Exception:
        corpus_texts = []
    bm25 = _try_init_bm25(corpus_texts) if corpus_texts else None
    return rows, vs, bm25

status_col1, status_col2, status_col3 = st.columns([1.3, 1, 1])
with status_col1:
    st.title("🎓 AUIBPT — AUIB Course Chatbot")
with status_col2:
    st.caption(f"Model: `{MODEL_NAME}` • k={TOP_K} • T={TEMPERATURE} • max={NUM_PREDICT}")
with status_col3:
    exists = os.path.exists(CSV_PATH)
    st.caption(f"CSV: {'✅ found' if exists else '❌ missing'}")

st.caption("Version 1.5 — AUIBPT (Ryunix Build)")
st.divider()

force_rebuild = st.checkbox("Rebuild FAISS index from CSV (one-time)", value=False)
try:
    rows_all, vs, bm25 = _build_or_load_index(CSV_PATH, INDEX_DIR, force_rebuild)
    rows = filter_rows_by_college(rows_all, college_filter)
    retriever = vs.as_retriever(search_kwargs={"k": int(TOP_K)})
except Exception as e:
    st.error(f"Failed to prepare index or load catalog: {e}")
    st.stop()

llm = make_llm(MODEL_NAME, TEMPERATURE, NUM_PREDICT)
course_chain = ChatPromptTemplate.from_template(COURSE_PROMPT) | llm
chat_chain   = ChatPromptTemplate.from_template(CHAT_PROMPT)   | llm
univ_chain   = ChatPromptTemplate.from_template(UNIV_PROMPT)   | llm

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
    # General
    "UNI101": "General",
    # Communication
    "ENL101": "Communication", "ENL201": "Communication", "ENL210": "Communication",
    # Quantitative
    "CSC101": "Quantitative", "MAT101": "Quantitative",
    # Humanities
    "HIS101": "Humanities", "HIS102": "Humanities", "HIS105": "Humanities",
    "HUM101": "Humanities", "LIT101": "Humanities", "PHA210": "Humanities",
    "PHI101": "Humanities", "POL125": "Humanities",
    "TLD100": "Humanities", "TLD101": "Humanities", "TLD102": "Humanities", "TLD103": "Humanities",
    # Social Sciences
    "COM101": "SocialScience", "ECO101": "SocialScience", "FIN101": "SocialScience",
    "HCT108": "SocialScience", "MIS101": "SocialScience", "POL101": "SocialScience",
    "POL112": "SocialScience", "POL191": "SocialScience", "PSY101": "SocialScience",
    "SOC101": "SocialScience",
    # Natural Sciences
    "CHE100": "NaturalScience", "ENV201": "NaturalScience", "GEO101": "NaturalScience",
    "PHY100": "NaturalScience", "PHY105": "NaturalScience",
}
LA_QUANT_BOTH = {"CSC101", "MAT101"}

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
    """Build a compact profile string used inside LLM prompts."""
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

# ---------------------- 🗓️ SCHEDULE BUILDER (+ swap) ----------------------
def build_semester_schedule(
    major_key: str,
    target_credits: int,
    taken_codes: Set[str],
    rows_all: List[Dict]
) -> Tuple[List[Dict], Dict[str,int], Dict[str,int], int]:
    """Build an initial schedule (list of rows) up to target_credits using LA first, then major."""
    la_rows = rows_all
    la_counts = la_completed_counts(taken_codes)
    la_remain = la_remaining(la_counts, taken_codes)
    la_pool = la_recommend_pool(taken_codes, la_rows, la_remain)
    major_info = MAJOR_MAP[major_key]
    major_pool = _eligible_major_rows(taken_codes, rows_all, major_info["prefixes"])

    schedule: List[Dict] = []
    used_codes: Set[str] = set(c.upper() for c in taken_codes)
    cur_credits = 0

    def _credits_from_course(r: Dict) -> int:
        return _credits_from_str(r.get("credits"))

    def try_add_course(r: Dict) -> bool:
        nonlocal cur_credits
        code = r["code"].upper()
        if code in used_codes: return False
        cr = _credits_from_course(r)
        if cur_credits + cr > target_credits: return False
        schedule.append(r); used_codes.add(code); cur_credits += cr
        return True

    # Force-include Quantitative (CSC101/MAT101) if missing
    for q_code in ["CSC101","MAT101"]:
        if la_remain.get("Quantitative",0) > 0 and q_code not in used_codes:
            r = next((x for x in rows_all if x["code"].upper()==q_code), None)
            if r:
                reqs = _parse_prereq_codes(r.get("prereqs",""))
                if all(rc in used_codes for rc in reqs):
                    try_add_course(r)

    # Fill other LA categories by remaining need
    cat_order = sorted(LA_REQUIREMENTS.keys(), key=lambda c: -la_remain.get(c,0))
    for cat in cat_order:
        need = la_remain.get(cat, 0)
        if need <= 0: continue
        for r in la_pool.get(cat, []):
            if need <= 0: break
            if try_add_course(r):
                need -= 1

    # Fill with major courses
    for r in major_pool:
        if cur_credits >= target_credits: break
        try_add_course(r)

    return schedule, la_counts, la_remain, cur_credits

def _rebuild_pools(major_key: str, taken_codes: Set[str], rows_all: List[Dict]) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    la_counts = la_completed_counts(taken_codes)
    la_remain = la_remaining(la_counts, taken_codes)
    la_pool = la_recommend_pool(taken_codes, rows_all, la_remain)
    major_pool = _eligible_major_rows(taken_codes, rows_all, MAJOR_MAP[major_key]["prefixes"])
    return la_pool, major_pool

def _export_schedule_csv(slots: List[Dict]) -> bytes:
    """Create CSV bytes for current schedule slots."""
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
            new_slots.append({"id": f"import-{len(new_slots)}", "origin": s["origin"], "candidates": cand_rows, "current_idx": int(s["current_idx"])})
    return new_slots

def _auto_top_up(major_key: str, target_credits: int, taken_codes: Set[str], slots: List[Dict], rows_all: List[Dict]) -> None:
    """Try to add eligible courses to reach target credits (without violating prereqs) by extending slots."""
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
                slots.append({"id": f"extra-{origin}-{len(slots)}", "origin": origin, "candidates": [r], "current_idx": 0})
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

# ---------------------- UI: SCHEDULE BUILDER ----------------------
with st.expander("🗓️ Schedule Builder — generate a full next-semester plan", expanded=True):
    major_key = st.selectbox("Your major / program", ["CS", "Pharmacy", "Dentistry"], index=["CS","Pharmacy","Dentistry"].index(st.session_state.schedule_major_key))
    st.session_state.schedule_major_key = major_key
    target_credits = st.slider("Target credits for next semester", 9, 21, st.session_state.schedule_target_credits, 1)
    st.session_state.schedule_target_credits = target_credits

    # Picker scope: Major only / Liberal Arts only / Both
    picker_scope = st.radio("Show in completed-courses picker:", ["Major only", "Liberal Arts only", "Both"], horizontal=True)

    major_prefixes = MAJOR_MAP[major_key]["prefixes"]
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

    preselected_labels = [lbl for lbl, code in label_to_code.items() if code in st.session_state.completed_codes_all]
    picked_labels = st.multiselect("I have completed:", labels, default=preselected_labels, key="completed_picker")
    picked_visible_codes = {label_to_code[lbl] for lbl in picked_labels}

    # Persist: keep hidden selections + merge visible picks
    hidden_kept = st.session_state.completed_codes_all - visible_codes
    st.session_state.completed_codes_all = hidden_kept | picked_visible_codes
    taken_codes_all = set(st.session_state.completed_codes_all)

    # Degree progress header
    completed_credits = _credits_completed(taken_codes_all, rows_all)
    degree_total = DEGREE_TOTAL[major_key]
    st.caption(f"Progress: **{completed_credits}** / **{degree_total}** credits • Target this term: **{target_credits}**")

    col_build_a, col_build_b, col_build_c, col_build_d = st.columns([0.4,0.2,0.2,0.2])
    with col_build_a:
        build_btn = st.button("🛠️ Build my schedule", use_container_width=True)
    with col_build_b:
        reset_btn = st.button("♻️ Reset schedule", use_container_width=True)
    with col_build_c:
        topup_btn = st.button("⚡ Auto top-up", use_container_width=True, disabled=not st.session_state.schedule_slots)
    with col_build_d:
        undo_btn = st.button("↩️ Undo last swap", use_container_width=True, disabled=not st.session_state.swap_history)

    if reset_btn:
        st.session_state.schedule_slots = []
        st.session_state.schedule_planned_credits = 0
        st.rerun()

    if build_btn:
        schedule, la_counts, la_remain, planned_credits = build_semester_schedule(
            major_key=major_key,
            target_credits=target_credits,
            taken_codes=taken_codes_all,
            rows_all=rows_all
        )
        la_pool, major_pool = _rebuild_pools(major_key, taken_codes_all, rows_all)

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

            slots.append({"id": f"{origin}-{idx}", "origin": origin, "candidates": candidates, "current_idx": 0})
            used.add(c["code"].upper())

        st.session_state.schedule_slots = slots
        st.session_state.schedule_planned_credits = sum(_credits_from_str(s["candidates"][0].get("credits")) for s in slots)

    if topup_btn and st.session_state.schedule_slots:
        _auto_top_up(major_key, target_credits, taken_codes_all, st.session_state.schedule_slots, rows_all)
        st.session_state.schedule_planned_credits = sum(_credits_from_str(s["candidates"][s["current_idx"]].get("credits")) for s in st.session_state.schedule_slots)
        st.rerun()

    if undo_btn:
        if _undo_swap():
            st.rerun()
        else:
            st.info("Nothing to undo.")

    if st.session_state.schedule_slots:
        st.markdown("### ✅ Suggested schedule")
        remove_msgs = []
        new_total = 0

        for i, slot in enumerate(st.session_state.schedule_slots):
            cur = slot["candidates"][slot["current_idx"]]
            pr = cur.get("prereqs") or "None/Unknown"
            cr = cur.get("credits") or "Unknown"

            cols = st.columns([0.64, 0.12, 0.12, 0.12])
            with cols[0]:
                lock_badge = "🔒" if slot.get("locked") else "🔓"
                st.markdown(f"**{cur['code']} — {cur['title']}**  \nCategory: {slot['origin']} • Credits: {cr} • Prereqs: {pr}  \n{lock_badge} {'Locked' if slot.get('locked') else 'Unlocked'}")
                why_bits = []
                if cur["code"].upper() in LA_CATEGORY:
                    why_bits.append(f"meets **{LA_CATEGORY[cur['code'].upper()]}**")
                else:
                    why_bits.append("major requirement/elective")
                reqs = _parse_prereq_codes(cur.get("prereqs",""))
                if not reqs:
                    why_bits.append("no explicit prereqs")
                else:
                    if all(rc in taken_codes_all for rc in reqs):
                        why_bits.append("you satisfy prereqs")
                    else:
                        why_bits.append("prereqs satisfied during planning")
                st.caption("Why this? " + " • ".join(why_bits))

            with cols[1]:
                if st.button("❌ swap", help="Replace with the next eligible option", key=f"swap_{slot['id']}", disabled=slot.get("locked")):
                    current_used = {c["candidates"][c["current_idx"]]["code"].upper() for c in st.session_state.schedule_slots}
                    current_used.discard(cur["code"].upper())

                    replaced = False
                    for j in range(slot["current_idx"] + 1, len(slot["candidates"])):
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
                        remove_msgs.append(f"No more eligible options left for **{cur['code']} — {cur['title']}** in {slot['origin']}.")
                    st.rerun()

            with cols[2]:
                if st.button("🔒 lock" if not slot.get("locked") else "🔓 unlock", key=f"lock_{slot['id']}"):
                    _toggle_lock(slot)
                    st.rerun()
            with cols[3]:
                pass

            new_total += _credits_from_str(cur.get("credits"))

        st.session_state.schedule_planned_credits = new_total
        st.success(f"Planned credits: **{new_total}** / Target **{st.session_state.schedule_target_credits}**")

        csv_bytes = _export_schedule_csv(st.session_state.schedule_slots)
        st.download_button("📥 Export schedule as CSV", data=csv_bytes, file_name="schedule.csv", mime="text/csv")

        json_bytes = export_schedule_json(st.session_state.schedule_slots)
        st.download_button("💾 Save schedule (JSON)", data=json_bytes, file_name="schedule.json", mime="application/json")
        up = st.file_uploader("Load a saved schedule (JSON)", type=["json"], key="sched_loader")
        if up is not None:
            try:
                st.session_state.schedule_slots = import_schedule_json(up.read())
                st.session_state.schedule_planned_credits = sum(_credits_from_str(s["candidates"][s["current_idx"]].get("credits")) for s in st.session_state.schedule_slots)
                st.rerun()
            except Exception as e:
                st.toast(f"Could not load schedule: {e}")

# ---------------------- CHAT HISTORY RENDER ----------------------
def render_history():
    assistant_avatar = "RP.png" if os.path.exists("RP.png") else None
    for m in st.session_state.messages:
        if m["role"] == "user":
            with st.chat_message("user", avatar=st.session_state.profile_avatar_path or "👤"):
                st.markdown(m["content"])
        else:
            with st.chat_message("assistant", avatar=assistant_avatar or "🤖"):
                st.markdown(m["content"])

render_history()

# ---------------------- CHAT ----------------------
def _student_profile_for_prompt() -> str:
    return student_context_from_taken(rows_all, st.session_state.completed_codes_all)

q = st.chat_input("Ask about AUIB, a professor, a course (e.g., 'what is CSC101?'), or just chat")
if q is None:
    pass
elif not q.strip():
    st.warning("Please type a message first.")
else:
    start_ts = time.time()
    with st.chat_message("user", avatar=st.session_state.profile_avatar_path or "👤"):
        st.markdown(q)
    maybe_capture_name(q)
    st.session_state.messages.append({"role": "user", "content": q})

    scopes = infer_scopes(q)
    direct_rows = find_rows_by_code(rows_all, q)
    title_rows = find_rows_by_title(rows, q) if not direct_rows else []
    intent = parse_catalog_intent(q)

    kb = ""; ans = None
    history_text = build_history_text()
    student_context = _student_profile_for_prompt()
    answer_lang_str = LANG_OPTIONS.get(answer_lang, "English")

    assistant_avatar = "RP.png" if os.path.exists("RP.png") else None
    with st.chat_message("assistant", avatar=assistant_avatar or "🤖"):
        ans_placeholder = st.empty()

        # ===== DECISION: which brain to use? =====
        if is_university_query(q):
            univ_kb_text = univ_kb_blocks_for(q) or "University facts: (none)\nFaculty: (none)"
            ans = ask_llm_stream(
                univ_chain,
                kb="",  # not used here
                history_text=history_text,
                q=q,
                answer_lang=answer_lang_str,
                student_context=student_context,
                placeholder=ans_placeholder,
                univ_kb=univ_kb_text,
            )

        elif direct_rows:
            if college_filter != "All":
                filtered = [r for r in direct_rows if (r.get("college","").upper() == college_filter)]
                if filtered:
                    direct_rows = filtered
            kb = rows_to_kb(direct_rows)
            if st.session_state.completed_codes_all:
                kb += "\n---\n" + rows_to_kb([r for r in rows_all if r["code"].upper() in st.session_state.completed_codes_all])
            ans = ask_llm_stream(course_chain, kb, history_text, q, answer_lang_str, student_context, ans_placeholder)

        elif title_rows:
            kb = rows_to_kb(title_rows)
            if st.session_state.completed_codes_all:
                kb += "\n---\n" + rows_to_kb([r for r in rows_all if r["code"].upper() in st.session_state.completed_codes_all])
            ans = ask_llm_stream(course_chain, kb, history_text, q, answer_lang_str, student_context, ans_placeholder)

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
                kb = prepare_kb_from_docs(docs)
                if st.session_state.completed_codes_all:
                    kb += ("\n---\n" if kb else "") + rows_to_kb([r for r in rows_all if r["code"].upper() in st.session_state.completed_codes_all])
                if kb and kb != "(no relevant context found)":
                    ans = ask_llm_stream(course_chain, kb, history_text, q, answer_lang_str, student_context, ans_placeholder)
                else:
                    ans_placeholder.markdown("I don't know from the provided data."); ans = "I don't know from the provided data."
            else:
                ans = ask_llm_stream(chat_chain, "", history_text, q, answer_lang_str, student_context, ans_placeholder)
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

# ---------------------- FOOTER ----------------------
st.markdown(
    """
    <div style="
        text-align:center;
        font-size:13px;
        color:rgba(226,232,240,0.85);
        margin-top:20px;
        padding-top:6px;
        border-top:1px solid rgba(255,255,255,0.1);
    ">Ryunix Productions © 2025</div>
    """,
    unsafe_allow_html=True
)
