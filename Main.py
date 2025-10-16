# main.py
# Run once (if you switch models):  ollama pull llama3.1:8b
from typing import List, Dict
import argparse, re, string, math
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from settings import MODEL_NAME, CSV_PATH, INDEX_DIR, TOP_K, TEMPERATURE, NUM_PREDICT
from data_loader import load_catalog_rows, rows_to_documents
from indexer import ensure_index, load_index, rebuild_index

# -------------------- Cleaners & helpers --------------------

COURSE_HINTS = [
    "course", "class", "prereq", "prerequisite", "credit", "credits",
    "catalog", "syllabus", "covers", "topic", "learn", "teaches",
    "semester", "enroll", "registration", "requirement", "requirements",
    "what is", "describe", "explain", "about"
]

COURSE_CODE_RE = re.compile(r"\b([A-Za-z]{2,4})\s*-?\s*(\d{3})\b")

def is_coursey(q: str) -> bool:
    ql = q.lower()
    if COURSE_CODE_RE.search(q):
        return True
    return any(h in ql for h in COURSE_HINTS + ["csc", "mat", "mth", "sta"])

def _to_str(x):
    try:
        from langchain_core.messages import AIMessage
        if isinstance(x, AIMessage):
            return x.content or ""
    except Exception:
        pass
    return x if isinstance(x, str) else str(x)

def _clean_output(text: str) -> str:
    # prefer <final>...</final> if present
    m = re.search(r"<final>(.*?)</final>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
    # remove any leaked think/echo
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r'(?i)\brules:\b.*', '', text)
    text = re.sub(r'(?i)\b(knowledge base|kb|instructions)\b.*', '', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "I don't know from the provided data."

def ask_llm(chain, kb, history, q):
    raw = chain.invoke({"kb": kb, "history": history, "question": q})
    raw_s = _to_str(raw).strip()
    if not raw_s:
        return "I don't know from the provided data."
    return _clean_output(raw_s)

# -------------------- Catalog helpers --------------------

def find_rows_by_code(rows: List[Dict], q: str) -> List[Dict]:
    """Return rows whose code matches any course code in q (CSC101 / CSC 101 / csc-101)."""
    if not rows:
        return []
    idx = { (r.get("code","") or "").replace(" ", "").upper(): r for r in rows if "code" in r }
    hits, seen = [], set()
    for dept, num in COURSE_CODE_RE.findall(q):
        key = f"{dept.upper()}{num}"
        row = idx.get(key)
        if row and row["code"] not in seen:
            hits.append(row); seen.add(row["code"])
    return hits

def _norm_text(s: str) -> str:
    return s.lower().translate(str.maketrans("", "", string.punctuation)).strip()

def find_rows_by_title(rows: List[Dict], q: str) -> List[Dict]:
    """Loose title match for queries like 'what is data structures?'."""
    qn = _norm_text(q)
    q_tokens = set(qn.split())
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

def rows_to_kb(rows_subset: List[Dict]) -> str:
    return "\n---\n".join(
        f"Course Code: {r['code']}\n"
        f"Title: {r['title']}\n"
        f"Description: {r['description']}\n"
        f"[source: courses.csv | code: {r['code']}]"
        for r in rows_subset
    )

# -------------------- Simple intents (count/list) --------------------

def _detect_scopes(q: str) -> List[str]:
    ql = q.lower()
    scopes = []
    if any(w in ql for w in ["computer science", "cs ", " cs", "c.s.", "csc", "comp sci"]):
        scopes.append("cs")
    if any(w in ql for w in ["math", "mathematics", "mat ", " mat", "mth", "sta", "statistics"]):
        scopes.append("math")
    if not scopes:
        scopes = ["all"]
    return scopes

def parse_catalog_intent(q: str):
    ql = q.lower()
    if "how many" in ql and "course" in ql:
        return {"type": "count", "scopes": _detect_scopes(ql)}
    if any(kw in ql for kw in ["list all courses", "show all courses", "list courses", "all courses"]):
        return {"type": "list", "scopes": _detect_scopes(ql), "limit": 100}
    return None

# -------------------- Freshman/Starter recommendation helpers --------------------

RECO_HINTS = [
    "first year", "freshman", "first-year", "year 1", "intro path",
    "starter", "beginner", "getting started", "new to", "foundation",
    "recommend", "suggest", "what .* classes should i take", "what .* courses should i take"
]

def parse_recommendation_intent(q: str):
    ql = q.lower().strip()
    m_num = re.search(r"\bwhat\s+(\d+)\s+(classes|courses)\s+should\s+i\s+take\b", ql)
    if m_num:
        num = int(m_num.group(1))
        return {"type": "recommend", "num": max(1, min(8, num))}
    if any(h in ql for h in RECO_HINTS):
        m = re.search(r"\b(\d+)\b", ql)
        num = int(m.group(1)) if m else 4
        return {"type": "recommend", "num": max(1, min(8, num))}
    return None

def _code_num(code: str) -> int:
    m = re.search(r"(\d{3})", code or "")
    return int(m.group(1)) if m else math.inf

def _has_no_prereqs(row: Dict) -> bool:
    p = (row.get("prereqs") or "").strip().lower()
    return p == "" or p in {"none", "n/a", "na", "null"}

def _looks_intro_title(title: str) -> bool:
    t = (title or "").lower()
    return any(kw in t for kw in [
        "introduction", "intro ", "foundations", "fundamentals",
        "calculus", "discrete", "programming", "linear algebra", "statistics"
    ])

# Priority seeds for a CS freshman (picked from what’s in your CSV)
PREFERRED_STARTERS = [
    "CSC101", "CSC140", "CSC230",
    "MAT111", "MAT112", "MAT130",
    "MAT202", "STA210"
]

def recommend_starter_courses(rows: List[Dict], want: int = 4) -> List[Dict]:
    # 1) plausible intros
    intros = []
    for r in rows:
        code = r.get("code", "")
        num = _code_num(code)
        if num >= 300:
            continue
        if _has_no_prereqs(r) or _looks_intro_title(r.get("title", "")):
            intros.append(r)
    # 2) dedupe
    seen, dedup = set(), []
    for r in intros:
        c = r.get("code", "")
        if c not in seen:
            seen.add(c)
            dedup.append(r)
    # 3) boost preferred
    idx = {r["code"]: r for r in dedup}
    ordered = [idx[c] for c in PREFERRED_STARTERS if c in idx]
    # 4) fill the rest: “intro look” first, then lowest number
    rest = [r for r in dedup if r not in ordered]
    rest.sort(key=lambda r: (_looks_intro_title(r.get("title","")) is False, _code_num(r.get("code",""))))
    ordered.extend(rest)
    return ordered[:want]

# -------------------- Prompts & LLM --------------------

COURSE_PROMPT = """
You are AUIBPT, a university course assistant.
Answer ONLY using the provided course knowledge base (kb).
If the kb does not contain the answer, reply exactly: "I don't know from the provided data."
Do NOT include any hidden reasoning or analysis.
Put ONLY your final answer inside <final>...</final>.

kb:
{kb}

history:
{history}

question:
{question}
"""

CHAT_PROMPT = """
You are AUIBPT, a friendly assistant for casual conversation.
Respond naturally and helpfully in ≤2 sentences.
Do NOT mention any rules, knowledge bases, or prompts.
Put ONLY your final answer inside <final>...</final>.

history:
{history}

question:
{question}
"""

def make_llm():
    return OllamaLLM(
        model=MODEL_NAME,          # e.g., "llama3.1:8b"
        temperature=TEMPERATURE,   # from settings
        num_predict=NUM_PREDICT,   # from settings
        stop=["</final>"]          # helps prevent rambling after final
    )

# -------------------- Main chat loop --------------------

def run_chat(csv_path: str, index_dir: str, top_k: int, force_rebuild: bool = False):
    rows = load_catalog_rows(csv_path)
    print(f"Loaded {len(rows)} rows from CSV")
    docs = rows_to_documents(rows)

    if force_rebuild:
        print("Forcing rebuild from CSV ..."); rebuild_index(docs, index_dir)
    else:
        ensure_index(docs, index_dir)

    vs = load_index(index_dir)
    retriever = vs.as_retriever(search_kwargs={"k": top_k})

    llm = make_llm()
    course_chain = ChatPromptTemplate.from_template(COURSE_PROMPT) | llm
    chat_chain   = ChatPromptTemplate.from_template(CHAT_PROMPT)   | llm

    history = ""
    print("Hello! I'm AUIBPT. Type 'exit' to leave.")
    while True:
        q = input("You: ")
        if q.strip().lower() == "exit":
            break

        # A) Direct code match
        direct_rows = find_rows_by_code(rows, q)
        if direct_rows:
            kb = rows_to_kb(direct_rows)
            ans = ask_llm(course_chain, kb, history, q)
            print("AUIBPT:", ans)
            history += f"\nUser: {q}\nAI: {ans}"
            continue

        # A2) Direct title match
        title_rows = find_rows_by_title(rows, q)
        if title_rows:
            kb = rows_to_kb(title_rows)
            ans = ask_llm(course_chain, kb, history, q)
            print("AUIBPT:", ans)
            history += f"\nUser: {q}\nAI: {ans}"
            continue

        # B) Recommendation intent (freshman / starter path)
        reco = parse_recommendation_intent(q)
        if reco and is_coursey(q):
            picks = recommend_starter_courses(rows, want=reco["num"])
            if not picks:
                ans = "I don't know from the provided data."
                print("AUIBPT:", ans)
                history += f"\nUser: {q}\nAI: {ans}"
                continue
            lines = [f"{r['code']} — {r['title']}\n  {r['description']}" for r in picks]
            ans = "Here’s a good starter set:\n\n" + "\n\n".join(lines)
            print("AUIBPT:", ans)
            history += f"\nUser: {q}\nAI: {ans}"
            continue

        # C) Intent shortcuts (count/list)
        intent = parse_catalog_intent(q)
        if intent:
            if intent["type"] == "count":
                ans = f"I currently know {len(rows)} courses from courses.csv."
                print("AUIBPT:", ans)
                history += f"\nUser: {q}\nAI: {ans}"
                continue
            if intent["type"] == "list":
                limit = intent.get("limit", 100)
                lines = [f"{r['code']} — {r['title']}" for r in rows]
                if len(lines) > limit:
                    more = len(lines) - limit
                    lines = lines[:limit] + [f"...and {more} more."]
                ans = "\n".join(lines)
                print("AUIBPT:\n" + ans)
                history += f"\nUser: {q}\nAI:\n{ans}"
                continue

        # D) Not coursey → pure chat
        if not is_coursey(q):
            ans = ask_llm(chat_chain, "", history, q)
            print("AUIBPT:", ans)
            history += f"\nUser: {q}\nAI: {ans}"
            continue

        # E) Retrieval fallback for coursey questions
        norm_q = q.replace(" ", "")
        try:
            docs = retriever.invoke(q) or []
        except AttributeError:
            docs = retriever.get_relevant_documents(q) or []
        if not docs and norm_q != q:
            try:
                docs = retriever.invoke(norm_q) or []
            except AttributeError:
                docs = retriever.get_relevant_documents(norm_q) or []

        kb = "\n---\n".join(
            d.page_content + f"\n[source: {d.metadata.get('source','?')} | code: {d.metadata.get('code','?')}]"
            for d in docs
        ).strip()

        if not kb or kb == "(no relevant context found)":
            print("AUIBPT:", "I don't know from the provided data.")
            history += f"\nUser: {q}\nAI: I don't know from the provided data."
            continue

        ans = ask_llm(course_chain, kb, history, q)
        print("AUIBPT:", ans)
        history += f"\nUser: {q}\nAI: {ans}"

# -------------------- CLI --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild FAISS from courses.csv")
    parser.add_argument("--k", type=int, default=TOP_K, help="Retriever top-k (default from config)")
    args = parser.parse_args()
    run_chat(CSV_PATH, INDEX_DIR, args.k, force_rebuild=args.rebuild)
