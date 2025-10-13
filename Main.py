# main.py
# Run once (if you switch models):  ollama pull llama3.1:8b
from typing import List, Dict
import argparse, json, re, string
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

TITLE_ALIASES = {
    # query text (lowercase) -> canonical title substring to search
    "computer networks": "introduction to computer networks",
    "computer network": "introduction to computer networks",
    "networks": "introduction to computer networks",
    "data structures": "data structure",
    "data structure": "data structure",
    "machine learning": "machine learning",
    "deep learning": "deep learning",
    "digital logic": "digital logic",
}

FOLLOWUP_HINTS = {"what is it", "what is that", "prereq", "prereqs", "prerequisite", "prerequisites"}

COURSE_CODE_RE = re.compile(r"\b([A-Za-z]{2,4})\s*-?\s*(\d{3})\b")

def is_coursey(q: str) -> bool:
    ql = q.lower()
    if COURSE_CODE_RE.search(q):
        return True
    return any(h in ql for h in COURSE_HINTS + ["csc", "mat", "mth", "sta"])

def is_followup(q: str) -> bool:
    ql = q.lower().strip("?!. ")
    return ql in FOLLOWUP_HINTS or any(h in ql for h in FOLLOWUP_HINTS)

def _to_str(x):
    try:
        from langchain_core.messages import AIMessage
        if isinstance(x, AIMessage):
            return x.content or ""
    except Exception:
        pass
    return x if isinstance(x, str) else str(x)

def _clean_output(text: str) -> str:
    # Keep only the last <final>...</final> if any
    finals = re.findall(r"<final>(.*?)</final>", text, flags=re.DOTALL | re.IGNORECASE)
    if finals:
        text = finals[-1]
    # Strip any think/echoing
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r'(?i)\brules:\b.*', "", text)
    text = re.sub(r'(?i)\b(knowledge base|kb|instructions)\b.*', "", text)
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
    """Loose title match for queries like 'what is data structures?' (with aliases)."""
    qn = _norm_text(q)
    for k, v in TITLE_ALIASES.items():
        if k in qn:
            qn = v
            break

    q_tokens = set(qn.split())
    hits = []
    for r in rows:
        tn = _norm_text(r.get("title", ""))
        if not tn:
            continue
        t_tokens = set(tn.split())
        overlap = len(q_tokens & t_tokens)
        if overlap >= 2 or qn in tn:
            hits.append((overlap, r))
    hits.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in hits]

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

# -------------------- Prompts & LLM --------------------

COURSE_PROMPT = """
You are AUIBT, a university course assistant.
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
You are AUIBT, a friendly assistant for casual conversation.
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
        model=MODEL_NAME,       # e.g., "llama3.1:8b"
        temperature=0.4,
        num_predict=192,
        stop=["</final>"]
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
    last_rows: List[Dict] = []

    print("Hello! I'm AUIBT. Type 'exit' to leave.")
    while True:
        q = input("You: ")
        if q.strip().lower() == "exit":
            break

        # Follow-up on last course(s)? (e.g., "prereqs?" / "what is it?")
        if is_followup(q) and last_rows:
            parts = []
            for r in last_rows:
                piece = f"{r['code']} — {r['title']}"
                if "prereq" in q.lower() and r.get("prereqs"):
                    piece += f"\nPrereqs: {r['prereqs']}"
                else:
                    piece += f"\nDescription: {r['description']}"
                parts.append(piece)
            ans = "\n\n".join(parts)
            print("AUIBT:", ans)
            history += f"\nUser: {q}\nAI: {ans}"
            continue

        # A) Direct code match (deterministic answer from CSV)
        direct_rows = find_rows_by_code(rows, q)
        if direct_rows:
            parts = []
            for row in direct_rows:
                piece = f"Course Code: {row['code']}\nTitle: {row['title']}\nDescription: {row['description']}"
                if row.get("prereqs"):
                    piece += f"\nPrereqs: {row['prereqs']}"
                parts.append(piece)
            ans = "\n\n".join(parts)
            print("AUIBT:", ans)
            last_rows = direct_rows[:]
            history += f"\nUser: {q}\nAI: {ans}"
            continue

        # A2) Direct title match (deterministic answer from CSV)
        title_rows = find_rows_by_title(rows, q)
        if title_rows:
            parts = []
            for r in title_rows[:3]:
                piece = f"Course Code: {r['code']}\nTitle: {r['title']}\nDescription: {r['description']}"
                if r.get("prereqs"):
                    piece += f"\nPrereqs: {r['prereqs']}"
                parts.append(piece)
            ans = "\n\n".join(parts)
            print("AUIBT:", ans)
            last_rows = title_rows[:]
            history += f"\nUser: {q}\nAI: {ans}"
            continue

        # B) Intent shortcuts
        intent = parse_catalog_intent(q)
        if intent:
            if intent["type"] == "count":
                ans = f"I currently know {len(rows)} courses from courses.csv."
                print("AUIBT:", ans)
                history += f"\nUser: {q}\nAI: {ans}"
                continue
            if intent["type"] == "list":
                limit = intent.get("limit", 100)
                lines = [f"{r['code']} — {r['title']}" for r in rows]
                if len(lines) > limit:
                    more = len(lines) - limit
                    lines = lines[:limit] + [f"...and {more} more."]
                ans = "\n".join(lines)
                print("AUIBT:\n" + ans)
                history += f"\nUser: {q}\nAI:\n{ans}"
                continue

        # C) Not coursey → pure chat
        if not is_coursey(q):
            ans = ask_llm(chat_chain, "", history, q)
            print("AUIBT:", ans)
            history += f"\nUser: {q}\nAI: {ans}"
            continue

        # D) Retrieval fallback for coursey questions
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
            print("AUIBT:", "I don't know from the provided data.")
            history += f"\nUser: {q}\nAI: I don't know from the provided data."
            continue

        ans = ask_llm(course_chain, kb, history, q)
        print("AUIBT:", ans)

        # Try to remember the last course from kb for follow-ups
        m = re.search(r"code:\s*([A-Z]{2,4}\s*-?\s*\d{3})", kb, flags=re.IGNORECASE)
        if m:
            code_key = re.sub(r"\s|-", "", m.group(1)).upper()
            last_rows = [r for r in rows if r["code"].replace(" ", "").upper() == code_key] or last_rows

        history += f"\nUser: {q}\nAI: {ans}"

# -------------------- CLI --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild FAISS from courses.csv")
    parser.add_argument("--k", type=int, default=TOP_K, help="Retriever top-k (default from config)")
    args = parser.parse_args()
    run_chat(CSV_PATH, INDEX_DIR, args.k, force_rebuild=args.rebuild)
