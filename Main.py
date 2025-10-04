# main.py
# Run:  pip install langchain langchain-community faiss-cpu sentence_transformers
#       ollama pull deepseek-r1:8b
from typing import List
import argparse, json
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

from settings import MODEL_NAME, CSV_PATH, INDEX_DIR, TOP_K, USE_JSON, TEMPERATURE, NUM_PREDICT
from data_loader import load_catalog_rows, rows_to_documents
from indexer import ensure_index, load_index, rebuild_index
# --- Intent & helpers ---
from typing import List, Dict, Tuple

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
        return {"type": "list", "scopes": _detect_scopes(ql)}
    return None

def is_followup_list(q: str) -> bool:
    ql = q.lower()
    # handle follow-ups like "list them", "list those", or just "list"
    return ("list them" in ql) or ("list those" in ql) or (ql.strip() == "list") or (ql.strip() == "list them")

def _is_cs(row: Dict[str,str]) -> bool:
    code = row["code"].upper()
    return code.startswith(("CSC","CSE","CS"))

def _is_math(row: Dict[str,str]) -> bool:
    code = row["code"].upper()
    return code.startswith(("MAT","MTH","MATH","STA"))  # counts Statistics with Math

def filter_by_scope(rows: List[Dict[str,str]], scope: str) -> List[Dict[str,str]]:
    if scope == "cs":
        return [r for r in rows if _is_cs(r)]
    if scope == "math":
        return [r for r in rows if _is_math(r)]
    return rows[:]  # 'all'

def format_list(rows: List[Dict[str,str]], limit: int = 100) -> str:
    lines = [f"{r['code']} — {r['title']}" for r in rows]
    if len(lines) > limit:
        return "\n".join(lines[:limit] + [f"...and {len(lines) - limit} more."])
    return "\n".join(lines)


# ---- Prompt (tight + JSON mode by default) ----
JSON_PROMPT = """
You are AUIBT, a curriculum assistant. Use ONLY the kb. If insufficient, answer "I don't know."

Return STRICT JSON with exactly:
{{
  "answer": "final answer",
  "sources": ["courses.csv"] // or [] if none used
}}

kb:
{kb}

history:
{history}

question:
{question}
"""

PLAIN_PROMPT = """
You are AUIBT, a curriculum assistant. Use ONLY the kb below. If insufficient, say "I don't know."
Do not include your reasoning.

kb:
{kb}
history:
{history}

question:
{question}
"""

def make_llm():
    kwargs = dict(model=MODEL_NAME, temperature=TEMPERATURE, num_predict=NUM_PREDICT)
    if USE_JSON:
        kwargs["format"] = "json"   # keeps outputs clean; no thinking blocks
    return Ollama(**kwargs)

def catalog_intent(q: str):
    ql = q.lower()
    if "how many" in ql and "course" in ql:
        return {"type": "count"}
    if any(kw in ql for kw in ["list all courses", "show all courses", "list courses", "all courses"]):
        return {"type": "list", "limit": 100}
    return None

def run_chat(csv_path: str, index_dir: str, top_k: int, force_rebuild: bool = False):
    # 1) Load CSV rows and convert to docs
    rows = load_catalog_rows(csv_path)
    docs = rows_to_documents(rows)

    # 2) Build/load FAISS
    if force_rebuild:
        print("Forcing rebuild from CSV ...")
        rebuild_index(docs, index_dir)
    else:
        ensure_index(docs, index_dir)
    vs = load_index(index_dir)
    retriever = vs.as_retriever(search_kwargs={"k": top_k})

    # 3) LLM + prompt chain
    llm = make_llm()
    prompt = ChatPromptTemplate.from_template(JSON_PROMPT if USE_JSON else PLAIN_PROMPT)
    chain = prompt | llm

    # 4) Chat loop
    history = ""
    print("Hello! I'm AUIBT. Type 'exit' to leave.")
    while True:
        q = input("You: ")
        if q.strip().lower() == "exit":
            break

        # Intent short-circuit (count/list) uses CSV directly
        intent = catalog_intent(q)
        if intent:
            if intent["type"] == "count":
                msg = f"I currently know {len(rows)} courses from courses.csv."
                print("AUIBT:", msg); history += f"\nUser: {q}\nAI: {msg}"
                continue
            if intent["type"] == "list":
                lines = [f"{r['code']} — {r['title']}" for r in rows]
                if len(lines) > intent["limit"]:
                    more = len(lines) - intent["limit"]
                    lines = lines[:intent["limit"]] + [f"...and {more} more."]
                msg = "\n".join(lines)
                print("AUIBT:\n" + msg); history += f"\nUser: {q}\nAI:\n{msg}"
                continue

        # Retrieve kb for this question
        try:
            docs = retriever.invoke(q)
        except AttributeError:
            docs = retriever.get_relevant_documents(q)
        kb = "\n---\n".join(
            d.page_content + f"\n[source: {d.metadata.get('source','?')} | code: {d.metadata.get('code','?')}]"
            for d in docs
        ) or "(no relevant context found)"

        # Ask the model
        raw = chain.invoke({"kb": kb, "history": history, "question": q})

        # Print final answer (if JSON mode: parse and show only the answer)
        if USE_JSON:
            try:
                data = json.loads(raw)
                ans = data.get("answer", "").strip()
            except Exception:
                ans = raw.strip()
        else:
            ans = raw.strip()

        print("AUIBT:", ans)
        history += f"\nUser: {q}\nAI: {ans}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild FAISS from courses.csv")
    parser.add_argument("--k", type=int, default=TOP_K, help="Retriever top-k (default from config)")
    args = parser.parse_args()
    run_chat(CSV_PATH, INDEX_DIR, args.k, force_rebuild=args.rebuild)
