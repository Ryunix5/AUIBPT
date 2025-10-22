# Helper functions for AUIBPT

import re
import string
from typing import List, Dict, Set, Tuple

from .constants import COURSE_CODE_RE, NAME_RE, DEPT_PREFIXES, LA_CATEGORY, LA_REQUIREMENTS, LA_QUANT_BOTH

def _norm_text(s: str) -> str:
    """Normalize text for comparison."""
    return (s or "").lower().translate(str.maketrans("", "", string.punctuation)).strip()

def expand_synonyms(q: str) -> str:
    """Expand common abbreviations in queries."""
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
    """Calculate token overlap between two strings."""
    A, B = set(_norm_text(a).split()), set(_norm_text(b).split())
    return len(A & B)

def rows_to_kb(rows_subset: List[Dict]) -> str:
    """Convert rows to knowledge base format."""
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
    """Check if query is course-related."""
    if not q:
        return False
    ql = q.lower()
    if COURSE_CODE_RE.search(q):
        return True
    return any(h in ql for h in ["csc", "mat", "mth", "sta", "pha", "che", "bio", "bds", "pharm", "dent"])

def maybe_capture_name(q: str) -> str:
    """Extract name from query if present."""
    if not q:
        return None
    m = NAME_RE.search(q)
    if m:
        return m.group(1).strip()
    return None

def friendly_prefix(user_name: str = None) -> str:
    """Generate friendly prefix with user name."""
    return f"{user_name}, " if user_name else ""

def build_history_text(messages: List[Dict], max_turns: int = 10) -> str:
    """Build history text from messages."""
    msgs = messages[-2 * max_turns:] if messages else []
    lines = []
    for m in msgs:
        role = "User" if m["role"] == "user" else "AI"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)

def find_rows_by_code(rows: List[Dict], q: str) -> List[Dict]:
    """Find rows by course code."""
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
    """Find rows by title similarity."""
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
    """Infer department and college scopes from query."""
    ql = (q or "").lower()
    dept_scopes: List[str] = []
    college_scopes: List[str] = []
    
    if any(w in ql for w in ["computer science", " comp sci", " cs ", " c.s.", "csc", "programming"]): 
        dept_scopes.append("cs")
    if any(w in ql for w in ["math", "mathematics", " mat ", "mth", "algebra", "calculus"]): 
        dept_scopes.append("math")
    if any(w in ql for w in ["statistics", " sta ", "probability"]): 
        dept_scopes.append("stats")
    if any(w in ql for w in ["physics", "mechanics", "optics"]): 
        dept_scopes.append("physics")
    if any(w in ql for w in ["chemistry", "organic", "inorganic", "che "]): 
        dept_scopes.append("chem")
    if any(w in ql for w in ["biology", "bio", "genetics"]): 
        dept_scopes.append("bio")
    if any(w in ql for w in ["pharmacy", "pharm", "pha"]): 
        dept_scopes.append("pharm")
    if any(w in ql for w in ["dentistry", "dent", "bds"]): 
        dept_scopes.append("dent")
    
    for tag in {"CAS","COP","COD"}:
        if tag.lower() in ql:
            college_scopes.append(tag)
    
    return {"dept": dept_scopes or ["all"], "college": college_scopes or ["all"]}

def filter_rows_by_college(rows: List[Dict], college_tag: str) -> List[Dict]:
    """Filter rows by college."""
    if college_tag == "All":
        return rows
    return [r for r in rows if (r.get("college","").upper() == college_tag)]

def reorder_docs_by_scopes(docs: List, scopes: Dict[str, List[str]], college_filter: str) -> List:
    """Reorder documents based on scopes."""
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
    """Parse catalog browsing intent from query."""
    if not q:
        return None
    ql = q.lower()
    scopes = infer_scopes(q)
    if "how many" in ql and "course" in ql:
        return {"type": "count", "scopes": scopes}
    if any(kw in ql for kw in ["list all courses", "show all courses", "list courses", "all courses"]):
        return {"type": "list", "limit": 150, "scopes": scopes}
    return None

def _parse_prereq_codes(prereq_text: str) -> List[str]:
    """Parse prerequisite codes from text."""
    if not prereq_text:
        return []
    codes = []
    parts = re.split(r"[;,/]+", prereq_text)
    for p in parts:
        for m in re.finditer(r"\b([A-Za-z]{2,4})\s*-?\s*(\d{3})\b", p):
            codes.append((m.group(1) + m.group(2)).upper())
    return sorted(set(codes))

def _credits_from_str(x: str) -> int:
    """Extract credits from string."""
    if x is None: 
        return 3
    s = str(x).strip()
    if not s: 
        return 3
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else 3

def la_completed_counts(taken_codes: Set[str]) -> Dict[str, int]:
    """Count completed LA requirements."""
    counts = {k: 0 for k in LA_REQUIREMENTS}
    for code in taken_codes:
        cat = LA_CATEGORY.get(code.upper())
        if cat:
            counts[cat] += 1
    return counts

def la_remaining(counts: Dict[str, int], taken_codes: Set[str]) -> Dict[str, int]:
    """Calculate remaining LA requirements."""
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
    """Get LA recommendation pool."""
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
    """Get eligible major rows."""
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
    """Calculate total completed credits."""
    idx = {r["code"].upper(): r for r in rows_all}
    total = 0
    for c in taken_codes:
        r = idx.get(c.upper())
        if r:
            total += _credits_from_str(r.get("credits"))
    return total

def student_context_from_taken(rows_all: List[Dict], taken_codes: Set[str]) -> str:
    """Generate student context from completed courses."""
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
