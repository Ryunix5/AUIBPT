# data_loader.py
import csv
import re
from typing import List, Dict, Tuple, Optional

# ✅ LC ≥0.2: Document lives in langchain_core
from langchain_core.documents import Document

# CSV schema (case-insensitive, BOM-safe). Required: code, title, description.
# Preferred/optional: college, credits, prereqs.
_COLLEGE_KEYS = {"college", "school", "faculty"}
_CODE_KEYS    = {"code", "course code", "course", "id"}
_TITLE_KEYS   = {"title", "course title", "name"}
_DESC_KEYS    = {"description", "course description", "desc", "details", "about", "summary"}
_CRED_KEYS    = {"credits", "credit", "cr"}
_PRER_KEYS    = {"prereqs", "prerequisites", "prereq", "pre-reqs", "pre-req"}

_BOMS = ("\ufeff", "\ufffe")  # UTF-8/16

def _strip_bom(s: str) -> str:
    s = s or ""
    for bom in _BOMS:
        if s.startswith(bom):
            s = s[len(bom):]
    return s

def _open_csv_any_encoding(csv_path: str):
    """Prefer utf-8-sig so BOM is stripped automatically; fall back gracefully."""
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            f = open(csv_path, newline="", encoding=enc)
            rdr = csv.DictReader(f)
            if rdr.fieldnames:
                rdr.fieldnames = [_strip_bom((h or "").strip()) for h in rdr.fieldnames]
                return f, rdr
            f.close()
        except Exception as e:
            last_err = e
    raise last_err or ValueError("Could not open CSV with a known encoding.")

def _resolve_headers(fieldnames: List[str]) -> Tuple[Optional[str], str, str, str, Optional[str], Optional[str]]:
    """Return actual header names as seen in the CSV for the fields of interest."""
    if not fieldnames:
        raise ValueError("CSV has no headers.")
    m = { _strip_bom((h or "").strip()).lower(): h for h in fieldnames }

    def pick(candidates: set) -> Optional[str]:
        for c in candidates:
            if c in m:
                return m[c]
        return None

    h_college = pick(_COLLEGE_KEYS)         # optional (but recommended)
    h_code    = pick(_CODE_KEYS)            # required
    h_title   = pick(_TITLE_KEYS)           # required
    h_desc    = pick(_DESC_KEYS)            # required
    h_cred    = pick(_CRED_KEYS)            # optional
    h_prer    = pick(_PRER_KEYS)            # optional

    missing_required = [k for k, v in {"code":h_code, "title":h_title, "description":h_desc}.items() if v is None]
    if missing_required:
        raise ValueError(f"CSV missing required columns after mapping: {missing_required}")
    return h_college, h_code, h_title, h_desc, h_cred, h_prer

def _normalize_code(code: str) -> str:
    """CSC-101 -> CSC101; trim spaces; uppercase."""
    return re.sub(r"[\s\-]+", "", (code or "").upper()).strip()

def _normalize_college(college: str) -> str:
    """Normalize common college tags to short forms (e.g., CAS/COP/COD)."""
    c = (college or "").strip()
    if not c:
        return ""
    c_low = c.lower()
    if c.upper() in {"CAS", "COP", "COD", "COB", "CON"}:
        return c.upper()
    if "pharm" in c_low: return "COP"
    if "dent"  in c_low: return "COD"
    if "art"   in c_low or "science" in c_low: return "CAS"
    if "business" in c_low or "econ" in c_low: return "COB"
    if "nurs" in c_low or "health" in c_low:   return "CON"
    return c.upper()

def load_catalog_rows(csv_path: str) -> List[Dict[str, str]]:
    """
    Load rows from CSV.
    - Required: code, title, description
    - Optional: college, credits, prereqs
    - BOM/encoding safe; header names case-insensitive and tolerant to variants.
    """
    f, rdr = _open_csv_any_encoding(csv_path)
    try:
        h_college, h_code, h_title, h_desc, h_cred, h_prer = _resolve_headers(rdr.fieldnames or [])
        rows: List[Dict[str, str]] = []
        for r in rdr:
            r = { _strip_bom(k): v for k, v in r.items() }
            college = _normalize_college((r.get(h_college, "") if h_college else "").strip())
            code    = _normalize_code(r.get(h_code, ""))
            title   = (r.get(h_title, "") or "").strip()
            desc    = (r.get(h_desc, "") or "").strip()
            cred    = (r.get(h_cred, "") or "").strip() if h_cred else ""
            prer    = (r.get(h_prer, "") or "").strip() if h_prer else ""
            if not code and not title and not desc:
                continue
            rows.append({
                "college": college,
                "code": code,
                "title": title,
                "description": desc,
                "credits": cred,
                "prereqs": prer,
            })
        if not rows:
            raise ValueError("CSV appears to have no valid rows.")
        return rows
    finally:
        f.close()

def rows_to_documents(rows: List[Dict[str, str]]) -> List[Document]:
    """Convert catalog rows to LangChain Documents (for vector indexing)."""
    docs: List[Document] = []
    for r in rows:
        text = (
            f"College: {r.get('college','')}\n"
            f"Course Code: {r.get('code','')}\n"
            f"Title: {r.get('title','')}\n"
            f"Credits: {r.get('credits','')}\n"
            f"Prerequisites: {r.get('prereqs','')}\n"
            f"Description: {r.get('description','')}\n"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "source": "courses.csv",
                "code": r.get("code",""),
                "college": r.get("college",""),
                "title": r.get("title",""),
            }
        ))
    return docs
