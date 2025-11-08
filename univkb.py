# univkb.py
from __future__ import annotations
import os, json, re
from typing import Dict, List

# Path to your JSON knowledge base
_KB_PATH = "auib_university_kb.json"

# Minimal fallback (used only if the JSON file is missing or invalid)
_FALLBACK: Dict = {"university": {}, "faculty": []}

def load_university_kb(path: str = _KB_PATH) -> Dict:
    """Load the institutional KB from JSON. Falls back to a minimal structure."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "university" in data and "faculty" in data:
                return data
        except Exception:
            pass
    return _FALLBACK

UNIV_KB: Dict = load_university_kb()

# Words that suggest the user is asking about AUIB / colleges / leadership
_UNIV_HOOK_WORDS = {
    "auib","american university of iraq","baghdad","cas","cop","cod",
    "college of arts","college of pharmacy","college of dentistry",
    "dean","professor","faculty","mission","vision","accreditation","acpe","adee",
    "campus","al-faw","al faw","partnership","vanderbilt","temple","exeter","sapienza"
}

def is_university_query(q: str) -> bool:
    """True if the prompt likely targets the AUIB institutional KB."""
    if not q:
        return False
    ql = q.lower()
    # Check if a faculty member’s full name appears, or any hook word
    has_name = any(
        isinstance(f, dict) and "name" in f and isinstance(f["name"], str)
        and f["name"].lower() in ql
        for f in UNIV_KB.get("faculty", [])
    )
    hook = any(w in ql for w in _UNIV_HOOK_WORDS)
    return has_name or hook

def _tokenize(s: str) -> List[str]:
    return re.sub(r"[^\w\s]", " ", (s or "")).lower().split()

def univ_kb_blocks_for(q: str, limit: int = 24) -> str:
    """
    Return a short, LLM-friendly text block summarizing:
      - relevant faculty (rough match to the query)
      - high-level university facts from the KB
    """
    q_tokens = set(_tokenize(q))
    fac = UNIV_KB.get("faculty", []) or []
    scored: List[tuple[int, dict]] = []

    # Crude token-overlap scoring for faculty relevance
    for f in fac:
        blob = " ".join([
            str(f.get("name","")), str(f.get("title","")), str(f.get("areas","")),
            str(f.get("prior","")), str(f.get("college",""))
        ])
        score = len(q_tokens & set(_tokenize(blob)))
        if score > 0:
            scored.append((score, f))

    scored.sort(key=lambda t: t[0], reverse=True)
    top_fac = [f for _, f in scored[:max(1, min(limit, 24))]]

    lines: List[str] = []

    # University overview
    uni = UNIV_KB.get("university", {})
    if isinstance(uni, dict):
        facts = uni.get("facts") or []
        accs = uni.get("accreditations") or []
        leader = uni.get("leadership") or []
        progs = uni.get("programs") or []
        parts = uni.get("partnerships") or []

        if facts:  lines.append("University facts: " + " | ".join(map(str, facts)))
        if accs:   lines.append("Accreditations: " + " | ".join(map(str, accs)))
        if leader: lines.append("Leadership: " + " | ".join(map(str, leader)))
        if progs:  lines.append("Programs: " + " | ".join(map(str, progs)))
        if parts:  lines.append("Partnerships: " + " | ".join(map(str, parts)))

    # Faculty cards
    if top_fac:
        lines.append("Faculty:")
        for f in top_fac:
            segs = []
            name = f.get("name")
            if name: segs.append(str(name))
            title = f.get("title")
            if title: segs.append(str(title))
            college = f.get("college")
            if college: segs.append(str(college))
            areas = f.get("areas")
            if areas: segs.append(f"Areas: {areas}")
            prior = f.get("prior")
            if prior: segs.append(f"Prior: {prior}")
            notes = f.get("notes")
            if notes: segs.append(f"Notes: {notes}")
            lines.append(" - " + " | ".join(segs))

    return "\n".join(lines).strip()
