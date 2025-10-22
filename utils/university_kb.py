# University knowledge base for AUIBPT

import os
import json
import re
import logging
from typing import Dict, List

log = logging.getLogger("app")

# University KB seed data
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
            "Recognized by Iraq's Ministry of Higher Education and Scientific Research.",
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

# University hook words for query detection
_UNIV_HOOK_WORDS = {
    "auib","american university of iraq","baghdad","cas","cop","cod","college of arts","college of pharmacy",
    "college of dentistry","dean","professor","faculty","mission","vision","accreditation","acpe","adee",
    "campus","al-faw","al faw","partnership","vanderbilt","temple","exeter","sapienza"
}

def load_university_kb() -> dict:
    """Load university knowledge base from file or return seed data."""
    path = "auib_university_kb.json"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict): 
                    raise ValueError("Root must be object")
                if "university" not in data or "faculty" not in data: 
                    raise ValueError("Missing keys")
                return data
        except Exception as e:
            log.warning(f"Failed reading auib_university_kb.json, using seed: {e}")
    return UNIV_KB_SEED

def is_university_query(q: str) -> bool:
    """Check if query is about university information."""
    if not q: 
        return False
    ql = q.lower()
    has_name = any(f["name"].lower() in ql for f in UNIV_KB_SEED.get("faculty", []))
    hook = any(w in ql for w in _UNIV_HOOK_WORDS)
    return has_name or hook

def univ_kb_blocks_for(q: str, limit: int = 24) -> str:
    """Generate university knowledge base blocks for a query."""
    ql = (q or "").lower()
    fac = UNIV_KB_SEED.get("faculty", [])
    scored = []
    
    for f in fac:
        blob = " ".join([f.get("name",""), f.get("title",""), f.get("areas",""), f.get("prior",""), f.get("college","")]).lower()
        A = set(re.sub(r"[^\w\s]", " ", ql).split())
        B = set(re.sub(r"[^\w\s]", " ", blob).split())
        score = len(A & B)
        if f.get("college","").lower() in ql: 
            score += 1
        scored.append((score, f))
    
    scored.sort(key=lambda x: (-x[0], x[1].get("college",""), x[1].get("name","")))
    fac_pick = [f for s,f in scored[:limit] if s > 0] or fac[: min(limit, 16)]

    lines = []
    uni = UNIV_KB_SEED.get("university", {})
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
