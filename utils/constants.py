# Constants and configuration for AUIBPT

import re
from typing import Dict, Set, Tuple

# Course hints for detection
COURSE_HINTS = [
    "course", "class", "prereq", "prerequisite", "credit", "credits",
    "catalog", "syllabus", "covers", "topic", "learn", "teaches",
    "semester", "enroll", "registration", "requirement", "requirements",
    "what is", "describe", "explain", "about"
]

# Regex patterns
COURSE_CODE_RE = re.compile(r"\b([A-Za-z]{2,4})\s*-?\s*(\d{3})\b")
NAME_RE = re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z0-9_\- ]{1,40})\b", re.IGNORECASE)

# Department prefixes mapping
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

# Known colleges
KNOWN_COLLEGES = {"All", "CAS", "COP", "COD"}

# Language options
LANG_OPTIONS = {"English": "English", "Arabic": "Arabic"}

# Degree totals
DEGREE_TOTAL = {"CS": 126, "Pharmacy": 180, "Dentistry": 189}

# Major mapping
MAJOR_MAP = {
    "CS": {"college": "CAS", "prefixes": ("CSC", "MAT", "STA")},
    "Pharmacy": {"college": "COP", "prefixes": ("PHA", "CHE", "BIO")},
    "Dentistry": {"college": "COD", "prefixes": ("BDS", "BIO", "CHE")},
}

# Liberal Arts requirements
LA_REQUIREMENTS = {
    "General": 1,
    "Communication": 3,
    "Quantitative": 2,   # must be CSC101 and MAT101
    "Humanities": 4,
    "SocialScience": 2,
    "NaturalScience": 2,
}

# LA category mapping
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

# LA quantitative requirements (both required)
LA_QUANT_BOTH = {"CSC101", "MAT101"}

# Weights for schedule building
MAJOR_WEIGHT = 3
LA_WEIGHT = 1

# Difficulty weight mapping
DIFFICULTY_WEIGHT_MAP = {
    "Easy":   (2.0, 1.0),
    "Medium": (3.0, 1.0),
    "Hard":   (4.0, 1.0),
}
