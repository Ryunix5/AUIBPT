import re
import csv
from typing import List, Dict
from langchain.schema import Document

REQUIRED_COLS = {"code", "title", "description"}

def load_catalog_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = REQUIRED_COLS - cols
        if missing:
            raise ValueError(f"CSV file is missing required columns: {missing}")
        rows = list(reader)

    # 🔧 normalize (spaces/hyphens/case)
    for r in rows:
        r["code"] = re.sub(r"[\s\-]+", "", (r.get("code") or "")).upper().strip()
        r["title"] = (r.get("title") or "").strip()
        r["description"] = (r.get("description") or "").strip()
    return rows

def rows_to_documents(rows: List[Dict[str, str]]) -> List[Document]:
    """Convert catalog rows to LangChain Documents."""
    docs = []
    for row in rows:
        text = (
            f"Course Code: {row['code']}\n"
            f"Title: {row['title']}\n"
            f"Description: {row['description']}\n"
        )
        docs.append(Document(page_content=text, metadata={"source": "courses.csv", "code": row["code"]}))
    return docs
