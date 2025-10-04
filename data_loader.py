import csv
from typing import List, Dict
from langchain.schema import Document

REQUIRED_COLS = {"code", "title", "description"}

def load_catalog_rows(csv_path: str) -> List[Dict[str, str]]:
    """load and validate rows from the course catalog CSV file."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = REQUIRED_COLS - cols
        if missing:
            raise ValueError(f"CSV file is missing required columns: {missing}")
        return list(reader)

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
