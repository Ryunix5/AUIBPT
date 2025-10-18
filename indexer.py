# indexer.py
import os
import shutil
from typing import List, TYPE_CHECKING

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Prefer new splitters package; fall back to old location if needed
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # LC ≥0.2 split package
except ModuleNotFoundError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter   # older LC

# 👉 Import Document only for type checking (no runtime dependency on its location)
if TYPE_CHECKING:
    from langchain_core.documents import Document  # LC ≥0.2
else:
    Document = object  # sentinel for type hints without importing at runtime

_EMBED = None
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
META_FILE = "index_meta.txt"  # store embedding model name to detect mismatch

def _embeddings():
    global _EMBED
    if _EMBED is None:
        _EMBED = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _EMBED

def _write_meta(persist_dir: str):
    os.makedirs(persist_dir, exist_ok=True)
    with open(os.path.join(persist_dir, META_FILE), "w", encoding="utf-8") as f:
        f.write(EMBED_MODEL)

def _read_meta(persist_dir: str) -> str | None:
    path = os.path.join(persist_dir, META_FILE)
    if not os.path.exists(path):
        return None
    try:
        return open(path, encoding="utf-8").read().strip()
    except Exception:
        return None

def build_index(docs: List[Document], persist_dir: str) -> None:
    """Create FAISS from docs and save to disk."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, _embeddings())
    vs.save_local(persist_dir)
    _write_meta(persist_dir)

def load_index(persist_dir: str) -> FAISS:
    """Load an existing FAISS index from disk; auto-rebuild if model meta mismatches."""
    meta = _read_meta(persist_dir)
    if meta and meta != EMBED_MODEL:
        raise RuntimeError(f"Embedding mismatch: index has '{meta}', code expects '{EMBED_MODEL}'")
    return FAISS.load_local(persist_dir, _embeddings(), allow_dangerous_deserialization=True)

def ensure_index(docs: List[Document], persist_dir: str) -> None:
    """Build the index only if it doesn't exist; rebuild if corrupt."""
    if not os.path.isdir(persist_dir):
        print("No index found — building it from courses.csv ...")
        build_index(docs, persist_dir)
        print("Index built ✔")
        return
    try:
        meta = _read_meta(persist_dir)
        if (meta is not None) and (meta != EMBED_MODEL):
            print("Embedding model changed — rebuilding index ...")
            shutil.rmtree(persist_dir, ignore_errors=True)
            build_index(docs, persist_dir)
            print("Index rebuilt ✔")
        else:
            _ = FAISS.load_local(persist_dir, _embeddings(), allow_dangerous_deserialization=True)
            print("Index found ✔")
    except Exception:
        print("Index corrupted — rebuilding ...")
        shutil.rmtree(persist_dir, ignore_errors=True)
        build_index(docs, persist_dir)
        print("Index rebuilt ✔")

def rebuild_index(docs: List[Document], persist_dir: str) -> None:
    """Force a fresh rebuild (handy during edits)."""
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)
    build_index(docs, persist_dir)
