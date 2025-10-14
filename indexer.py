# indexer.py
import os, shutil
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

_EMBED = None  # lazy singleton so build+load use the exact same model

def _embeddings():
    global _EMBED
    if _EMBED is None:
        _EMBED = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _EMBED

def build_index(docs: List[Document], persist_dir: str) -> None:
    """Create FAISS from docs and save to disk."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, _embeddings())
    vs.save_local(persist_dir)

def load_index(persist_dir: str) -> FAISS:
    """Load an existing FAISS index from disk."""
    return FAISS.load_local(persist_dir, _embeddings(), allow_dangerous_deserialization=True)

def ensure_index(docs: List[Document], persist_dir: str) -> None:
    """Build the index only if it doesn't exist."""
    if not os.path.isdir(persist_dir):
        print("No index found — building it from courses.csv ...")
        build_index(docs, persist_dir)
        print("Index built ✔")
    else:
        print("Index found ✔")

def rebuild_index(docs: List[Document], persist_dir: str) -> None:
    """Force a fresh rebuild (handy during edits)."""
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir)
    build_index(docs, persist_dir)
