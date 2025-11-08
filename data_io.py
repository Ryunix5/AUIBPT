# data_io.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import importlib.util
import os

import streamlit as st  # NEW

from data_loader import load_catalog_rows, rows_to_documents
from indexer import ensure_index, load_index, rebuild_index

# Cache CSV rows by (path, mtime) so changes invalidate cache
@st.cache_data(show_spinner=False)
def _load_rows_cached(csv_path: str, csv_mtime: float) -> List[Dict]:
    return load_catalog_rows(csv_path)

# Cache FAISS vector store in memory per index_dir
@st.cache_resource(show_spinner=False)
def _load_vs_cached(index_dir: str):
    return load_index(index_dir)

def _try_init_bm25(corpus_texts: List[str]):
    ...
    # (unchanged)
    ...

def build_or_load_index(csv_path: str, index_dir: str, force: bool = False) -> Tuple[List[Dict], object, Optional[object]]:
    # 1) CSV → rows (cached by file timestamp)
    csv_mtime = os.path.getmtime(csv_path) if os.path.exists(csv_path) else 0.0
    rows = _load_rows_cached(csv_path, csv_mtime)
    docs = rows_to_documents(rows)

    # 2) Ensure/rebuild index only when needed
    if force:
        rebuild_index(docs, index_dir)
        # reset session flag so we re-ensure next time if needed
        st.session_state.pop("index_ready", None)
    else:
        # Run ensure_index once per session (suppresses repeated "index found" prints)
        if not st.session_state.get("index_ready"):
            ensure_index(docs, index_dir)
            st.session_state.index_ready = True

    # 3) Load FAISS from disk (cached as a resource)
    vs = _load_vs_cached(index_dir)

    # 4) Optional BM25
    try:
        corpus_texts = [d.page_content for d in vs.docstore._dict.values()]
    except Exception:
        corpus_texts = []
    bm25 = _try_init_bm25(corpus_texts) if corpus_texts else None

    return rows, vs, bm25
