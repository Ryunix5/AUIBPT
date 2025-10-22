# Retrieval functionality for AUIBPT

import importlib.util
import logging
from typing import List, Dict, Tuple

from ..utils.helpers import expand_synonyms, simple_token_overlap

log = logging.getLogger("app")

def _try_init_bm25(corpus_texts: List[str]):
    """Try to initialize BM25 for hybrid retrieval."""
    try:
        spec = importlib.util.find_spec("rank_bm25")
        if spec is None: 
            return None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        BM25Okapi = getattr(mod, "BM25Okapi", None)
        if BM25Okapi is None: 
            return None
        tokenized = [t.split() for t in corpus_texts]
        return BM25Okapi(tokenized)
    except Exception:
        return None

def hybrid_retrieve(q: str, retriever, vs, top_k: int, bm25=None) -> List:
    """Hybrid retrieval using both vector search and BM25."""
    if not q:
        return []
    
    qx = expand_synonyms(q)
    vec_docs = []
    
    try:
        vec_docs = retriever.invoke(qx) or []
    except AttributeError:
        vec_docs = retriever.get_relevant_documents(qx) or []
    except Exception as e:
        log.error(f"Vector retrieve error: {e}")
    
    bm_docs = []
    if bm25 is not None:
        try:
            all_docs = list(vs.docstore._dict.values())
            scores = bm25.get_scores(qx.split())
            best_ids = sorted(range(len(all_docs)), key=lambda i: -scores[i])[:top_k]
            bm_docs = [all_docs[i] for i in best_ids]
        except Exception as e:
            log.warning(f"BM25 failed, continuing with vectors only: {e}")
    
    keyed = {}
    for d in (vec_docs + bm_docs):
        keyed[(d.metadata.get("code"), d.page_content)] = d
    
    merged = list(keyed.values())
    merged.sort(key=lambda d: -simple_token_overlap(qx, d.page_content))
    return merged[: max(top_k * 2, top_k)]

def prepare_kb_from_docs(docs) -> str:
    """Prepare knowledge base from documents."""
    if not docs:
        return ""
    blocks = []
    for d in docs:
        meta = d.metadata or {}
        text = d.page_content
        blocks.append(text + f"\n[source: {meta.get('source','?')} | code: {meta.get('code','?')}]")
    return "\n---\n".join(blocks).strip()
