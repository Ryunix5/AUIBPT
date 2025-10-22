# app.py - Refactored AUIBPT Application
# AUIBPT — Course, Liberal-Arts & Schedule Builder (Ryunix build)

import os
import logging
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

# Import our modules
from utils.constants import KNOWN_COLLEGES, LANG_OPTIONS
from utils.university_kb import load_university_kb, is_university_query, univ_kb_blocks_for
from core.llm import make_llm, _make_groq_llm
from core.retrieval import _try_init_bm25
from ui.theme import setup_page_config, setup_global_styles, setup_footer, apply_theme, _render_appearance_controls
from ui.chat import render_history, handle_chat_input
from ui.schedule_builder import render_schedule_builder

# Import settings and data loaders
from settings import MODEL_NAME, CSV_PATH, INDEX_DIR, TOP_K, TEMPERATURE, NUM_PREDICT, USE_OPENAI
from data_loader import load_catalog_rows, rows_to_documents
from indexer import ensure_index, load_index, rebuild_index

# Setup logging
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("app")

# Load university KB
UNIV_KB = load_university_kb()

# Setup page configuration
setup_page_config()
setup_global_styles()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "_splash_shown" not in st.session_state:
    st.session_state._splash_shown = False
if "profile_avatar_path" not in st.session_state:
    st.session_state.profile_avatar_path = None

# Theme state
if "theme_primary" not in st.session_state:
    st.session_state.theme_primary = "#4d1212"
if "theme_bg" not in st.session_state:
    st.session_state.theme_bg = "#000000"
if "theme_text" not in st.session_state:
    st.session_state.theme_text = "#FFFFFF"

# Schedule state
if "schedule_slots" not in st.session_state:
    st.session_state.schedule_slots = []
if "schedule_planned_credits" not in st.session_state:
    st.session_state.schedule_planned_credits = 0
if "schedule_target_credits" not in st.session_state:
    st.session_state.schedule_target_credits = 15
if "schedule_major_key" not in st.session_state:
    st.session_state.schedule_major_key = "CS"
if "completed_codes_all" not in st.session_state:
    st.session_state.completed_codes_all = set()
if "swap_history" not in st.session_state:
    st.session_state.swap_history = []
if "show_schedule" not in st.session_state:
    st.session_state.show_schedule = False
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

# Apply current theme
apply_theme(st.session_state.get("theme_primary", "#4f46e5"),
            st.session_state.get("theme_bg", "#0b1220"),
            st.session_state.get("theme_text", "#e2e8f0"))

# Show splash if not shown
if not st.session_state._splash_shown:
    from ui.theme import show_splash
    show_splash()

# Data & Index loading
@st.cache_data(show_spinner=True, ttl=60)
def _load_rows(csv_path: str):
    return load_catalog_rows(csv_path)

@st.cache_resource(show_spinner=True)
def _build_or_load_index(csv_path: str, index_dir: str, force: bool):
    rows = _load_rows(csv_path)
    docs = rows_to_documents(rows)
    if force:
        rebuild_index(docs, index_dir)
    else:
        ensure_index(docs, index_dir)
    vs = load_index(index_dir)
    try:
        corpus_texts = [d.page_content for d in vs.docstore._dict.values()]
    except Exception:
        corpus_texts = []
    bm25 = _try_init_bm25(corpus_texts) if corpus_texts else None
    return rows, vs, bm25

# Header
status_col1, status_col2, status_col3 = st.columns([1.3, 1, 1])
with status_col1:
    st.title("AUIBPT — AUIB Course Chatbot", anchor=None, help=None, width="stretch")
    st.caption(f"Model: `{MODEL_NAME}` • k={TOP_K} • T={TEMPERATURE} • max={NUM_PREDICT}")
    st.caption("BETA — AUIBPT (Ryunix Build)")

with status_col3:
    with st.expander("Appearance", expanded=False):
        _render_appearance_controls()
    
    with st.expander("Settings", expanded=False):
        answer_lang = st.selectbox("Answer language", ["English","Arabic"], index=0)
        debug = st.toggle("Debug", help="Show knowledge base and timing details")
        
        # College filter
        try:
            colleges = sorted(list(KNOWN_COLLEGES))
        except Exception:
            colleges = []
        options = ["All"] + colleges
        if "college_filter" not in st.session_state:
            st.session_state.college_filter = "All"
        try:
            _idx = options.index(st.session_state.college_filter)
        except ValueError:
            _idx = 0
        college_filter = st.selectbox("College filter", options, index=_idx)
        st.session_state.college_filter = college_filter

    toggle_label = "Close Schedule Builder" if st.session_state.get("show_schedule", False) else "Open Schedule Builder"
    cols_hdr = st.columns(2)
    with cols_hdr[0]:
        if st.button(toggle_label, key="toggle_schedule_hdr"):
            st.session_state.show_schedule = not st.session_state.get("show_schedule", False)
            st.rerun()
    with cols_hdr[1]:
        clear = st.button("Clear chat")
    
    exists = os.path.exists(CSV_PATH)
    st.caption(f"CSV: {'found' if exists else 'missing'}")
    
    force_rebuild = st.checkbox("Rebuild FAISS index from CSV (one-time)", value=False)

# Load data and index
try:
    rows_all, vs, bm25 = _build_or_load_index(CSV_PATH, INDEX_DIR, force_rebuild)
    college_filter = st.session_state.get("college_filter", "All")
    from utils.helpers import filter_rows_by_college
    rows = filter_rows_by_college(rows_all, college_filter)
    retriever = vs.as_retriever(search_kwargs={"k": int(TOP_K)})
except Exception as e:
    st.error(f"Failed to prepare index or load catalog: {e}")
    st.stop()

# Initialize LLM
llm = make_llm(MODEL_NAME, TEMPERATURE, NUM_PREDICT)

# Create chains
from ui.chat import COURSE_PROMPT, CHAT_PROMPT, UNIV_PROMPT
course_chain = ChatPromptTemplate.from_template(COURSE_PROMPT) | llm
chat_chain   = ChatPromptTemplate.from_template(CHAT_PROMPT)   | llm
univ_chain   = ChatPromptTemplate.from_template(UNIV_PROMPT)   | llm

# Groq fallback chains
groq_fallback_course = None
groq_fallback_chat   = None
groq_fallback_univ   = None

# Get GROQ key
from core.llm import GROQ_KEY
if GROQ_KEY:
    try:
        groq_llm = _make_groq_llm("llama-3.1-8b-instant", TEMPERATURE, NUM_PREDICT)
        groq_fallback_course = ChatPromptTemplate.from_template(COURSE_PROMPT) | groq_llm
        groq_fallback_chat   = ChatPromptTemplate.from_template(CHAT_PROMPT)   | groq_llm
        groq_fallback_univ   = ChatPromptTemplate.from_template(UNIV_PROMPT)   | groq_llm
    except Exception as e:
        st.warning(f"Groq fallback unavailable: {e}")

# Render chat history
render_history()

# Chat input
input_disabled = st.session_state.get("is_generating", False)
if st.session_state.get("is_generating", False):
    st.info("Assistant is generating a response...")

q = st.chat_input("Ask anything", disabled=input_disabled)

if q is None:
    pass
elif not q.strip():
    st.warning("Please type a message first.")
else:
    handle_chat_input(
        q, rows_all, rows, college_filter, retriever, vs, bm25,
        course_chain, chat_chain, univ_chain,
        groq_fallback_course, groq_fallback_chat, groq_fallback_univ,
        answer_lang, debug, LANG_OPTIONS
    )

# Clear chat action
try:
    if clear:
        st.session_state.messages = []
        st.rerun()
except NameError:
    pass

# Schedule Builder
if st.session_state.get("show_schedule", False):
    render_schedule_builder(rows_all, vs, bm25)

# Setup footer
setup_footer()
