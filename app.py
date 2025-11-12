"""Streamlit entrypoint for the AUIBPT assistant."""
from __future__ import annotations

import logging
import os
import time
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

from data_io import build_or_load_index
from ui import apply_theme, render_appearance_controls
from univkb import is_university_query, univ_kb_blocks_for

from app_core.auth import (
    sb,
    create_chat,
    current_user,
    delete_chat,
    ensure_current_chat,
    list_chats,
    load_chat_messages_into_ui,
    save_message,
    sign_in,
    sign_out,
    sign_up,
    rename_chat,
)
from app_core.knowledge import (
    CHUNK_CHAR_CAP,
    LANG_OPTIONS,
    KNOWN_COLLEGES,
    build_history_text,
    build_kb_from_docs,
    expand_synonyms,
    fastpath_course_code,
    filter_rows_by_college,
    find_rows_by_code,
    find_rows_by_title,
    friendly_prefix,
    hybrid_retrieve,
    infer_scopes,
    is_coursey,
    load_general_kb,
    maybe_capture_name,
    needs_prep_tips,
    parse_catalog_intent,
    reorder_docs_by_scopes,
    rows_to_kb,
    web_enrichment_snippet,
)
from app_core.llm import ask_llm_stream, make_llm
from app_core.prompts import COURSE_PROMPT, UNIV_PROMPT
from app_core.schedule import render_schedule_builder, student_context_from_taken

# ---------------------- Settings (safe fallbacks) ----------------------
try:
    from settings import MODEL_NAME, CSV_PATH, INDEX_DIR, TOP_K, TEMPERATURE, NUM_PREDICT, USE_OPENAI
except Exception:
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    CSV_PATH = os.getenv("CSV_PATH", "course.csv")
    INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
    TOP_K = int(os.getenv("TOP_K", "3"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
    NUM_PREDICT = int(os.getenv("NUM_PREDICT", "512"))
    USE_OPENAI = os.getenv("USE_OPENAI", "true").strip().lower() in {"1", "true", "yes", "on"}
else:
    TOP_K = int(TOP_K)

try:
    from settings import USE_GROQ_ONLY as _USE_GROQ_ONLY
except Exception:
    _USE_GROQ_ONLY = None

_USE_GROQ_ONLY_ENV = os.getenv("USE_GROQ_ONLY", "").strip().lower()
if _USE_GROQ_ONLY_ENV in {"1", "true", "yes", "on"}:
    USE_GROQ_ONLY = True
elif _USE_GROQ_ONLY_ENV in {"0", "false", "no", "off"}:
    USE_GROQ_ONLY = False
elif isinstance(_USE_GROQ_ONLY, bool):
    USE_GROQ_ONLY = _USE_GROQ_ONLY
else:
    USE_GROQ_ONLY = False

# ---------------------- Theme & session defaults ----------------------
if "theme_primary" not in st.session_state:
    st.session_state.theme_primary = "#4d1212"
if "theme_bg" not in st.session_state:
    st.session_state.theme_bg = "#000000"
if "theme_text" not in st.session_state:
    st.session_state.theme_text = "#e2e8f0"
apply_theme(st.session_state.theme_primary, st.session_state.theme_bg, st.session_state.theme_text)

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("app")

for key, default in {
    "messages": [],
    "user_name": None,
    "profile_avatar_path": None,
    "schedule_slots": [],
    "schedule_planned_credits": 0,
    "schedule_target_credits": 15,
    "schedule_major_key": "CS",
    "completed_codes_all": set(),
    "swap_history": [],
    "show_schedule": False,
    "is_generating": False,
    "answer_lang": "English",
    "debug": False,
    "college_filter": "All",
}.items():
    st.session_state.setdefault(key, default)

# ---------------------- Header / status ----------------------
status_col1, status_col2, status_col3 = st.columns([1.4, 1, 1])
with status_col1:
    st.markdown("### AUIBPT")
    st.caption(f"Model: `{MODEL_NAME}` • k={TOP_K} • T={TEMPERATURE} • max={NUM_PREDICT}")
with status_col3:
    with st.expander("Chats", expanded=False):
        if sb is None:
            st.info("Sign in to save and manage chats.")
        else:
            uid, email = current_user()
            if not uid:
                st.info("Please sign in to view your chats.")
            else:
                ensure_current_chat()
                try:
                    chat_rows = list_chats(uid) or []
                except Exception as exc:
                    chat_rows = []
                    st.warning(f"Could not fetch chats: {exc}")

                c1, c2 = st.columns([0.6, 0.4])
                with c1:
                    new_title = st.text_input("New chat title", value="New chat", key="new_chat_title_hdr")
                with c2:
                    if st.button("Create chat", use_container_width=True, key="btn_create_chat_hdr"):
                        try:
                            chat_id = create_chat(uid, new_title.strip() or "New chat")
                            st.session_state.current_chat_id = chat_id
                            st.session_state.messages = []
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Create failed: {exc}")

                if chat_rows:
                    titles = [row.get("title") or "(untitled)" for row in chat_rows]
                    ids = [row["id"] for row in chat_rows]
                    current_chat = st.session_state.get("current_chat_id", ids[0])
                    try:
                        idx = ids.index(current_chat) if current_chat in ids else 0
                    except ValueError:
                        idx = 0
                    picked = st.radio(
                        "Your chats",
                        options=list(range(len(ids))),
                        format_func=lambda i: titles[i],
                        index=idx,
                        key="chat_pick_hdr",
                    )
                    chosen_id = ids[picked]

                    bA, bB, bC = st.columns([0.33, 0.33, 0.34])
                    with bA:
                        if st.button("Open", use_container_width=True, key="btn_open_chat_hdr"):
                            st.session_state.current_chat_id = chosen_id
                            load_chat_messages_into_ui(chosen_id)
                            st.rerun()
                    with bB:
                        new_name = st.text_input("Rename to", value=titles[picked], key="rename_title_hdr")
                        if st.button("Rename", use_container_width=True, key="btn_rename_chat_hdr"):
                            try:
                                rename_chat(chosen_id, new_name.strip() or "(untitled)")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Rename failed: {exc}")
                    with bC:
                        if st.button("Delete", use_container_width=True, key="btn_delete_chat_hdr"):
                            try:
                                delete_chat(chosen_id)
                                if st.session_state.get("current_chat_id") == chosen_id:
                                    st.session_state.current_chat_id = None
                                    st.session_state.messages = []
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Delete failed: {exc}")
                else:
                    st.caption("No chats yet. Create your first chat above.")

    with st.expander("Account", expanded=False):
        if sb is None:
            st.info("Login disabled (no Supabase keys configured).")
        else:
            uid, email = current_user()
            if uid:
                st.success(f"Signed in as {email}")
                colA, colB = st.columns(2)
                with colA:
                    if st.button("Sign out", use_container_width=True):
                        try:
                            sign_out()
                        finally:
                            st.rerun()
                with colB:
                    st.caption("Chats will be saved to your account.")
            else:
                em_in = st.text_input("Email", key="auth_email_hdr")
                pw_in = st.text_input("Password", type="password", key="auth_pw_hdr")
                if st.button("Sign in", use_container_width=True):
                    try:
                        sign_in(em_in, pw_in)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Sign in failed: {exc}")
                with st.expander("Create account"):
                    new_em = st.text_input("Email (new)", key="auth_email_new_hdr")
                    new_pw = st.text_input("Password (new)", type="password", key="auth_pw_new_hdr")
                    if st.button("Create account", use_container_width=True, key="hdr_create_acct"):
                        try:
                            sign_up(new_em, new_pw)
                        except Exception as exc:
                            st.error(f"Sign up failed: {exc}")

    with st.expander("Appearance", expanded=False):
        render_appearance_controls()
    with st.expander("Settings", expanded=False):
        st.session_state.answer_lang = st.selectbox(
            "Answer language",
            ["English", "Arabic"],
            index=["English", "Arabic"].index(st.session_state.answer_lang),
        )
        st.session_state.debug = st.toggle("Debug", value=st.session_state.debug)
        colleges = sorted(list(KNOWN_COLLEGES))
        options = ["All"] + colleges
        try:
            idx = options.index(st.session_state.college_filter)
        except ValueError:
            idx = 0
        st.session_state.college_filter = st.selectbox("College filter", options, index=idx)
        st.session_state.setdefault("mobile_mode", False)
        st.session_state.mobile_mode = st.toggle(
            "Mobile-friendly controls",
            value=st.session_state.mobile_mode,
            help="Use sliders/radios/pills instead of inputs on phones.",
        )

    cols_hdr = st.columns(2)
    with cols_hdr[0]:
        if st.button("Open Schedule Builder" if not st.session_state.show_schedule else "Close Schedule Builder"):
            st.session_state.show_schedule = not st.session_state.show_schedule
            st.rerun()
    with cols_hdr[1]:
        clear_clicked = st.button("Clear chat")

    exists = os.path.exists(CSV_PATH)
    st.caption(f"CSV: {'found' if exists else 'missing'}")

with status_col1:
    st.caption("BETA — AUIBPT (Ryunix Build)")

# ---------------------- Load index / catalog ----------------------
try:
    rows_all, vs, bm25 = build_or_load_index(CSV_PATH, INDEX_DIR, force=False)
    st.caption(f"Loaded {len(rows_all)} courses • Vector index ready")
    college_filter = st.session_state.get("college_filter", "All")
    rows = filter_rows_by_college(rows_all, college_filter)
    retriever = vs.as_retriever(search_kwargs={"k": int(TOP_K)})
except Exception as exc:
    st.error(f"Failed to prepare index or load catalog: {exc}")
    st.stop()

GENERAL_KB_TEXT = load_general_kb("general_academic_kb.json")

# ---------------------- LLM init ----------------------
llm = make_llm(MODEL_NAME, TEMPERATURE, NUM_PREDICT, use_openai=USE_OPENAI)
course_chain = ChatPromptTemplate.from_template(COURSE_PROMPT) | llm
univ_chain = ChatPromptTemplate.from_template(UNIV_PROMPT) | llm

groq_fallback_course = groq_fallback_univ = None
if os.getenv("GROQ_API_KEY"):
    try:
        groq_llm = make_llm("llama-3.1-8b-instant", TEMPERATURE, NUM_PREDICT, use_openai=False)
        groq_fallback_course = ChatPromptTemplate.from_template(COURSE_PROMPT) | groq_llm
        groq_fallback_univ = ChatPromptTemplate.from_template(UNIV_PROMPT) | groq_llm
    except Exception as exc:
        st.warning(f"Groq fallback unavailable: {exc}")

# ---------------------- Chat history render ----------------------
def render_history():
    assistant_avatar = "RP.png" if os.path.exists("RP.png") else None
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar=st.session_state.get("profile_avatar_path")):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar=assistant_avatar or ""):
                st.markdown(message["content"])

render_history()

# ---------------------- Chat input & handling ----------------------
if 'clear_clicked' in locals() and clear_clicked:
    st.session_state.messages = []
    st.rerun()

input_disabled = st.session_state.get("is_generating", False)
if st.session_state.get("is_generating", False):
    st.info("Assistant is generating a response...")
question = st.chat_input("Type your message…", disabled=input_disabled)

if question is not None and not question.strip():
    st.warning("Please type a message first.")

if question is not None and question.strip():
    st.session_state.is_generating = True
    start_ts = time.time()

    with st.chat_message("user", avatar=st.session_state.get("profile_avatar_path")):
        st.markdown(question)
    maybe_capture_name(question)
    st.session_state.messages.append({"role": "user", "content": question})
    if st.session_state.get("current_chat_id"):
        save_message(st.session_state["current_chat_id"], "user", question)

    scopes = infer_scopes(question)
    direct_rows = find_rows_by_code(rows_all, question)
    title_rows = find_rows_by_title(rows, question) if not direct_rows else []
    intent = parse_catalog_intent(question)

    kb = ""
    answer = None
    history_text = build_history_text()
    student_context = student_context_from_taken(rows_all, st.session_state.completed_codes_all)
    answer_lang_str = LANG_OPTIONS.get(st.session_state.answer_lang, "English")

    assistant_avatar = "RP.png" if os.path.exists("RP.png") else None
    with st.chat_message("assistant", avatar=assistant_avatar or None):
        ans_placeholder = st.empty()

        hit = fastpath_course_code(question, rows_all)
        if hit:
            ans_placeholder.markdown(hit)
            answer = hit
        elif is_university_query(question):
            univ_kb_text = univ_kb_blocks_for(question) or "University facts: (none)\nFaculty: (none)"
            answer = ask_llm_stream(
                univ_chain,
                kb="",
                history_text=history_text,
                question=question,
                answer_lang=answer_lang_str,
                student_context=student_context,
                placeholder=ans_placeholder,
                univ_kb=univ_kb_text,
                groq_fallback_chain=groq_fallback_univ,
            )
        elif direct_rows:
            if st.session_state.college_filter != "All":
                filtered = [row for row in direct_rows if (row.get("college", "").upper() == st.session_state.college_filter)]
                if filtered:
                    direct_rows = filtered
            kb = rows_to_kb(direct_rows)
            if st.session_state.completed_codes_all:
                kb += "\n---\n" + rows_to_kb([row for row in rows_all if row["code"].upper() in st.session_state.completed_codes_all])
            answer = ask_llm_stream(
                course_chain,
                kb=kb,
                history_text=history_text,
                question=question,
                answer_lang=answer_lang_str,
                student_context=student_context,
                placeholder=ans_placeholder,
                groq_fallback_chain=groq_fallback_course,
                general_kb=(GENERAL_KB_TEXT if needs_prep_tips(question) else ""),
                web_snippet=(web_enrichment_snippet(question) if needs_prep_tips(question) else ""),
            )
        elif title_rows:
            kb = rows_to_kb(title_rows)
            if st.session_state.completed_codes_all:
                kb += "\n---\n" + rows_to_kb([row for row in rows_all if row["code"].upper() in st.session_state.completed_codes_all])
            answer = ask_llm_stream(
                course_chain,
                kb=kb,
                history_text=history_text,
                question=question,
                answer_lang=answer_lang_str,
                student_context=student_context,
                placeholder=ans_placeholder,
                groq_fallback_chain=groq_fallback_course,
                general_kb=(GENERAL_KB_TEXT if needs_prep_tips(question) else ""),
                web_snippet=(web_enrichment_snippet(question) if needs_prep_tips(question) else ""),
            )
        elif intent:
            if intent["type"] == "count":
                scoped_rows = rows
                depts = intent["scopes"].get("dept", ["all"])
                if "all" not in depts:
                    prefixes = tuple(
                        prefix
                        for scope in depts
                        for prefix in {
                            "cs": ("CSC", "CSE"),
                            "math": ("MAT", "MTH"),
                            "stats": ("STA",),
                            "chem": ("CHE", "CHEM"),
                            "bio": ("BIO", "BIOL"),
                            "pharm": ("PHA",),
                            "dent": ("BDS",),
                        }.get(scope, ())
                    )
                    scoped_rows = [row for row in scoped_rows if row["code"].upper().startswith(prefixes)]
                answer = f"I currently know {len(scoped_rows)} courses from courses.csv."
                ans_placeholder.markdown(answer)
            elif intent["type"] == "list":
                limit = intent.get("limit", 150)
                scoped_rows = rows
                depts = intent["scopes"].get("dept", ["all"])
                if "all" not in depts:
                    prefixes = tuple(
                        prefix
                        for scope in depts
                        for prefix in {
                            "cs": ("CSC", "CSE"),
                            "math": ("MAT", "MTH"),
                            "stats": ("STA",),
                            "chem": ("CHE", "CHEM"),
                            "bio": ("BIO", "BIOL"),
                            "pharm": ("PHA",),
                            "dent": ("BDS",),
                        }.get(scope, ())
                    )
                    scoped_rows = [row for row in scoped_rows if row["code"].upper().startswith(prefixes)]
                lines = [f"{(row.get('college') or 'UNK')} • {row['code']} — {row['title']}" for row in scoped_rows]
                if len(lines) > limit:
                    more = len(lines) - limit
                    lines = lines[:limit] + [f"...and {more} more."]
                answer = "\n".join(lines)
                ans_placeholder.code(answer, language="markdown")
        else:
            expanded = expand_synonyms(question)
            if is_coursey(expanded):
                try:
                    docs = hybrid_retrieve(expanded, retriever, vs, int(TOP_K), bm25=bm25)
                except Exception as exc:
                    log.error(f"hybrid_retrieve error: {exc}")
                    docs = []
                docs = reorder_docs_by_scopes(docs, scopes, st.session_state.college_filter)
                bm25_docs = []
                tokens = (question or "").strip().split()
                use_bm25 = len(tokens) >= 4 and any(len(token) >= 4 for token in tokens)
                if use_bm25 and bm25:
                    try:
                        all_docs = list(vs.docstore._dict.values())
                        scores = bm25.get_scores(tokens)
                        best_ids = sorted(range(len(all_docs)), key=lambda i: -scores[i])[: int(TOP_K)]
                        bm25_docs = [all_docs[i] for i in best_ids]
                    except Exception:
                        bm25_docs = []
                kb = build_kb_from_docs(docs, bm25_docs, top_k=TOP_K, cap=CHUNK_CHAR_CAP)
                if st.session_state.completed_codes_all:
                    kb += ("\n---\n" if kb else "") + rows_to_kb(
                        [row for row in rows_all if row["code"].upper() in st.session_state.completed_codes_all]
                    )
                if kb and kb != "(no relevant context found)":
                    answer = ask_llm_stream(
                        course_chain,
                        kb=kb,
                        history_text=history_text,
                        question=question,
                        answer_lang=answer_lang_str,
                        student_context=student_context,
                        placeholder=ans_placeholder,
                        groq_fallback_chain=groq_fallback_course,
                        general_kb=(GENERAL_KB_TEXT if needs_prep_tips(question) else ""),
                        web_snippet=(web_enrichment_snippet(question) if needs_prep_tips(question) else ""),
                    )
                else:
                    answer = "I don't know from the provided data."
                    ans_placeholder.markdown(answer)
            else:
                answer = ask_llm_stream(
                    course_chain,
                    kb=kb,
                    history_text=history_text,
                    question=question,
                    answer_lang=answer_lang_str,
                    student_context=student_context,
                    placeholder=ans_placeholder,
                    groq_fallback_chain=groq_fallback_course,
                    general_kb=(GENERAL_KB_TEXT if needs_prep_tips(question) else ""),
                    web_snippet=(web_enrichment_snippet(question) if needs_prep_tips(question) else ""),
                )
                if st.session_state.get("user_name") and answer and not answer.lower().startswith(
                    st.session_state["user_name"].lower()
                ):
                    answer = friendly_prefix() + answer
                    ans_placeholder.markdown(answer)

        if st.session_state.debug:
            elapsed = f"{(time.time() - start_ts):.2f}s"
            with st.expander(f"Debug: retrieved KB • {elapsed}"):
                st.code(kb or "(none)")
            if is_university_query(question):
                with st.expander("Debug: UNIV_KB view"):
                    st.code(univ_kb_blocks_for(question), language="markdown")
            st.caption(f"Answered in {elapsed} • Model: {MODEL_NAME} • k={TOP_K} • T={TEMPERATURE}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    if st.session_state.get("current_chat_id"):
        save_message(st.session_state["current_chat_id"], "assistant", answer)
    st.session_state.is_generating = False

# ---------------------- Optional schedule builder panel ----------------------
if st.session_state.get("show_schedule", False):
    render_schedule_builder(rows_all, vs, bm25)
