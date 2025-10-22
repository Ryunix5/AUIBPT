# Chat functionality for AUIBPT

import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

from ..core.llm import make_llm, ask_llm_stream
from ..utils.helpers import (
    build_history_text, maybe_capture_name, friendly_prefix,
    find_rows_by_code, find_rows_by_title, infer_scopes, parse_catalog_intent,
    rows_to_kb, is_coursey, expand_synonyms, reorder_docs_by_scopes
)
from ..utils.university_kb import is_university_query, univ_kb_blocks_for
from ..core.retrieval import hybrid_retrieve, prepare_kb_from_docs

# Prompts
COURSE_PROMPT = """
You are AUIBPT, a sharp and friendly university course assistant. Your vibe is upbeat, helpful, and concise.
Use ONLY the provided course knowledge base (kb). If an item is missing in kb, write "Unknown"—do not invent.
Keep it student-friendly and practical.

When relevant, factor in the student's completed courses from the section "student_profile" (e.g., prerequisites already satisfied, avoid recommending repeats).

Format your final answer exactly as:
- College: <college tag or 'Unknown'>
- Summary: <one lively sentence, ~15–25 words>
- Key topics: <comma-separated>
- Prerequisites: <text or 'Unknown'>
- Credits: <text or 'Unknown'>
- Source: <course code(s)>

Respond in {answer_lang}. Put ONLY your final answer inside <final>...</final>.

student_profile:
{student_context}

kb:
{kb}

history:
{history}

question:
{question}
"""

CHAT_PROMPT = """
You are AUIBPT, a helpful, upbeat campus buddy. Be concise (≤3 sentences), conversational, and encouraging.
Use plain language, optionally offer a short follow-up nudge.

Personalize advice using the student's completed courses from "student_profile" (e.g., what they could take next or what they've unlocked).

Respond in {answer_lang}. Put ONLY your final answer inside <final>...</final>.

student_profile:
{student_context}

history:
{history}

question:
{question}
"""

UNIV_PROMPT = """
You are AUIBPT, a friendly, factual assistant for AUIB. Use ONLY the supplied 'univ_kb' block.
If something is unknown, say "Unknown" briefly. Be concise, useful, and student-facing.

Personalize if relevant using 'student_profile' (e.g., suggest how a professor's area aligns with the student's path).

Format the final answer EXACTLY as:
Topic: <short topic>
Highlights: <1–2 sentences>
Details:
- <bullet 1>
- <bullet 2>
- <bullet 3>
Source: AUIB institutional KB

Respond in {answer_lang}. Put ONLY your final answer inside <final>...</final>.

student_profile:
{student_context}

univ_kb:
{univ_kb}

history:
{history}

question:
{question}
"""

def render_history():
    """Render chat history."""
    assistant_avatar = "RP.png" if os.path.exists("RP.png") else None
    for m in st.session_state.messages:
        if m["role"] == "user":
            with st.chat_message("user", avatar=st.session_state.get("profile_avatar_path")):
                st.markdown(m["content"])
        else:
            with st.chat_message("assistant", avatar=assistant_avatar or ""):
                st.markdown(m["content"])

def _student_profile_for_prompt(rows_all, completed_codes_all) -> str:
    """Generate student profile context for prompts."""
    from ..utils.helpers import student_context_from_taken
    return student_context_from_taken(rows_all, completed_codes_all)

def handle_chat_input(q, rows_all, rows, college_filter, retriever, vs, bm25, 
                     course_chain, chat_chain, univ_chain, 
                     groq_fallback_course, groq_fallback_chat, groq_fallback_univ,
                     answer_lang, debug, LANG_OPTIONS):
    """Handle chat input and generate response."""
    import time
    
    st.session_state.is_generating = True
    start_ts = time.time()

    with st.chat_message("user", avatar=st.session_state.get("profile_avatar_path")):
        st.markdown(q)
    
    captured_name = maybe_capture_name(q)
    if captured_name:
        st.session_state.user_name = captured_name
    
    st.session_state.messages.append({"role": "user", "content": q})

    scopes = infer_scopes(q)
    direct_rows = find_rows_by_code(rows_all, q)
    title_rows = find_rows_by_title(rows, q) if not direct_rows else []
    intent = parse_catalog_intent(q)

    kb = ""
    ans = None
    history_text = build_history_text(st.session_state.messages)
    student_context = _student_profile_for_prompt(rows_all, st.session_state.completed_codes_all)
    answer_lang_str = LANG_OPTIONS.get(answer_lang, "English")

    assistant_avatar = "RP.png" if os.path.exists("RP.png") else None
    with st.chat_message("assistant", avatar=assistant_avatar or None):
        ans_placeholder = st.empty()

        if is_university_query(q):
            univ_kb_text = univ_kb_blocks_for(q) or "University facts: (none)\nFaculty: (none)"
            ans = ask_llm_stream(
                univ_chain,
                kb="",
                history_text=history_text,
                q=q,
                answer_lang=answer_lang_str,
                student_context=student_context,
                placeholder=ans_placeholder,
                univ_kb=univ_kb_text,
                groq_fallback_chain=groq_fallback_univ,
            )

        elif direct_rows:
            if college_filter != "All":
                filtered = [r for r in direct_rows if (r.get("college","").upper() == college_filter)]
                if filtered:
                    direct_rows = filtered
            kb = rows_to_kb(direct_rows)
            if st.session_state.completed_codes_all:
                kb += "\n---\n" + rows_to_kb([r for r in rows_all if r["code"].upper() in st.session_state.completed_codes_all])
            ans = ask_llm_stream(
                course_chain, kb, history_text, q, answer_lang_str, student_context,
                ans_placeholder, groq_fallback_chain=groq_fallback_course
            )

        elif title_rows:
            kb = rows_to_kb(title_rows)
            if st.session_state.completed_codes_all:
                kb += "\n---\n" + rows_to_kb([r for r in rows_all if r["code"].upper() in st.session_state.completed_codes_all])
            ans = ask_llm_stream(
                course_chain, kb, history_text, q, answer_lang_str, student_context,
                ans_placeholder, groq_fallback_chain=groq_fallback_course
            )

        elif intent:
            if intent["type"] == "count":
                scoped_rows = rows
                depts = intent["scopes"].get("dept", ["all"])
                if "all" not in depts:
                    from ..utils.constants import DEPT_PREFIXES
                    prefixes = tuple(p for s in depts for p in DEPT_PREFIXES.get(s, ()))
                    scoped_rows = [r for r in scoped_rows if r["code"].upper().startswith(prefixes)]
                ans_text = f"I currently know {len(scoped_rows)} courses from courses.csv."
                ans_placeholder.markdown(ans_text); ans = ans_text
            elif intent["type"] == "list":
                limit = intent.get("limit", 150)
                scoped_rows = rows
                depts = intent["scopes"].get("dept", ["all"])
                if "all" not in depts:
                    from ..utils.constants import DEPT_PREFIXES
                    prefixes = tuple(p for s in depts for p in DEPT_PREFIXES.get(s, ()))
                    scoped_rows = [r for r in scoped_rows if r["code"].upper().startswith(prefixes)]
                lines = [f"{(r.get('college') or 'UNK')} • {r['code']} — {r['title']}" for r in scoped_rows]
                if len(lines) > limit:
                    more = len(lines) - limit
                    lines = lines[:limit] + [f"...and {more} more."]
                ans_text = "\n".join(lines)
                ans_placeholder.code(ans_text, language="markdown"); ans = ans_text

        else:
            qx = expand_synonyms(q)
            if is_coursey(qx):
                try:
                    docs = hybrid_retrieve(qx, retriever, vs, 5, bm25=bm25)  # Using TOP_K from settings
                except Exception as e:
                    import logging
                    log = logging.getLogger("app")
                    log.error(f"hybrid_retrieve error: {e}"); docs = []
                docs = reorder_docs_by_scopes(docs, scopes, college_filter)
                kb = prepare_kb_from_docs(docs)
                if st.session_state.completed_codes_all:
                    kb += ("\n---\n" if kb else "") + rows_to_kb([r for r in rows_all if r["code"].upper() in st.session_state.completed_codes_all])
                if kb and kb != "(no relevant context found)":
                    ans = ask_llm_stream(course_chain, kb, history_text, q, answer_lang_str, student_context, ans_placeholder)
                else:
                    ans_placeholder.markdown("I don't know from the provided data."); ans = "I don't know from the provided data."
            else:
                ans = ask_llm_stream(
                    chat_chain, "", history_text, q, answer_lang_str, student_context,
                    ans_placeholder, groq_fallback_chain=groq_fallback_chat
                )
                if st.session_state.get("user_name") and ans and not ans.lower().startswith(st.session_state["user_name"].lower()):
                    prefixed = friendly_prefix(st.session_state.get("user_name")) + ans
                    ans_placeholder.markdown(prefixed); ans = prefixed

        if debug:
            elapsed = f"{(time.time() - start_ts):.2f}s"
            with st.expander(f"Debug: retrieved KB • {elapsed}"):
                st.code(kb or "(none)")
            if is_university_query(q):
                with st.expander("Debug: UNIV_KB view"):
                    st.code(univ_kb_blocks_for(q), language="markdown")
            st.caption(f"Answered in {elapsed}")

    st.session_state.messages.append({"role": "assistant", "content": ans})
    st.session_state.is_generating = False
