# ui.py
from __future__ import annotations
import os
import streamlit as st

def apply_theme(primary: str, bg: str, text: str):
    css = f"""
    <style>
    .stApp {{
        background: {bg} !important;
        color: {text} !important;
    }}
    .stButton>button, .stDownloadButton>button {{
        background: {primary} !important;
        color: white !important;
        border: 0 !important;
        border-radius: 8px !important;
    }}
    .stChatMessage .stMarkdown, .stMarkdown p {{
        color: {text} !important;
    }}
    a, .stMarkdown a {{ color: {primary} !important; }}
    .stSidebar {{ color: {text} !important; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_appearance_controls():
    avatar_path = st.session_state.get("profile_avatar_path")

    c1, c2, c3 = st.columns(3)
    with c1:
        _primary = st.color_picker("Accent", st.session_state.get("theme_primary", "#4f46e5"),
                                   key="mini_pick_primary")
    with c2:
        _bg = st.color_picker("Background", st.session_state.get("theme_bg", "#0b1220"),
                              key="mini_pick_bg")
    with c3:
        _textc = st.color_picker("Text", st.session_state.get("theme_text", "#e2e8f0"),
                                 key="mini_pick_text")

    _pp = st.file_uploader("Profile picture", type=["png","jpg","jpeg","gif"], key="mini_profile_pic_up")
    if _pp is not None:
        try:
            _avatar_path = "user_avatar.png"
            with open(_avatar_path, "wb") as f:
                f.write(_pp.read())
            st.session_state.profile_avatar_path = _avatar_path
            avatar_path = _avatar_path
            st.image(avatar_path, width=64, caption="Current profile")
        except Exception as e:
            st.warning(f"Could not save avatar: {e}")
    elif avatar_path and os.path.exists(avatar_path):
        st.image(avatar_path, width=64, caption="Current profile")

    if (_primary != st.session_state.get("theme_primary")) or \
       (_bg != st.session_state.get("theme_bg")) or \
       (_textc != st.session_state.get("theme_text")):
        st.session_state.theme_primary = _primary
        st.session_state.theme_bg = _bg
        st.session_state.theme_text = _textc
        apply_theme(st.session_state.theme_primary, st.session_state.theme_bg, st.session_state.theme_text)
