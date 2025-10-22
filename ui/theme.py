# Theme and appearance functionality for AUIBPT

import os
import time
import streamlit as st

def show_splash():
    """Show splash screen."""
    st.markdown("""
        <style>
        @keyframes fadeInOut { 0%{opacity:0} 10%{opacity:1} 80%{opacity:1} 100%{opacity:0} }
        .splash-container{
            position:fixed; z-index:9999; inset:0; display:flex; align-items:center; justify-content:center;
            background: radial-gradient(circle at 50% 50%, #0b1220 0%, #050812 60%, #000 100%);
            animation: fadeInOut 1.6s ease-in-out forwards;
        }
        .splash-title{ font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial; font-size:40px; letter-spacing:2px; color:#e2e8f0; text-shadow:0 2px 12px rgba(255,255,255,0.25); }
        </style>
        <div class="splash-container"><div class="splash-title">Ryunix Productions</div></div>
    """, unsafe_allow_html=True)
    time.sleep(1.65)
    st.session_state._splash_shown = True
    st.rerun()

def apply_theme(primary: str, bg: str, text: str):
    """Apply custom theme."""
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

def _render_appearance_controls():
    """Render appearance controls."""
    avatar_path = st.session_state.get("profile_avatar_path")

    c1, c2, c3 = st.columns(3)
    with c1:
        _primary = st.color_picker("Accent", st.session_state.get("theme_primary", "#4f46e5"), key="mini_pick_primary")
    with c2:
        _bg = st.color_picker("Background", st.session_state.get("theme_bg", "#0b1220"), key="mini_pick_bg")
    with c3:
        _textc = st.color_picker("Text", st.session_state.get("theme_text", "#e2e8f0"), key="mini_pick_text")

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

    if (_primary != st.session_state.get("theme_primary")) or (_bg != st.session_state.get("theme_bg")) or (_textc != st.session_state.get("theme_text")):
        st.session_state.theme_primary = _primary
        st.session_state.theme_bg = _bg
        st.session_state.theme_text = _textc
        apply_theme(st.session_state.theme_primary, st.session_state.theme_bg, st.session_state.theme_text)

def setup_page_config():
    """Setup page configuration."""
    page_icon = "RP.png" if os.path.exists("RP.png") else None
    st.set_page_config(
        page_title="AUIBPT",
        page_icon=page_icon,
        layout="wide",
        menu_items={
            "About": "AUIBPT v1.5 — Course & Schedule assistant for AUIB.",
            "Get Help": "mailto:ali.1241375@auib.edu.iq",
            "Report a bug": "mailto:ali.1241375@auib.edu.iq",
        }
    )

def setup_global_styles():
    """Setup global CSS styles."""
    # Hide sidebar; widen main
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none !important; }
        .main .block-container { max-width: 1200px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Global polish CSS
    st.markdown(
        f"""
        <style>
        .stApp {{
            font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Ubuntu, Arial, "Noto Sans", sans-serif;
            line-height: 1.55;
        }}
        .main .block-container {{
            max-width: 1100px;
            padding-top: 1.25rem;
        }}
        [data-testid="stChatMessage"] {{
            padding: 0.6rem 0.75rem;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.02);
            margin-bottom: 0.35rem;
        }}
        [data-testid="stChatMessage"] .stMarkdown p {{ margin-bottom: 0.35rem; }}
        .stButton>button, .stDownloadButton>button {{
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.15);
            transition: transform .02s ease-in;
        }}
        .stButton>button:active {{ transform: translateY(1px); }}
        pre, code {{ font-size: 0.93rem; }}
        hr {{ border-color: rgba(255,255,255,0.12) !important; }}
        </style>
        """,
        unsafe_allow_html=True
    )

def setup_footer():
    """Setup footer."""
    st.markdown(
        """
        <style>
        /* No extra gap below the chat input */
        div[data-testid="stChatInput"] { margin-bottom: 0 !important; }

        /* No extra bottom padding on the page */
        .main .block-container { padding-bottom: 0 !important; }

        /* Hide the custom footer if it still exists in the DOM */
        .custom-chat-footer { display: none !important; }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            text-align:center;
            font-size:13px;
            color:rgba(226,232,240,0.85);
            margin-top:20px;
            padding-top:6px;
            border-top:1px solid rgba(255,255,255,0.1);
        "></div>
        """,
        unsafe_allow_html=True
    )
