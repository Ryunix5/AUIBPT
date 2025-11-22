# app.py — AUIBPT (Fixed build: env-first auth, inline login, no-rerun chat)
from __future__ import annotations
import os, re, csv, io, json, time, string, logging
from typing import List, Dict, Tuple, Optional, Set

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

# Local modules
from univkb import UNIV_KB, is_university_query, univ_kb_blocks_for
from ui import apply_theme, render_appearance_controls
from data_io import build_or_load_index
from ui import apply_theme, render_appearance_controls

# ---------------------- Settings (safe fallbacks) ----------------------
try:
    from settings import MODEL_NAME, CSV_PATH, INDEX_DIR, TOP_K, TEMPERATURE, NUM_PREDICT, USE_OPENAI
except Exception:
    MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
    CSV_PATH     = os.getenv("CSV_PATH", "course.csv")
    INDEX_DIR    = os.getenv("INDEX_DIR", "faiss_index")
    TOP_K        = int(os.getenv("TOP_K", "3"))
    TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.2"))
    NUM_PREDICT  = int(os.getenv("NUM_PREDICT", "512"))
    USE_OPENAI   = os.getenv("USE_OPENAI", "true").strip().lower() in {"1","true","yes","on"}

try:
    from settings import USE_GROQ_ONLY as _USE_GROQ_ONLY
except Exception:
    _USE_GROQ_ONLY = None

_USE_GROQ_ONLY_ENV = os.getenv("USE_GROQ_ONLY","").strip().lower()
if _USE_GROQ_ONLY_ENV in {"1","true","yes","on"}:
    USE_GROQ_ONLY = True
elif _USE_GROQ_ONLY_ENV in {"0","false","no","off"}:
    USE_GROQ_ONLY = False
elif isinstance(_USE_GROQ_ONLY, bool):
    USE_GROQ_ONLY = _USE_GROQ_ONLY
else:
    USE_GROQ_ONLY = False

# ---------------------- Small helpers ----------------------
def _opt_index(options, value):
    try:
        return options.index(value)
    except Exception:
        return 0

def ui_select(label, options, *, default=None, key=None, help=None):
    """On mobile: radio (touch). On desktop: selectbox."""
    if st.session_state.get("mobile_mode", False):
        idx = _opt_index(options, default)
        return st.radio(label, options, index=idx, horizontal=True, key=key, help=help)
    else:
        idx = _opt_index(options, default)
        return st.selectbox(label, options, index=idx, key=key, help=help)

def ui_multi(label, options, *, default=None, key=None, help=None):
    """On mobile: pills or checkboxes; on desktop: multiselect."""
    default = default or []
    if st.session_state.get("mobile_mode", False):
        # Prefer pills if your Streamlit version supports it
        if hasattr(st, "pills"):
            return st.pills(label, options, default=default, selection_mode="multi", key=key, help=help)
        # Fallback: checkbox grid (no keyboard)
        st.caption(label)
        cols = st.columns(3)
        picked = []
        for i, opt in enumerate(options):
            if cols[i % 3].checkbox(opt, value=(opt in default), key=f"{key or label}_{i}"):
                picked.append(opt)
        return picked
    else:
        return st.multiselect(label, options, default=default, key=key, help=help)

def ui_int(label, *, min_value, max_value, value, step=1, key=None, help=None):
    """On mobile: select_slider (no keyboard). On desktop: number_input."""
    if st.session_state.get("mobile_mode", False):
        return st.select_slider(
            label,
            options=list(range(min_value, max_value + 1, step)),
            value=value,
            key=key,
            help=help,
        )
    else:
        return st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            value=value,
            step=step,
            key=key,
            help=help,
        )

def _to_str(x) -> str:
    try:
        from langchain_core.messages import AIMessage
        if isinstance(x, AIMessage):
            return x.content or ""
    except Exception:
        pass
    return x if isinstance(x, str) else str(x)

def _clean_output(text: str) -> str:
    """
    Keep the assistant's Markdown (lists, newlines) intact while stripping think/final tags.
    """
    if not text:
        return "I don't know from the provided data."

    #  Extract last <final>...</final> block if present
    finals = re.findall(r"<final>(.*?)</final>", text, flags=re.DOTALL | re.IGNORECASE)
    if finals:
        text = next((blk.strip() for blk in reversed(finals) if blk.strip()), finals[-1].strip())
    else:
        m_open = re.search(r"<final>(.*)$", text, flags=re.DOTALL | re.IGNORECASE)
        if m_open:
            text = m_open.group(1).strip()

    
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?final\b[^>]*>", "", text, flags=re.IGNORECASE)

    
    # - convert CRLF to LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # - trim trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # - collapse 3+ blank lines to max 2 (keeps paragraphs & lists readable)
    text = re.sub(r"\n{3,}", "\n\n", text)


    # 4) Remove overly aggressive scrub that could eat content (e.g., lines starting "rules:")
    # (We previously stripped anything after 'rules:'; that's gone now.)

    text = text.strip()
    return text or "I don't know from the provided data."


def _get_secret(key: str) -> Optional[str]:
    v = os.getenv(key)
    if v:
        return v
    try:
        return st.secrets[key]
    except Exception:
        return None
# ---- General KB (local JSON) helpers ----
def load_general_kb(path="general_academic_kb.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        lines = []
        for it in items:
            topic = (it.get("topic") or "").strip()
            content = (it.get("content") or "").strip()
            if topic and content:
                lines.append(f"- {topic}: {content}")
        return "\n".join(lines[:20])
    except Exception:
        return ""

def needs_prep_tips(q: str) -> bool:
    ql = (q or "").lower()
    hot = [
        "prepare", "preparation", "study tips", "how to study", "how do i prepare",
        "how can i prepare", "succeed", "revision", "exam", "practice", "resources",
        "what to expect", "how to get ready", "how to revise"
    ]
    return any(k in ql for k in hot)


# ---------------------- Auth / Supabase (env-first; safe fallback) ----------------------
from typing import Optional, Any, TYPE_CHECKING

try:
    from supabase import create_client  # runtime factory
except Exception:
    create_client = None  # type: ignore

# For type checking only – don't import at runtime if not installed
if TYPE_CHECKING:
    from supabase import Client as SupabaseClient
else:
    SupabaseClient = Any  # fallback for runtime when lib may be missing

@st.cache_resource(show_spinner=False)
def get_supabase() -> Optional[SupabaseClient]:
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_ANON_KEY")
    if not url or not key or (create_client is None):
        st.info("Login disabled (no Supabase keys or SDK).")
        return None
    try:
        return create_client(url, key)  # Works with supabase-py >=2
    except Exception as e:
        st.warning(f"Could not init Supabase: {e}. Running without auth.")
        return None

sb: Optional[SupabaseClient] = get_supabase()
# ---- Supabase/PostgREST compatibility helpers ----
def _pg_tbl(name: str):
    """Return a query builder for a table, compatible across supabase/postgrest versions."""
    if sb is None:
        return None
    # Preferred (supabase-py v2)
    if hasattr(sb, "table"):
        try:
            return sb.table(name)
        except Exception:
            pass
    # Fallbacks
    if hasattr(sb, "from_"):
        return sb.from_(name)  # some builds expose from_ at root
    if hasattr(sb, "postgrest") and hasattr(sb.postgrest, "from_"):
        return sb.postgrest.from_(name)
    raise RuntimeError("No compatible supabase/postgrest table accessor found.")

def _pg_select(qb, cols="*"):
    """Do a select in a version-tolerant way."""
    # Newer postgrest builders
    if hasattr(qb, "select"):
        return qb.select(cols)
    # Old builders sometimes expose 'execute' only (rare). Try to use 'select' if present, else fail.
    raise AttributeError("This PostgREST builder does not support .select(); please upgrade 'supabase' and 'postgrest' packages.")

def _pg_order(qb, col: str, desc: bool = False):
    if hasattr(qb, "order"):
        return qb.order(col, desc=desc)
    return qb  # ignore if not supported

def _pg_eq(qb, col: str, val):
    if hasattr(qb, "eq"):
        return qb.eq(col, val)
    raise AttributeError("This PostgREST builder does not support .eq(); please upgrade packages.")

def _pg_insert(qb, row: dict):
    if hasattr(qb, "insert"):
        return qb.insert(row)
    raise AttributeError("This PostgREST builder does not support .insert(); please upgrade packages.")

def _pg_update(qb, row: dict):
    if hasattr(qb, "update"):
        return qb.update(row)
    raise AttributeError("This PostGREST builder does not support .update(); please upgrade packages.")

def _pg_delete(qb):
    if hasattr(qb, "delete"):
        return qb.delete()
    raise AttributeError("This PostGREST builder does not support .delete(); please upgrade packages.")

def _pg_single(qb):
    # Some versions expose .single() for a single row select/insert returning row
    return qb.single() if hasattr(qb, "single") else qb

def _pg_exec(qb):
    """Execute and normalize the response into .data."""
    res = qb.execute() if hasattr(qb, "execute") else qb
    # supabase-py v2 returns an object with .data; earlier may return dict
    if hasattr(res, "data"):
        return res.data
    if isinstance(res, dict) and "data" in res:
        return res["data"]
    return res  # best effort


# ---------- Autosave utilities ----------
def _auth_user():
    """Return (uid, email) if signed in, else (None, None)."""
    user = st.session_state.get("_auth_user")
    if isinstance(user, dict) and user.get("id"):
        return user["id"], user.get("email")
    try:
        if sb and hasattr(sb, "auth") and hasattr(sb.auth, "get_user"):
            resp = sb.auth.get_user()
            u = getattr(resp, "user", None)
            if u:
                uid = getattr(u, "id", None)
                em  = getattr(u, "email", None)
                if uid:
                    st.session_state["_auth_user"] = {"id": uid, "email": em}
                    return uid, em
    except Exception:
        pass
    return None, None

def queue_autosave():
    st.session_state["_autosave_needed"] = True

def run_autosave_if_needed():
    if not st.session_state.get("_autosave_needed"):
        return
    uid, em = _auth_user()
    if uid:
        try:
            save_profile_settings(uid)   # <-- your existing saver
            st.session_state["_autosave_needed"] = False
        except Exception:
            # don't crash the UI; just keep the flag for next run
            pass
# [ANCHOR: AUTOSAVE_UTILS]



def sign_in(email: str, password: str):
    if sb is None:
        return
    sb.auth.sign_out()
    res = sb.auth.sign_in_with_password({"email": email, "password": password})
    user = getattr(res, "user", None)
    uid  = getattr(user, "id", None)
    em   = getattr(user, "email", None)
    st.session_state._auth_user = {"id": uid, "email": em}

    try:
        sb.table("profiles").upsert({"id": uid, "email": em}).execute()
    except Exception:
        pass

    try:
        load_profile_settings(uid)
    except Exception:
        pass

def sign_up(email: str, password: str):
    if sb is None: return
    res = sb.auth.sign_up({"email": email, "password": password})
    st.success("Check your email to verify your account, then sign in.")

def sign_out():
    if sb is None:
        return
    sb.auth.sign_out()
    # reset to a clean state
    st.session_state._auth_user = None
    st.session_state.pop("current_chat_id", None)
    st.session_state.pop("messages", None)
def _auth_enabled() -> bool:
    return sb is not None

# ---------- User settings persistence ----------

SETTINGS_KEYS = [
    # appearance & UI
    "theme_primary", "theme_bg", "theme_text", "answer_lang",
    "college_filter", "mobile_mode", "profile_avatar_path",
    # schedule / study prefs (optional but handy)
    "schedule_target_credits", "schedule_major_key", "completed_codes_all",
]

def _get_profile(uid: str):
    """Return the first row from profiles for uid (or {})."""
    if sb is None:
        return {}
    qb = _pg_tbl("profiles")
    rows = _pg_exec(_pg_eq(_pg_select(qb, "*"), "id", uid))
    rows = rows if isinstance(rows, list) else [rows]
    return rows[0] if rows else {}

def _apply_settings_to_session(settings: dict):
    """Map saved settings JSON into st.session_state safely."""
    if not isinstance(settings, dict):
        return
    for k in SETTINGS_KEYS:
        if k in settings:
            # special-case: completed_codes_all is a set in session
            if k == "completed_codes_all":
                try:
                    st.session_state[k] = set(settings[k]) if settings[k] is not None else set()
                except Exception:
                    st.session_state[k] = set()
            else:
                st.session_state[k] = settings[k]
    # re-apply theme immediately if present
    if "theme_primary" in settings or "theme_bg" in settings or "theme_text" in settings:
        apply_theme(
            st.session_state.get("theme_primary", "#4d1212"),
            st.session_state.get("theme_bg", "#000000"),
            st.session_state.get("theme_text", "#e2e8f0"),
        )

def load_profile_settings(uid: str) -> bool:
    """Fetch settings from Supabase and apply to session. Returns True if applied."""
    if sb is None:
        return False
    try:
        row = _get_profile(uid)
        settings = row.get("settings") or {}
        _apply_settings_to_session(settings)
        return bool(settings)
    except Exception as e:
        st.warning(f"Could not load saved settings: {e}")
        return False

def save_profile_settings(uid: str) -> bool:
    """Collect current session prefs and persist to Supabase."""
    if sb is None:
        return False
    try:
        payload = {}
        for k in SETTINGS_KEYS:
            val = st.session_state.get(k)
            # JSON-safe for sets
            if isinstance(val, set):
                val = sorted(list(val))
            payload[k] = val
        qb = _pg_tbl("profiles")
        _pg_exec(_pg_update(qb, {"settings": payload, "updated_at": "now()"}).eq("id", uid))
        return True
    except Exception as e:
        st.warning(f"Could not save settings: {e}")
        return False

# DAL (safe when auth disabled)
def list_chats(uid: str):
    if not _auth_enabled(): return []
    qb = _pg_tbl("chats")
    data = _pg_exec(_pg_order(_pg_eq(_pg_select(qb, "*"), "user_id", uid), "created_at", desc=True))
    return _as_rows(data)

def create_chat(uid: str, title: str = "New chat") -> Optional[str]:
    if not _auth_enabled(): return None
    qb = _pg_tbl("chats")
    ins = _pg_insert(qb, {"user_id": uid, "title": title})
    row = _first_row(_pg_exec(_pg_single(ins)))
    return row.get("id")

def rename_chat(chat_id: str, title: str):
    if not _auth_enabled(): return
    qb = _pg_tbl("chats")
    _pg_exec(_pg_eq(_pg_update(qb, {"title": title}), "id", chat_id))

def delete_chat(chat_id: str):
    if not _auth_enabled(): return
    qb = _pg_tbl("chats")
    _pg_exec(_pg_eq(_pg_delete(qb), "id", chat_id))

def load_messages(chat_id: str):
    if not _auth_enabled(): return []
    qb = _pg_tbl("messages")
    rows = _pg_exec(_pg_order(_pg_eq(_pg_select(qb, "*"), "chat_id", chat_id), "created_at", desc=False))
    return [{"role": r.get("role"), "content": r.get("content")} for r in _as_rows(rows)]

def save_message(chat_id: str, role: str, content: str):
    if not _auth_enabled(): return
    qb = _pg_tbl("messages")
    _pg_exec(_pg_insert(qb, {"chat_id": chat_id, "role": role, "content": content}))
# --- Chat helpers (compat layer for single-file app) ---

def current_user():
    """Return (uid, email) if signed in, else (None, None)."""
    # Use cached user first
    u = st.session_state.get("_auth_user")
    if isinstance(u, dict) and u.get("id"):
        return u.get("id"), u.get("email")
    # Ask Supabase
    try:
        if sb and hasattr(sb, "auth") and hasattr(sb.auth, "get_user"):
            resp = sb.auth.get_user()
            supa_user = getattr(resp, "user", None)
            if supa_user:
                uid = getattr(supa_user, "id", None)
                em  = getattr(supa_user, "email", None)
                if uid:
                    st.session_state["_auth_user"] = {"id": uid, "email": em}
                    return uid, em
    except Exception:
        pass
    return None, None


def ensure_current_chat():
    """Ensure st.session_state.current_chat_id exists for the signed-in user."""
    if not sb:
        return
    uid, _ = current_user()
    if not uid:
        st.session_state.pop("current_chat_id", None)
        return
    if st.session_state.get("current_chat_id"):
        return
    try:
        q = sb.table("chats").select("id").eq("user_id", uid).order("created_at", desc=True).limit(1).execute()
        data = getattr(q, "data", q) or []
        if isinstance(data, list) and data:
            st.session_state["current_chat_id"] = data[0]["id"]
            return
        # Create a first chat if none exist
        ins = sb.table("chats").insert({"user_id": uid, "title": "New chat"}).execute()
        d = getattr(ins, "data", ins) or []
        if isinstance(d, list) and d:
            st.session_state["current_chat_id"] = d[0]["id"]
    except Exception as e:
        st.warning(f"Could not initialize chats: {e}")


def load_chat_messages_into_ui(chat_id: str):
    """Load chat messages into st.session_state.messages."""
    if not sb or not chat_id:
        return
    try:
        res = sb.table("messages").select("*").eq("chat_id", chat_id).order("created_at", desc=False).execute()
        data = getattr(res, "data", res) or []
        st.session_state["messages"] = [
            {"role": row.get("role"), "content": row.get("content", "")} for row in data
        ]
    except Exception as e:
        st.warning(f"Could not load messages: {e}")
def render_chats_ui(prefix: str = "hdr_chats"):
    import streamlit as st
    # Sign-in guard
    if sb is None:
        st.info("Sign in to save and manage chats.")
        return

    # current_user(), ensure_current_chat(), list_chats(), create_chat(),
    # rename_chat(), delete_chat(), load_chat_messages_into_ui()
    # must exist in this file (your compat layer) or be imported.

    uid, email = current_user()
    if not uid:
        st.info("Please sign in to view your chats.")
        return

    ensure_current_chat()

    try:
        chat_rows = list_chats(uid) or []
    except Exception as exc:
        chat_rows = []
        st.warning(f"Could not fetch chats: {exc}")

    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        new_title = st.text_input(
            "New chat title",
            value="New chat",
            key=f"{prefix}_new_title",
        )
    with c2:
        if st.button("Create chat", use_container_width=True, key=f"{prefix}_create"):
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
            key=f"{prefix}_pick",
        )
        chosen_id = ids[picked]

        bA, bB, bC = st.columns([0.33, 0.33, 0.34])
        with bA:
            if st.button("Open", use_container_width=True, key=f"{prefix}_open"):
                st.session_state.current_chat_id = chosen_id
                load_chat_messages_into_ui(chosen_id)
                st.rerun()
        with bB:
            new_name = st.text_input(
                "Rename to",
                value=titles[picked],
                key=f"{prefix}_rename_input",
            )
            if st.button("Rename", use_container_width=True, key=f"{prefix}_rename"):
                try:
                    rename_chat(chosen_id, new_name.strip() or "(untitled)")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Rename failed: {exc}")
        with bC:
            if st.button("Delete", use_container_width=True, key=f"{prefix}_delete"):
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

# ---- Chat session helpers (ensure a current chat and load its history) ----
def _ensure_current_chat():
    """If signed in, make sure there's a current chat id in session; create one if none."""
    if not _auth_enabled():
        return
    uid, _ = _auth_user()
    if not uid:
        st.session_state.pop("current_chat_id", None)
        return
    if st.session_state.get("current_chat_id"):
        return
    try:
        rows = list_chats(uid) or []
        if rows:
            st.session_state.current_chat_id = rows[0]["id"]
        else:
            new_id = create_chat(uid, title="New chat")
            st.session_state.current_chat_id = new_id
    except Exception as e:
        st.warning(f"Could not initialize chats: {e}")

def _load_chat_messages_into_ui(chat_id: str):
    """Replace st.session_state.messages with what's in DB for this chat."""
    if not _auth_enabled() or not chat_id:
        return
    try:
        msgs = load_messages(chat_id) or []
        # Normalize into our UI format
        st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in msgs]
    except Exception as e:
        st.warning(f"Could not load messages: {e}")

def _as_rows(obj):
    """Normalize Supabase result into a list[dict]."""
    if obj is None:
        return []
    # supabase v2: execute() -> object with .data (already handled in _pg_exec)
    # but upstream callers may pass raw lists/dicts too
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    # pydantic-ish
    if hasattr(obj, "model_dump"):
        dumped = obj.model_dump()
        return [dumped] if isinstance(dumped, dict) else list(dumped or [])
    return [obj]

def _first_row(obj) -> dict:
    """Return the first row as dict, or {} if absent."""
    rows = _as_rows(obj)
    return rows[0] if rows else {}
# ---- Optional web enrichment (no API key needed) ----
def web_enrichment_snippet(q: str, max_chars=800) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            # Focus on prep/overview style results
            results = list(ddgs.text(q, region="wt-wt", safesearch="moderate", max_results=3))
        texts = []
        for r in results:
            # r has keys like 'title','href','body'
            body = (r.get("body") or "").strip()
            if body:
                texts.append(body)
        snippet = " ".join(texts)
        snippet = re.sub(r"\s+", " ", snippet).strip()
        return snippet[:max_chars]
    except Exception:
        return ""

# ---------------------- UI config ----------------------
page_icon = "RP.png" if os.path.exists("RP.png") else None
st.set_page_config(
    page_title="AUIBPT",
    page_icon=page_icon,
    layout="wide",
    menu_items={
        "About": "AUIBPT — Course & Schedule assistant for AUIB.",
        "Get Help": "mailto:ali.1241375@auib.edu.iq",
        "Report a bug": "mailto:ali.1241375@auib.edu.iq",
    }
)

# Hide sidebar; tighten layout
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    .main .block-container { max-width: 1100px; padding-top: 1rem; padding-bottom: 0 !important; }
    [data-testid="stChatMessage"] { padding: .6rem .75rem; border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.02); margin-bottom: .35rem; }
    [data-testid="stChatMessage"] .stMarkdown p { margin-bottom: .35rem; }
    .stButton>button, .stDownloadButton>button { border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,.15); }
    pre, code { font-size: .93rem; }
    div[data-testid="stChatInput"] { margin-bottom: 0 !important; }
    /* Make top-right expanders look like pill buttons and align nicely */
    .block-container .streamlit-expanderHeader {
  font-size: 0.90rem !important;
    }
    div[data-testid="stExpander"] > details > summary {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 999px;
  padding: 6px 12px;
  display: inline-block;
  width: auto !important;
  margin-left: 6px;
    }
    div[data-testid="stExpander"] > details[open] > summary {
  background: rgba(255,255,255,0.08);
    </style>
    """,
    unsafe_allow_html=True
)

# Theme (defaults if first run)
if "theme_primary" not in st.session_state: st.session_state.theme_primary = "#4d1212"
if "theme_bg" not in st.session_state:      st.session_state.theme_bg = "#000000"
if "theme_text" not in st.session_state:    st.session_state.theme_text = "#e2e8f0"
apply_theme(st.session_state.theme_primary, st.session_state.theme_bg, st.session_state.theme_text)

# ---------------------- Logging ----------------------
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("app")

# ---------------------- Session defaults ----------------------
for k, v in {
    "messages": [], "user_name": None, "profile_avatar_path": None,
    "schedule_slots": [], "schedule_planned_credits": 0, "schedule_target_credits": 15,
    "schedule_major_key": "CS", "completed_codes_all": set(), "swap_history": [],
    "show_schedule": False, "is_generating": False, "answer_lang": "English", "debug": False,
    "college_filter": "All"
}.items():
    st.session_state.setdefault(k, v)

# ---------------------- Constants / regex ----------------------
COURSE_HINTS = [
    "course","class","prereq","prerequisite","credit","credits","catalog","syllabus","covers",
    "topic","learn","teaches","semester","enroll","registration","requirement","requirements",
    "what is","describe","explain","about"
]
COURSE_CODE_RE = re.compile(r"\b([A-Za-z]{2,4})\s*-?\s*(\d{3})\b")
NAME_RE = re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z0-9_\- ]{1,40})\b", re.IGNORECASE)

KNOWN_COLLEGES = {"CAS","COP","COD"}
LANG_OPTIONS = {"English":"English","Arabic":"Arabic"}

DEGREE_TOTAL = {"CS": 126, "Pharmacy": 180, "Dentistry": 189}
MAJOR_MAP = {
    "CS": {"college": "CAS", "prefixes": ("CSC","MAT","STA")},
    "Pharmacy": {"college": "COP", "prefixes": ("PHA","CHE","BIO")},
    "Dentistry": {"college": "COD", "prefixes": ("BDS","BIO","CHE")},
}

TOP_K = int(TOP_K)
CHUNK_CHAR_CAP = 900
HISTORY_TURNS = 6
_CODE_RE = re.compile(r"^[A-Za-z]{2,5}\s?\d{3}$")

def _trim(s,n): s=(s or "").strip(); return (s[:n]+"…") if len(s)>n else s

def prepare_history_text(messages, n_pairs=HISTORY_TURNS):
    out, user_seen = [], 0
    for m in reversed(messages or []):
        out.append(m)
        if m.get("role") == "user":
            user_seen += 1
            if user_seen >= n_pairs:
                break
    out.reverse()
    return "\n".join(f"{m['role']}: {m.get('content','')}" for m in out if m.get("content"))

def fastpath_course_code(q, rows):
    if not q or not _CODE_RE.match(q.strip()): return None
    key = re.sub(r"\s+","",q).upper()
    for r in rows or []:
        code = re.sub(r"\s+","",str(r.get("code",""))).upper()
        if code == key:
            return f"**{r.get('code','')} — {(r.get('title') or '').strip()}**\n\n{(r.get('description') or '').strip()}"
    return None

def build_kb_from_docs(semantic_docs, bm25_docs, top_k=TOP_K, cap=CHUNK_CHAR_CAP):
    docs = []
    if semantic_docs: docs.extend(semantic_docs)
    if bm25_docs:     docs.extend(bm25_docs)
    docs = docs[:top_k]
    return "\n\n".join(f"[{i+1}] {_trim(getattr(d,'page_content',str(d)), cap)}" for i,d in enumerate(docs))

def get_current_major_key() -> str:
    opts = sorted(MAJOR_MAP.keys())
    if "schedule_major_key" not in st.session_state or st.session_state.schedule_major_key not in opts:
        st.session_state.schedule_major_key = opts[0]
    return st.session_state.schedule_major_key

def _norm_text(s: str) -> str:
    return (s or "").lower().translate(str.maketrans("", "", string.punctuation)).strip()

def expand_synonyms(q: str) -> str:
    if not q: return q
    repl = {
        r"\bds\b":"data structures", r"\bdata struct(ure)?s?\b":"data structures",
        r"\balgo(rithms)?\b":"algorithms", r"\bai\b":"artificial intelligence",
        r"\bml\b":"machine learning", r"\bprog\b":"programming", r"\bpharm\b":"pharmacy",
        r"\bdent(al)?\b":"dentistry",
    }
    out = " " + q.lower() + " "
    for pat, sub in repl.items():
        out = re.sub(pat, sub, out)
    return out.strip()

def simple_token_overlap(a: str, b: str) -> int:
    A, B = set(_norm_text(a).split()), set(_norm_text(b).split())
    return len(A & B)

def rows_to_kb(rows_subset: List[Dict]) -> str:
    return "\n---\n".join(
        f"College: {(r.get('college') or 'Unknown')}\n"
        f"Course Code: {r['code']}\n"
        f"Title: {r['title']}\n"
        f"Description: {r['description']}\n"
        f"Prerequisites: {(r.get('prereqs') or 'Unknown')}\n"
        f"Credits: {(r.get('credits') or 'Unknown')}\n"
        f"[source: courses.csv | code: {r['code']}]"
        for r in rows_subset
    )

def is_coursey(q: str) -> bool:
    if not q: return False
    if COURSE_CODE_RE.search(q): return True
    ql = q.lower()
    return any(h in ql for h in COURSE_HINTS + ["csc","mat","mth","sta","pha","che","bio","bds","pharm","dent"])

def maybe_capture_name(q: str) -> None:
    if not q: return
    m = NAME_RE.search(q)
    if m: st.session_state.user_name = m.group(1).strip()

def friendly_prefix() -> str:
    n = st.session_state.get("user_name")
    return f"{n}, " if n else ""

def build_history_text(max_turns: int = 10) -> str:
    hist_msgs_text = prepare_history_text(st.session_state.get("messages", []), n_pairs=HISTORY_TURNS)
    return hist_msgs_text

def find_rows_by_code(rows: List[Dict], q: str) -> List[Dict]:
    if not rows or not q: return []
    idx = {(r.get("code","") or "").replace(" ","").upper(): r for r in rows if "code" in r}
    hits, seen = [], set()
    for dept, num in COURSE_CODE_RE.findall(q):
        key = f"{dept.upper()}{num}"
        row = idx.get(key)
        if row and row["code"] not in seen:
            hits.append(row); seen.add(row["code"])
    return hits

def find_rows_by_title(rows: List[Dict], q: str) -> List[Dict]:
    if not rows or not q: return []
    q_tokens = set(_norm_text(q).split())
    best, best_score = None, 0
    for r in rows:
        tn = _norm_text(r.get("title","")); 
        if not tn: continue
        t_tokens = set(tn.split())
        overlap = len(q_tokens & t_tokens)
        if overlap > best_score and overlap >= 2:
            best, best_score = r, overlap
    return [best] if best else []

def infer_scopes(q: str) -> Dict[str, List[str]]:
    ql = (q or "").lower()
    dept_scopes, college_scopes = [], []
    if any(w in ql for w in ["computer science"," comp sci"," cs "," c.s.","csc","programming"]): dept_scopes.append("cs")
    if any(w in ql for w in ["math","mathematics"," mat ","mth","algebra","calculus"]): dept_scopes.append("math")
    if any(w in ql for w in ["statistics"," sta ","probability"]): dept_scopes.append("stats")
    if any(w in ql for w in ["chemistry","organic","inorganic","che "]): dept_scopes.append("chem")
    if any(w in ql for w in ["biology","bio","genetics"]): dept_scopes.append("bio")
    if any(w in ql for w in ["pharmacy","pharm","pha"]): dept_scopes.append("pharm")
    if any(w in ql for w in ["dentistry","dent","bds"]): dept_scopes.append("dent")
    for tag in {"CAS","COP","COD"}:
        if tag.lower() in ql: college_scopes.append(tag)
    return {"dept": dept_scopes or ["all"], "college": college_scopes or ["all"]}

def filter_rows_by_college(rows: List[Dict], college_tag: str) -> List[Dict]:
    if college_tag == "All": return rows
    return [r for r in rows if (r.get("college","").upper() == college_tag)]

def reorder_docs_by_scopes(docs: List, scopes: Dict[str, List[str]], college_filter: str) -> List:
    if not docs: return docs
    colleges = set([c.upper() for c in scopes.get("college", []) if c != "all"])
    if college_filter != "All": colleges.add(college_filter.upper())
    dept_prefixes = tuple(p for s in scopes.get("dept", []) if s != "all"
                          for p in {"cs":("CSC","CSE"),"math":("MAT","MTH"),"stats":("STA",),"chem":("CHE","CHEM"),
                                    "bio":("BIO","BIOL"),"pharm":("PHA",),"dent":("BDS",)}.get(s,()))
    def score(d):
        s = 0
        code = (getattr(d,"metadata",{}) or {}).get("code","").upper()
        college = (getattr(d,"metadata",{}) or {}).get("college","").upper()
        if colleges and college in colleges: s -= 2
        if dept_prefixes and code.startswith(dept_prefixes): s -= 1
        return s
    return sorted(docs, key=score)

def parse_catalog_intent(q: str) -> Dict | None:
    if not q: return None
    ql = q.lower()
    scopes = infer_scopes(q)
    if "how many" in ql and "course" in ql: return {"type": "count", "scopes": scopes}
    if any(kw in ql for kw in ["list all courses","show all courses","list courses","all courses"]):
        return {"type": "list", "limit": 150, "scopes": scopes}
    return None

def hybrid_retrieve(q: str, retriever, vs, top_k: int, bm25=None) -> List:
    if not q: return []
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
            log.warning(f"BM25 failed: {e}")
    keyed = {}
    for d in (vec_docs + bm_docs):
        meta = getattr(d,"metadata",{}) or {}
        keyed[(meta.get("code"), getattr(d,"page_content",""))] = d
    merged = list(keyed.values())
    merged.sort(key=lambda d: -simple_token_overlap(qx, getattr(d,"page_content","")))
    return merged[: max(top_k * 2, top_k)]

def prepare_kb_from_docs(docs) -> str:
    if not docs: return ""
    blocks = []
    for d in docs:
        meta = getattr(d,"metadata",{}) or {}
        text = getattr(d,"page_content","")
        blocks.append(text + f"\n[source: {meta.get('source','?')} | code: {meta.get('code','?')}]")
    return "\n---\n".join(blocks).strip()

# ---------------------- Prompts ----------------------
COURSE_PROMPT = """
You are AUIBPT, a friendly and insightful university assistant.
You have access to two kinds of knowledge:
(1) A structured course catalog (the 'kb') that includes course details.
(2) General academic knowledge about how students can succeed, prepare, or plan their studies.

When you answer:
- Start with a short, natural explanation of what the course is *about* (in 2–3 sentences).
- Include its main concepts and how they connect to real-world skills or future courses.
- If the user asks for preparation, study tips, or expectations, give thoughtful, human-style advice.
- Keep your tone conversational — as if you’re an academic advisor guiding a student.
- End with a positive, encouraging line.

Make sure any course facts (credits, prerequisites, etc.) come from the kb if available.
If the kb doesn’t contain that info, guess carefully but state that it’s a general guideline.
Use warm, approachable language (use "you" and "your").
Avoid sounding like a catalog entry.
Encourage curiosity or suggest what course to take next.

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
You are AUIBPT, a friendly campus assistant. Keep your tone upbeat and ≤3 sentences.
Respond in {answer_lang}. Put ONLY your final answer inside <final>...</final>.
student_profile:
{student_context}
history:
{history}
question:
{question}
"""

UNIV_PROMPT = """
You are AUIBPT for AUIB. Use ONLY the supplied 'univ_kb' block. If unknown, say "Unknown".
Format:
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

# ---------------------- LLM & streaming ----------------------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder): self.placeholder = placeholder; self.text = ""
    def on_llm_new_token(self, token, **_): self.text += token; self.placeholder.markdown(self.text)

def _make_openai_llm(model_name: str, temperature: float, max_tokens: int, callbacks=None):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens,
                      streaming=True, callbacks=callbacks or [], max_retries=8, timeout=60.0)

def _make_groq_llm(model_name: str, temperature: float, max_tokens: int, callbacks=None):
    from langchain_groq import ChatGroq
    return ChatGroq(model_name=model_name, temperature=temperature, max_tokens=max_tokens,
                    streaming=True, callbacks=callbacks or [], max_retries=8, timeout=60.0)

def _make_ollama_llm(model_name: str, temperature: float, max_tokens: int, callbacks=None):
    from langchain_ollama import OllamaLLM
    return OllamaLLM(model=model_name, temperature=temperature, num_predict=max_tokens,
                     stop=["</final>"], callbacks=callbacks or [])

def make_llm(model_name: str, temperature: float, max_tokens: int, callbacks=None):
    OPENAI_KEY = _get_secret("OPENAI_API_KEY")
    GROQ_KEY   = _get_secret("GROQ_API_KEY")
    if OPENAI_KEY and USE_OPENAI:
        try: return _make_openai_llm(model_name, temperature, max_tokens, callbacks)
        except Exception as e: st.warning(f"OpenAI init failed: {e}. Trying Groq…")
    if GROQ_KEY:
        try:
            groq_model = model_name
            if "gpt-" in model_name.lower():
                groq_model = "llama-3.1-8b-instant"
            return _make_groq_llm(groq_model, temperature, max_tokens, callbacks)
        except Exception as e:
            st.warning(f"Groq init failed: {e}. Trying Ollama…")
    try: return _make_ollama_llm(model_name, temperature, max_tokens, callbacks)
    except Exception:
        from langchain_openai import ChatOpenAI
        st.warning("No Groq/Ollama available. Using OpenAI mini as last resort.")
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, max_tokens=max_tokens,
                          streaming=True, callbacks=callbacks or [], max_retries=8, timeout=60.0)

def ask_llm_stream(chain,
                   kb: str,
                   history_text: str,
                   q: str,
                   answer_lang: str,
                   student_context: str,
                   placeholder,
                   univ_kb: str = "",
                   groq_fallback_chain=None,
                   general_kb: str = "",
                   web_snippet: str = "") -> str:
    handler = StreamHandler(placeholder)
    payload = {
        "kb": kb,
        "univ_kb": univ_kb,
        "general_kb": general_kb,
        "web_snippet": web_snippet,
        "history": history_text,
        "question": q,
        "answer_lang": answer_lang,
        "student_context": student_context
    }
    try:
        raw = chain.invoke(payload, config={"callbacks":[handler]})
        final_text = _clean_output(_to_str(raw).strip())
        placeholder.markdown(final_text)
        return final_text
    except Exception as e:
        if groq_fallback_chain is not None:
            try:
                raw = groq_fallback_chain.invoke(payload, config={"callbacks":[handler]})
                final_text = _clean_output(_to_str(raw).strip())
                placeholder.markdown(final_text)
                return final_text
            except Exception as ee:
                log.error(f"LLM failure: {ee}")
        log.error(f"LLM failure: {e}")
        msg = "We’re a bit busy right now. Please try again shortly."
        placeholder.warning(msg)
        return msg

# ---------------------- Header / status ----------------------
status_col1, status_col2, status_col3 = st.columns([1.4, 1, 1])
with status_col1:
    st.markdown("### AUIBPT")
    st.caption(f"Model: `{MODEL_NAME}` • k={TOP_K} • T={TEMPERATURE} • max={NUM_PREDICT}")
    uid, email = _auth_user()
run_autosave_if_needed()
with status_col3:
    # Inline account box (since sidebar is hidden)
        # ---- Chats (top-right) ----
    with st.expander("Chats", expanded=False):
        prefix = "hdr_hdr"  # <-- unique namespace so keys never collide

        if sb is None:
            st.info("Sign in to save and manage chats.")
        else:
            # Get current Supabase user
            uid, email = (None, None)
            try:
                resp = sb.auth.get_user()
                u = getattr(resp, "user", None)
                if u:
                    uid = getattr(u, "id", None)
                    email = getattr(u, "email", None)
            except Exception:
                pass

            if not uid:
                st.info("Please sign in to view your chats.")
            else:
                # Ensure we have a current chat id (create one if none exist)
                try:
                    q = (
                        sb.table("chats")
                        .select("id,title,created_at")
                        .eq("user_id", uid)
                        .order("created_at", desc=True)
                        .limit(1)
                        .execute()
                    )
                    data = getattr(q, "data", q) or []
                    if not st.session_state.get("current_chat_id"):
                        if data:
                            st.session_state["current_chat_id"] = data[0]["id"]
                        else:
                            ins = sb.table("chats").insert({"user_id": uid, "title": "New chat"}).execute()
                            ins_data = getattr(ins, "data", ins) or []
                            if ins_data:
                                st.session_state["current_chat_id"] = ins_data[0]["id"]
                except Exception as e:
                    st.warning(f"Could not initialize chats: {e}")

                # Fetch all chats for the user
                try:
                    res = (
                        sb.table("chats")
                        .select("id,title,created_at")
                        .eq("user_id", uid)
                        .order("created_at", desc=True)
                        .execute()
                    )
                    chat_rows = getattr(res, "data", res) or []
                except Exception as exc:
                    chat_rows = []
                    st.warning(f"Could not fetch chats: {exc}")

                c1, c2 = st.columns([0.6, 0.4])
                with c1:
                    new_title = st.text_input(
                        "New chat title",
                        value="New chat",
                        key=f"{prefix}_new_title",
                    )
                with c2:
                    if st.button("Create chat", use_container_width=True, key=f"{prefix}_create"):
                        try:
                            ins = sb.table("chats").insert({"user_id": uid, "title": (new_title or 'New chat').strip()}).execute()
                            d = getattr(ins, "data", ins) or []
                            if d:
                                st.session_state["current_chat_id"] = d[0]["id"]
                                st.session_state["messages"] = []
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Create failed: {exc}")

                if chat_rows:
                    titles = [row.get("title") or "(untitled)" for row in chat_rows]
                    ids = [row["id"] for row in chat_rows]
                    current_chat = st.session_state.get("current_chat_id", ids[0] if ids else None)
                    try:
                        idx = ids.index(current_chat) if current_chat in ids else 0
                    except ValueError:
                        idx = 0

                    picked = st.radio(
                        "Your chats",
                        options=list(range(len(ids))),
                        format_func=lambda i: titles[i],
                        index=idx,
                        key=f"{prefix}_pick",
                    )
                    chosen_id = ids[picked]

                    bA, bB, bC = st.columns([0.33, 0.33, 0.34])
                    with bA:
                        if st.button("Open", use_container_width=True, key=f"{prefix}_open"):
                            try:
                                res = (
                                    sb.table("messages")
                                    .select("*")
                                    .eq("chat_id", chosen_id)
                                    .order("created_at", desc=False)
                                    .execute()
                                )
                                msgs = getattr(res, "data", res) or []
                                st.session_state["current_chat_id"] = chosen_id
                                st.session_state["messages"] = [
                                    {"role": m.get("role"), "content": m.get("content", "")} for m in msgs
                                ]
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Open failed: {exc}")

                    with bB:
                        new_name = st.text_input(
                            "Rename to",
                            value=titles[picked],
                            key=f"{prefix}_rename_input",
                        )
                        if st.button("Rename", use_container_width=True, key=f"{prefix}_rename"):
                            try:
                                (
                                    sb.table("chats")
                                    .update({"title": (new_name or '(untitled)').strip()})
                                    .eq("id", chosen_id)
                                    .execute()
                                )
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Rename failed: {exc}")

                    with bC:
                        if st.button("Delete", use_container_width=True, key=f"{prefix}_delete"):
                            try:
                                sb.table("chats").delete().eq("id", chosen_id).execute()
                                if st.session_state.get("current_chat_id") == chosen_id:
                                    st.session_state["current_chat_id"] = None
                                    st.session_state["messages"] = []
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Delete failed: {exc}")
                else:
                    st.caption("No chats yet. Create your first chat above.")

               
            # Auto-load settings on first paint of a signed-in session
            if uid and not st.session_state.get("_settings_applied_once"):
                if load_profile_settings(uid):
                    st.session_state["_settings_applied_once"] = True
        # Optional: force a rerun to immediately reflect theme/UI
                    st.rerun()

        
        # Best-effort session persistence flag (email only; no passwords are stored)
            st.session_state.setdefault("remember_me", True)

        if sb is None:
            st.info("Login is disabled (no Supabase keys configured).")
        else:
            uid, em = _auth_user()

            # If we previously asked to remember and Supabase still has a session,
            # _auth_user() will return a uid. Nothing else to do here.
            if uid:
                st.success(f"Signed in as {em or 'unknown'}")

                a1, a2 = st.columns(2)
                with a1:
                    if st.button("Sign out", use_container_width=True, key="acct_signout"):
                        try:
                            sign_out()
                        finally:
                            st.rerun()
                

                st.caption("— Your chats and settings are saved to your account —")

                
            else:
                # Not signed in yet
                st.caption("Sign in to save chats & settings.")
                em_in = st.text_input("Email", key="auth_email_hdr")
                pw_in = st.text_input("Password", type="password", key="auth_pw_hdr")

                # Remember me toggle (stores preference; cannot store passwords)
                st.toggle("Remember me (stay signed in)", key="remember_me",
                          value=st.session_state.get("remember_me", True),
                          help="We’ll try to keep your session active so you don’t have to sign in again.")

                if st.button("Sign in", use_container_width=True, key="acct_signin"):
                    try:
                        sign_in(em_in, pw_in)
                        # If remember_me is on, we rely on Supabase to keep the session valid.
                        # (No password or token is stored by this app.)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sign in failed: {e}")

                with st.expander("Create account"):
                    new_em = st.text_input("Email (new)", key="auth_email_new_hdr")
                    new_pw = st.text_input("Password (new)", type="password", key="auth_pw_new_hdr")
                    if st.button("Create account", use_container_width=True, key="hdr_create_acct"):
                        try:
                            sign_up(new_em, new_pw)
                        except Exception as e:
                            st.error(f"Sign up failed: {e}")


    with st.expander("Appearance", expanded=False):
        render_appearance_controls()
    with st.expander("Settings", expanded=False):
        st.session_state.answer_lang = st.selectbox(
    "Answer language",
    ["English", "Arabic"],
    index=["English", "Arabic"].index(st.session_state.answer_lang),
    on_change=queue_autosave,
    key="answer_lang_select",
)
# [ANCHOR: AUTOSAVE_LANG]
        st.session_state.debug = st.toggle(
    "Debug (show internal details)",
    value=st.session_state.debug,
    help="For developers: shows retrieved context, timings, and extra logs.",
    on_change=queue_autosave,
    key="debug_toggle",
)
# [ANCHOR: AUTOSAVE_DEBUG]
        try:
            colleges = sorted(list(KNOWN_COLLEGES))
        except Exception:
            colleges = []
        options = ["All"] + colleges
        try:
            _idx = options.index(st.session_state.college_filter)
        except ValueError:
            _idx = 0
        st.session_state.college_filter = st.selectbox("College filter", options, index=_idx,on_change=queue_autosave)
        st.session_state.setdefault("mobile_mode", False)
    st.session_state.mobile_mode = st.toggle(
    "Mobile-friendly controls", value=st.session_state.mobile_mode, help="Use sliders/radios/pills instead of inputs on phones.",on_change=queue_autosave,key="mobile_mode_toggle",
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
except Exception as e:
    st.error(f"Failed to prepare index or load catalog: {e}")
    st.stop()


GENERAL_KB_TEXT = load_general_kb("general_academic_kb.json")

# ---------------------- LLM init ----------------------
llm = make_llm(MODEL_NAME, TEMPERATURE, NUM_PREDICT)
course_chain = ChatPromptTemplate.from_template(COURSE_PROMPT) | llm
chat_chain   = ChatPromptTemplate.from_template(CHAT_PROMPT)   | llm
univ_chain   = ChatPromptTemplate.from_template(UNIV_PROMPT)   | llm

groq_fallback_course = groq_fallback_chat = groq_fallback_univ = None
if _get_secret("GROQ_API_KEY"):
    try:
        groq_llm = _make_groq_llm("llama-3.1-8b-instant", TEMPERATURE, NUM_PREDICT)
        groq_fallback_course = ChatPromptTemplate.from_template(COURSE_PROMPT) | groq_llm
        groq_fallback_chat   = ChatPromptTemplate.from_template(CHAT_PROMPT)   | groq_llm
        groq_fallback_univ   = ChatPromptTemplate.from_template(UNIV_PROMPT)   | groq_llm
    except Exception as e:
        st.warning(f"Groq fallback unavailable: {e}")

# ---------------------- LA rules & schedule builder ----------------------
LA_REQUIREMENTS = {"General":1,"Communication":3,"Quantitative":2,"Humanities":4,"SocialScience":2,"NaturalScience":2}
LA_CATEGORY = {
    "UNI101":"General",
    "ENL101":"Communication","ENL201":"Communication","ENL210":"Communication",
    "CSC101":"Quantitative","MAT101":"Quantitative",
    "HIS101":"Humanities","HIS102":"Humanities","HIS105":"Humanities",
    "HUM101":"Humanities","LIT101":"Humanities","PHA210":"Humanities",
    "PHI101":"Humanities","POL125":"Humanities","TLD100":"Humanities","TLD101":"Humanities","TLD102":"Humanities","TLD103":"Humanities",
    "COM101":"SocialScience","ECO101":"SocialScience","FIN101":"SocialScience","HCT108":"SocialScience","MIS101":"SocialScience",
    "POL101":"SocialScience","POL112":"SocialScience","POL191":"SocialScience","PSY101":"SocialScience","SOC101":"SocialScience",
    "CHE100":"NaturalScience","ENV201":"NaturalScience","GEO101":"NaturalScience","PHY100":"NaturalScience","PHY105":"NaturalScience",
}
LA_QUANT_BOTH = {"CSC101","MAT101"}
MAJOR_WEIGHT, LA_WEIGHT = 3, 1
DIFFICULTY_WEIGHT_MAP = {"Easy":(2.0,1.0),"Medium":(3.0,1.0),"Hard":(4.0,1.0)}

def _parse_prereq_codes(prereq_text: str) -> List[str]:
    if not prereq_text: return []
    codes = []
    for p in re.split(r"[;,/]+", prereq_text):
        for m in re.finditer(r"\b([A-Za-z]{2,4})\s*-?\s*(\d{3})\b", p):
            codes.append((m.group(1)+m.group(2)).upper())
    return sorted(set(codes))

def _credits_from_str(x: str) -> int:
    if x is None: return 3
    s = str(x).strip()
    if not s: return 3
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else 3

def la_completed_counts(taken_codes: Set[str]) -> Dict[str,int]:
    counts = {k:0 for k in LA_REQUIREMENTS}
    for code in taken_codes:
        cat = LA_CATEGORY.get(code.upper())
        if cat: counts[cat]+=1
    return counts

def la_remaining(counts: Dict[str,int], taken_codes: Set[str]) -> Dict[str,int]:
    remain = {}
    for cat, need in LA_REQUIREMENTS.items():
        have = counts.get(cat,0)
        remain[cat] = max(0, need - have)
    have_both = LA_QUANT_BOTH.issubset({c.upper() for c in taken_codes})
    remain["Quantitative"] = 0 if have_both else len(LA_QUANT_BOTH - {c.upper() for c in taken_codes})
    return remain

def la_recommend_pool(taken_codes: Set[str], rows_scope: List[Dict], remain: Dict[str,int]) -> Dict[str,List[Dict]]:
    taken_codes = {c.upper() for c in taken_codes}
    code_to_row = {r["code"].upper(): r for r in rows_scope}
    by_cat = {k: [] for k in LA_REQUIREMENTS}
    for code, cat in LA_CATEGORY.items():
        r = code_to_row.get(code)
        if not r: continue
        if remain.get(cat,0) <= 0: continue
        if code in taken_codes: continue
        req_codes = _parse_prereq_codes(r.get("prereqs",""))
        if not all(rc in taken_codes for rc in req_codes): continue
        by_cat[cat].append(r)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda r: (int(re.search(r"(\d{3})$", r["code"]).group(1)) if re.search(r"(\d{3})$", r["code"]) else 999, r.get("title","")))
    return by_cat

def _eligible_major_rows(taken_codes: Set[str], rows_scope: List[Dict], prefixes: Tuple[str,...]) -> List[Dict]:
    taken_codes = {c.upper() for c in taken_codes}
    pool = []
    for r in rows_scope:
        code = r["code"].upper()
        if code in taken_codes: continue
        if not code.startswith(prefixes): continue
        req_codes = _parse_prereq_codes(r.get("prereqs",""))
        if req_codes and not all(rc in taken_codes for rc in req_codes): continue
        pool.append(r)
    def score(r):
        req = _parse_prereq_codes(r.get("prereqs",""))
        lvl_m = re.search(r"(\d{3})$", r["code"].upper())
        lvl = int(lvl_m.group(1)) if lvl_m else 0
        return (-len(req), lvl, r.get("title",""))
    return sorted(pool, key=score)

def _credits_completed(taken_codes: Set[str], rows_all: List[Dict]) -> int:
    idx = {r["code"].upper(): r for r in rows_all}
    total = 0
    for c in taken_codes:
        r = idx.get(c.upper())
        if r: total += _credits_from_str(r.get("credits"))
    return total

def student_context_from_taken(rows_all: List[Dict], taken_codes: Set[str]) -> str:
    if not taken_codes: return "Completed: (none)"
    idx = {r["code"].upper(): r for r in rows_all}
    items = []
    for c in sorted({x.upper() for x in taken_codes}):
        r = idx.get(c); 
        if not r: continue
        cr = r.get("credits") or ""
        title = r.get("title") or ""
        items.append(f"{c} ({title}; {cr} cr)")
    completed_credits = _credits_completed(taken_codes, rows_all)
    return "Completed (" + str(completed_credits) + " credits): " + "; ".join(items)

def get_semester_weights(difficulty: str) -> tuple[float,float]:
    return DIFFICULTY_WEIGHT_MAP.get(difficulty, DIFFICULTY_WEIGHT_MAP["Medium"])

def build_semester_schedule(
    major_key: str,
    target_credits: int,
    taken_codes: Set[str],
    rows_all: List[Dict],
    difficulty: str,
    desired_major: Optional[int] = None,
    desired_la: Optional[int] = None,
) -> Tuple[List[Dict], Dict[str,int], Dict[str,int], int]:
    used_codes: Set[str] = set(c.upper() for c in taken_codes)
    code_to_row = {r["code"].upper(): r for r in rows_all}

    la_counts = la_completed_counts(used_codes)
    la_remain = la_remaining(la_counts, used_codes)
    la_pool_by_cat = la_recommend_pool(used_codes, rows_all, la_remain)

    la_flat: List[Tuple[str, Dict]] = []
    for cat, lst in la_pool_by_cat.items():
        for r in lst: la_flat.append((cat, r))

    major_info = MAJOR_MAP[major_key]
    major_pool = _eligible_major_rows(used_codes, rows_all, major_info["prefixes"])

    def cr_of(r: Dict) -> int: return _credits_from_str(r.get("credits"))
    schedule: List[Dict] = []; cur_credits = 0

    def try_add_row(r: Dict) -> bool:
        nonlocal cur_credits
        cu = r["code"].upper()
        if cu in used_codes: return False
        reqs = _parse_prereq_codes(r.get("prereqs",""))
        if any(rc not in used_codes for rc in reqs): return False
        c = cr_of(r)
        if cur_credits + c > target_credits: return False
        schedule.append(r); used_codes.add(cu); cur_credits += c
        return True

    # Ensure Quantitative pair if missing
    for q_code in ["CSC101","MAT101"]:
        if la_remain.get("Quantitative",0) > 0 and q_code not in used_codes:
            rq = code_to_row.get(q_code)
            if rq and try_add_row(rq):
                la_counts = la_completed_counts(used_codes)
                la_remain = la_remaining(la_counts, used_codes)
                la_pool_by_cat = la_recommend_pool(used_codes, rows_all, la_remain)
                la_flat = []
                for cat, lst in la_pool_by_cat.items():
                    for r in lst: la_flat.append((cat, r))

    # Decide goals
    if desired_major is not None or desired_la is not None:
        avg_cr = 3
        est_slots = max(1, min(7, target_credits // avg_cr))
        if desired_major is None and desired_la is not None:
            desired_major = max(0, est_slots - int(desired_la))
        if desired_la is None and desired_major is not None:
            desired_la = max(0, est_slots - int(desired_major))
        desired_major = max(0, int(desired_major or 0))
        desired_la = max(0, int(desired_la or 0))
    else:
        major_weight, la_weight = get_semester_weights(difficulty)
        avg_cr = 3; total_slots = max(1, min(7, target_credits // avg_cr))
        if major_weight + la_weight <= 0:
            major_goal, la_goal = total_slots, 0
        else:
            major_goal = round(total_slots * (major_weight/(major_weight+la_weight)))
            la_goal = total_slots - major_goal
        desired_major, desired_la = max(0, major_goal), max(0, la_goal)

    major_remaining = int(desired_major)
    la_remaining_goal = int(desired_la)

    guard, MAX_ITERS = 0, 1000
    while cur_credits < target_credits and guard < MAX_ITERS:
        guard += 1
        major_pool = [r for r in _eligible_major_rows(used_codes, rows_all, major_info["prefixes"])
                      if r["code"].upper() not in used_codes]

        la_counts = la_completed_counts(used_codes)
        la_remain = la_remaining(la_counts, used_codes)
        la_pool_by_cat = la_recommend_pool(used_codes, rows_all, la_remain)
        la_flat = [(cat, r) for cat, lst in la_pool_by_cat.items() for r in lst if r["code"].upper() not in used_codes]
        la_any_available = bool(la_flat)

        picked = False

        def pick_major_first() -> bool:
            if major_remaining > 0 and la_remaining_goal > 0:
                if any(v > 0 for v in la_remain.values()): return False
                return True
            if major_remaining > 0: return True
            if la_remaining_goal > 0: return False
            return bool(major_pool)

        try_major = pick_major_first()

        for bucket in (["major","la"] if try_major else ["la","major"]):
            if bucket == "major":
                if major_remaining <= 0 or not major_pool: continue
                for r in major_pool:
                    if try_add_row(r):
                        major_remaining -= 1; picked = True; break
                if picked: break
            else:
                if la_remaining_goal <= 0 or not la_any_available: continue
                for (cat, r) in la_flat:
                    if la_remain.get(cat,0) <= 0 and any(v>0 for v in la_remain.values()):
                        continue
                    if try_add_row(r):
                        la_remaining_goal -= 1; picked = True; break
                if picked: break

        if not picked:
            for r in major_pool:
                if try_add_row(r): picked = True; break
            if not picked:
                for (cat, r) in la_flat:
                    if try_add_row(r): picked = True; break

        if not picked: break
        if major_remaining <= 0 and la_remaining_goal <= 0 and cur_credits >= target_credits - 2:
            break

    return schedule, la_counts, la_remain, cur_credits

def _rebuild_pools(major_key: str, taken_codes: Set[str], rows_all: List[Dict]) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    la_counts = la_completed_counts(taken_codes)
    la_remain = la_remaining(la_counts, taken_codes)
    la_pool = la_recommend_pool(taken_codes, rows_all, la_remain)
    major_pool = _eligible_major_rows(taken_codes, rows_all, MAJOR_MAP[major_key]["prefixes"])
    return la_pool, major_pool

def _export_schedule_csv(slots: List[Dict]) -> bytes:
    output = io.StringIO(); writer = csv.writer(output)
    writer.writerow(["code","title","credits","category","prereqs"])
    for s in slots:
        r = s["candidates"][s["current_idx"]]
        writer.writerow([r["code"], r["title"], r.get("credits") or "", s["origin"], r.get("prereqs") or ""])
    return output.getvalue().encode("utf-8")

def export_schedule_json(slots):
    data = [{"origin": s["origin"], "current_idx": s["current_idx"],
             "candidates": [{"code": r["code"],"title": r["title"],"credits": r.get("credits"),"prereqs": r.get("prereqs")}
                            for r in s["candidates"]]} for s in slots]
    return json.dumps({"version":"1.0","slots":data}, ensure_ascii=False, indent=2).encode("utf-8")

def import_schedule_json(payload_bytes):
    obj = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(obj, dict) or "slots" not in obj: raise ValueError("Invalid schedule file.")
    new_slots = []
    for s in obj["slots"]:
        if not {"origin","current_idx","candidates"} <= set(s): continue
        cand_rows = []
        for r in s["candidates"]:
            if "code" in r and "title" in r:
                cand_rows.append({"code":r["code"],"title":r["title"],"credits":r.get("credits"),"prereqs":r.get("prereqs")})
        if cand_rows:
            new_slots.append({"id": f"import-{len(new_slots)}","origin": s["origin"],"candidates": cand_rows,
                              "current_idx": int(s["current_idx"]),"locked": bool(s.get("locked", False))})
    return new_slots

def _auto_top_up(major_key: str, target_credits: int, taken_codes: Set[str], slots: List[Dict], rows_all: List[Dict]) -> None:
    def total(): return sum(_credits_from_str(s["candidates"][s["current_idx"]].get("credits")) for s in slots)
    current_total = total()
    if current_total >= target_credits: return
    used = set(taken_codes) | {s["candidates"][s["current_idx"]]["code"].upper() for s in slots}
    la_pool, major_pool = _rebuild_pools(major_key, used, rows_all)
    la_candidates = [r for pool in la_pool.values() for r in pool if r["code"].upper() not in used]
    major_candidates = [r for r in major_pool if r["code"].upper() not in used]
    for origin, cand_list in [("LA:Any", la_candidates), ("Major", major_candidates)]:
        for r in cand_list:
            cr = _credits_from_str(r.get("credits"))
            if current_total + cr <= target_credits:
                slots.append({"id": f"extra-{origin}-{len(slots)}","origin": origin,"candidates": [r],"current_idx": 0,"locked": False})
                used.add(r["code"].upper()); current_total += cr
                if current_total >= target_credits: return

def _toggle_lock(slot):
    slot["locked"] = not bool(slot.get("locked"))

def _push_swap(slot_idx, prev_idx): st.session_state.swap_history.append((slot_idx, prev_idx))
def _undo_swap():
    if not st.session_state.swap_history: return False
    slot_idx, prev_idx = st.session_state.swap_history.pop()
    if 0 <= slot_idx < len(st.session_state.schedule_slots):
        st.session_state.schedule_slots[slot_idx]["current_idx"] = prev_idx; return True
    return False

def render_schedule_builder(rows_all, vs, bm25):
    st.markdown("### Schedule Builder")

    # Mobile mode toggle (put this here if you don't already have it in your header)
    mcol1, mcol2 = st.columns([1, 1])
    with mcol2:
        st.session_state.mobile_mode = st.toggle(
            "Mobile-friendly controls",
            value=st.session_state.get("mobile_mode", False),
            help="Use touch controls on phones (no typing)."
        )

    # Defaults
    if "schedule_difficulty" not in st.session_state:
        st.session_state.schedule_difficulty = "Medium"
    if "schedule_custom_mode" not in st.session_state:
        st.session_state.schedule_custom_mode = False
    if "schedule_target_credits" not in st.session_state:
        st.session_state.schedule_target_credits = 15
    st.session_state.setdefault("schedule_major_count", 3)
    st.session_state.setdefault("schedule_la_count", 2)

    # Difficulty (radio is already mobile-friendly)
    diff_opts = ["Easy", "Medium", "Hard", "Custom"]
    current_choice = "Custom" if st.session_state.schedule_custom_mode else st.session_state.schedule_difficulty
    difficulty_choice = st.radio(
        "Semester difficulty",
        options=diff_opts,
        index=diff_opts.index(current_choice),
        horizontal=True,
        help="Easy favors LA; Hard favors Major; Custom lets you set exact counts."
    )
    if difficulty_choice == "Custom":
        st.session_state.schedule_custom_mode = True
    else:
        st.session_state.schedule_custom_mode = False
        st.session_state.schedule_difficulty = difficulty_choice

    # Major (select → radio on mobile)
    options = sorted(MAJOR_MAP.keys())
    current = get_current_major_key()
    # Major (select → radio on mobile)
    options = sorted(MAJOR_MAP.keys())
    current = get_current_major_key()
    major_key = ui_select(
    "Major / program",
    options,
    default=(current if current in options else (options[0] if options else None)),
    key="schedule_major_key"
    )
    _major_key = get_current_major_key()


    # Target credits (slider is already touch-friendly; keep)
    target_credits = st.slider(
        "Target credits",
        min_value=9, max_value=21,
        value=int(st.session_state.schedule_target_credits), step=1 ,on_change=queue_autosave,
    )
    st.session_state.schedule_target_credits = target_credits

    # Custom counts (number_input → select_slider on mobile)
    if st.session_state.schedule_custom_mode:
        cols_counts = st.columns(2)
        with cols_counts[0]:
            st.session_state.schedule_major_count = ui_int(
                "Major courses this term",
                min_value=0, max_value=7, step=1,
                value=int(st.session_state.get("schedule_major_count", 3)),
                key="ui_major_count"
            )
        with cols_counts[1]:
            st.session_state.schedule_la_count = ui_int(
                "Liberal Arts courses this term",
                min_value=0, max_value=7, step=1,
                value=int(st.session_state.get("schedule_la_count", 2)),
                key="ui_la_count"
            )

    # Picker scope (radio is fine)
    picker_scope = st.radio("Completed-course picker scope:", ["Major only","Liberal Arts only","Both"], horizontal=True)
    major_prefixes = MAJOR_MAP[_major_key]["prefixes"]
    major_only_rows = [r for r in rows_all if r["code"].upper().startswith(major_prefixes)]
    la_only_rows    = [r for r in rows_all if r["code"].upper() in LA_CATEGORY]
    if picker_scope == "Major only":
        picker_rows = major_only_rows
    elif picker_scope == "Liberal Arts only":
        picker_rows = la_only_rows
    else:
        seen = set(); picker_rows = []
        for r in major_only_rows + la_only_rows:
            cu = r["code"].upper()
            if cu not in seen:
                picker_rows.append(r); seen.add(cu)

    labels = [f"{r['code']} — {r['title']}" for r in picker_rows]
    label_to_code = {f"{r['code']} — {r['title']}": r["code"].upper() for r in picker_rows}
    visible_codes = set(label_to_code.values())

    # Completed courses (multiselect → pills/checkbox grid on mobile)
    preselected_labels = [lbl for lbl, code in label_to_code.items() if code in st.session_state.completed_codes_all]
    picked_labels = ui_multi(
        "I have completed:",
        labels,
        default=preselected_labels,
        key="completed_picker"
    )
    picked_visible_codes = {label_to_code[lbl] for lbl in picked_labels}
    hidden_kept = st.session_state.completed_codes_all - visible_codes
    st.session_state.completed_codes_all = hidden_kept | picked_visible_codes
    taken_codes_all = set(st.session_state.completed_codes_all)

    # Progress
    completed_credits = _credits_completed(taken_codes_all, rows_all)
    degree_total = DEGREE_TOTAL.get(_major_key, 126)
    st.caption(f"Progress: {completed_credits} / {degree_total} credits • Target this term: {target_credits}")

    # Action buttons
    col_build_a, col_build_b, col_build_c, col_build_d, col_build_e = st.columns([0.35,0.2,0.2,0.15,0.1])
    with col_build_a: build_btn = st.button("Build schedule", use_container_width=True)
    with col_build_b: reset_btn = st.button("Reset", use_container_width=True)
    with col_build_c: topup_btn = st.button("Auto top-up", use_container_width=True, disabled=not st.session_state.get("schedule_slots"))
    with col_build_d: undo_btn  = st.button("Undo swap", use_container_width=True, disabled=not st.session_state.get("swap_history"))
    with col_build_e:
        if st.button("Close", use_container_width=True):
            st.session_state.show_schedule = False; st.rerun()

    if reset_btn:
        st.session_state.schedule_slots = []; st.session_state.schedule_planned_credits = 0; st.rerun()

    if build_btn:
        desired_major = st.session_state.schedule_major_count if st.session_state.schedule_custom_mode else None
        desired_la    = st.session_state.schedule_la_count    if st.session_state.schedule_custom_mode else None
        schedule, la_counts, la_remain, planned_credits = build_semester_schedule(
            major_key=_major_key, target_credits=target_credits, taken_codes=taken_codes_all,
            rows_all=rows_all, difficulty=st.session_state.get("schedule_difficulty","Medium"),
            desired_major=desired_major, desired_la=desired_la
        )
        la_pool, major_pool = _rebuild_pools(_major_key, taken_codes_all, rows_all)
        slots = []; used = set(taken_codes_all)
        for idx, c in enumerate(schedule):
            if c["code"].upper() in LA_CATEGORY:
                origin = f"LA:{LA_CATEGORY[c['code'].upper()]}"; pool = la_pool.get(LA_CATEGORY[c["code"].upper()], [])
            else:
                origin = "Major"; pool = major_pool
            candidates, seen_codes = [], set()
            candidates.append(c); seen_codes.add(c["code"].upper())
            for r in pool:
                cu = r["code"].upper()
                if cu not in seen_codes and cu not in used:
                    candidates.append(r); seen_codes.add(cu)
            slots.append({"id": f"{origin}-{idx}","origin": origin,"candidates": candidates,"current_idx": 0,"locked": False})
            used.add(c["code"].upper())
        st.session_state.schedule_slots = slots
        st.session_state.schedule_planned_credits = sum(_credits_from_str(s["candidates"][0].get("credits")) for s in slots)

    if topup_btn and st.session_state.get("schedule_slots"):
        _auto_top_up(_major_key, target_credits, taken_codes_all, st.session_state.schedule_slots, rows_all)
        st.session_state.schedule_planned_credits = sum(_credits_from_str(s["candidates"][s["current_idx"]].get("credits")) for s in st.session_state.schedule_slots)
        st.rerun()

    if undo_btn:
        if _undo_swap(): st.rerun()
        else: st.info("Nothing to undo.")

    if st.session_state.get("schedule_slots"):
        st.markdown("### Suggested schedule")
        new_total = 0
        for i, slot in enumerate(st.session_state.schedule_slots):
            cur = slot["candidates"][slot["current_idx"]]
            pr = cur.get("prereqs") or "None/Unknown"
            cr = cur.get("credits") or "Unknown"
            cols = st.columns([0.64,0.12,0.12,0.12])
            with cols[0]:
                st.markdown(f"**{cur['code']} — {cur['title']}**  \nCategory: {slot['origin']} • Credits: {cr} • Prereqs: {pr}  \nStatus: {'Locked' if slot.get('locked') else 'Unlocked'}")
                why_bits = []
                if cur["code"].upper() in LA_CATEGORY: why_bits.append(f"meets {LA_CATEGORY[cur['code'].upper()]}")
                else: why_bits.append("major requirement/elective")
                reqs = _parse_prereq_codes(cur.get("prereqs",""))
                if not reqs: why_bits.append("no explicit prerequisites")
                else:
                    if all(rc in taken_codes_all for rc in reqs): why_bits.append("prerequisites satisfied")
                    else: why_bits.append("prerequisites satisfied during planning")
                st.caption("Why this? " + " • ".join(why_bits))
            with cols[1]:
                if st.button("Swap", key=f"swap_{slot['id']}", disabled=bool(slot.get("locked", False))):
                    current_used = {c["candidates"][c["current_idx"]]["code"].upper() for c in st.session_state.schedule_slots}
                    current_used.discard(cur["code"].upper())
                    replaced = False
                    for j in range(slot["current_idx"] + 1, len(slot["candidates"])):  # find next eligible
                        cand = slot["candidates"][j]; code_u = cand["code"].upper()
                        if code_u in current_used or code_u in taken_codes_all: continue
                        old_cr = _credits_from_str(cur.get("credits")); new_cr = _credits_from_str(cand.get("credits"))
                        current_total = sum(_credits_from_str(s["candidates"][s["current_idx"]].get("credits")) for s in st.session_state.schedule_slots)
                        if current_total - old_cr + new_cr <= st.session_state.schedule_target_credits:
                            _push_swap(i, slot["current_idx"]); slot["current_idx"] = j; replaced = True; break
                    if not replaced: st.info(f"No more eligible options for {cur['code']} — {cur['title']}.")
                    st.rerun()
            with cols[2]:
                if st.button("Lock" if not slot.get("locked") else "Unlock", key=f"lock_{slot['id']}"):
                    _toggle_lock(slot); st.rerun()
            with cols[3]:
                pass
            new_total += _credits_from_str(cur.get("credits"))
        st.session_state.schedule_planned_credits = new_total
        st.success(f"Planned credits: {new_total} / Target {st.session_state.schedule_target_credits}")

        csv_bytes = _export_schedule_csv(st.session_state.schedule_slots)
        st.download_button("Export schedule as CSV", data=csv_bytes, file_name="schedule.csv", mime="text/csv")

        json_bytes = export_schedule_json(st.session_state.schedule_slots)
        st.download_button("Save schedule (JSON)", data=json_bytes, file_name="schedule.json", mime="application/json")
        up = st.file_uploader("Load a saved schedule (JSON)", type=["json"], key="sched_loader")
        if up is not None:
            try:
                st.session_state.schedule_slots = import_schedule_json(up.read())
                st.session_state.schedule_planned_credits = sum(_credits_from_str(s["candidates"][s["current_idx"]].get("credits")) for s in st.session_state.schedule_slots)
                st.rerun()
            except Exception as e:
                st.warning(f"Could not load schedule: {e}")

# ---------------------- Chat history render ----------------------
def render_history():
    assistant_avatar = "RP.png" if os.path.exists("RP.png") else None
    for m in st.session_state.messages:
        if m["role"] == "user":
            with st.chat_message("user", avatar=st.session_state.get("profile_avatar_path")):
                st.markdown(m["content"])
        else:
            with st.chat_message("assistant", avatar=assistant_avatar or ""):
                st.markdown(m["content"])

render_history()

def _student_profile_for_prompt() -> str:
    return student_context_from_taken(rows_all, st.session_state.completed_codes_all)

# ---------------------- Chat input (no rerun; stream answer) ----------------------
if 'clear_clicked' in locals() and clear_clicked:
    st.session_state.messages = []
    st.rerun()

input_disabled = st.session_state.get("is_generating", False)
if st.session_state.get("is_generating", False):
    st.info("Assistant is generating a response...")
q = st.chat_input("Type your message…", disabled=input_disabled)

# Validate empty input
if q is not None and not q.strip():
    st.warning("Please type a message first.")

if q is not None and q.strip():
    st.session_state.is_generating = True
    start_ts = time.time()

    # user bubble + persist user message
    with st.chat_message("user", avatar=st.session_state.get("profile_avatar_path")):
        st.markdown(q)
    maybe_capture_name(q)
    st.session_state.messages.append({"role": "user", "content": q})
    if st.session_state.get("current_chat_id"):
        save_message(st.session_state["current_chat_id"], "user", q)

    scopes = infer_scopes(q)
    direct_rows = find_rows_by_code(rows_all, q)
    title_rows  = find_rows_by_title(rows, q) if not direct_rows else []
    intent = parse_catalog_intent(q)

    kb = ""
    ans = None
    history_text = build_history_text()
    student_context = _student_profile_for_prompt()
    answer_lang_str = LANG_OPTIONS.get(st.session_state.answer_lang, "English")

    assistant_avatar = "RP.png" if os.path.exists("RP.png") else None
    with st.chat_message("assistant", avatar=assistant_avatar or None):
        ans_placeholder = st.empty()

        # fast course card first
        hit = fastpath_course_code(q, rows_all)
        if hit:
            ans_placeholder.markdown(hit); ans = hit
        elif is_university_query(q):
            univ_kb_text = univ_kb_blocks_for(q) or "University facts: (none)\nFaculty: (none)"
            ans = ask_llm_stream(univ_chain, kb="", history_text=history_text, q=q,
                                 answer_lang=answer_lang_str, student_context=student_context,
                                 placeholder=ans_placeholder, univ_kb=univ_kb_text,
                                 groq_fallback_chain=groq_fallback_univ)
        elif direct_rows:
            if st.session_state.college_filter != "All":
                filtered = [r for r in direct_rows if (r.get("college","").upper() == st.session_state.college_filter)]
                if filtered: direct_rows = filtered
            kb = rows_to_kb(direct_rows)
            if st.session_state.completed_codes_all:
                kb += "\n---\n" + rows_to_kb([r for r in rows_all if r["code"].upper() in st.session_state.completed_codes_all])
            ans = ask_llm_stream(
    course_chain,
    kb, history_text, q, answer_lang_str, student_context,
    ans_placeholder,
    groq_fallback_chain=groq_fallback_course,
    general_kb=(GENERAL_KB_TEXT if needs_prep_tips(q) else ""),
    web_snippet=(web_enrichment_snippet(q) if needs_prep_tips(q) else "")
)

        elif title_rows:
            kb = rows_to_kb(title_rows)
            if st.session_state.completed_codes_all:
                kb += "\n---\n" + rows_to_kb([r for r in rows_all if r["code"].upper() in st.session_state.completed_codes_all])
            ans = ask_llm_stream(
    course_chain,
    kb, history_text, q, answer_lang_str, student_context,
    ans_placeholder,
    groq_fallback_chain=groq_fallback_course,
    general_kb=(GENERAL_KB_TEXT if needs_prep_tips(q) else ""),
    web_snippet=(web_enrichment_snippet(q) if needs_prep_tips(q) else "")
)

        elif intent:
            if intent["type"] == "count":
                scoped_rows = rows
                depts = intent["scopes"].get("dept", ["all"])
                if "all" not in depts:
                    prefixes = tuple(p for s in depts for p in {"cs":("CSC","CSE"),"math":("MAT","MTH"),
                                                                "stats":("STA",),"chem":("CHE","CHEM"),
                                                                "bio":("BIO","BIOL"),"pharm":("PHA",),
                                                                "dent":("BDS",)}.get(s,()))
                    scoped_rows = [r for r in scoped_rows if r["code"].upper().startswith(prefixes)]
                ans = f"I currently know {len(scoped_rows)} courses from courses.csv."
                ans_placeholder.markdown(ans)
            elif intent["type"] == "list":
                limit = intent.get("limit", 150)
                scoped_rows = rows
                depts = intent["scopes"].get("dept", ["all"])
                if "all" not in depts:
                    prefixes = tuple(p for s in depts for p in {"cs":("CSC","CSE"),"math":("MAT","MTH"),
                                                                "stats":("STA",),"chem":("CHE","CHEM"),
                                                                "bio":("BIO","BIOL"),"pharm":("PHA",),
                                                                "dent":("BDS",)}.get(s,()))
                    scoped_rows = [r for r in scoped_rows if r["code"].upper().startswith(prefixes)]
                lines = [f"{(r.get('college') or 'UNK')} • {r['code']} — {r['title']}" for r in scoped_rows]
                if len(lines) > limit:
                    more = len(lines) - limit
                    lines = lines[:limit] + [f"...and {more} more."]
                ans = "\n".join(lines)
                ans_placeholder.code(ans, language="markdown")
        else:
            qx = expand_synonyms(q)
            if is_coursey(qx):
                try:
                    docs = hybrid_retrieve(qx, retriever, vs, int(TOP_K), bm25=bm25)
                except Exception as e:
                    log.error(f"hybrid_retrieve error: {e}")
                    docs = []
                docs = reorder_docs_by_scopes(docs, scopes, st.session_state.college_filter)
                bm25_docs = []
                tokens = (q or "").strip().split()
                use_bm25 = len(tokens) >= 4 and any(len(t) >= 4 for t in tokens)
                if use_bm25 and bm25:
                    try:
                        all_docs = list(vs.docstore._dict.values())
                        scores = bm25.get_scores(tokens)
                        best_ids = sorted(range(len(all_docs)), key=lambda i: -scores[i])[:int(TOP_K)]
                        bm25_docs = [all_docs[i] for i in best_ids]
                    except Exception:
                        bm25_docs = []
                kb = build_kb_from_docs(docs, bm25_docs, top_k=TOP_K, cap=CHUNK_CHAR_CAP)
                if st.session_state.completed_codes_all:
                    kb += ("\n---\n" if kb else "") + rows_to_kb([r for r in rows_all if r["code"].upper() in st.session_state.completed_codes_all])
                if kb and kb != "(no relevant context found)":
                    ans = ask_llm_stream(
    course_chain,
    kb, history_text, q, answer_lang_str, student_context,
    ans_placeholder,
    groq_fallback_chain=groq_fallback_course,
    general_kb=(GENERAL_KB_TEXT if needs_prep_tips(q) else ""),
    web_snippet=(web_enrichment_snippet(q) if needs_prep_tips(q) else "")
)

                else:
                    ans = "I don't know from the provided data."; ans_placeholder.markdown(ans)
            else:
                ans = ask_llm_stream(
    course_chain,
    kb, history_text, q, answer_lang_str, student_context,
    ans_placeholder,
    groq_fallback_chain=groq_fallback_course,
    general_kb=(GENERAL_KB_TEXT if needs_prep_tips(q) else ""),
    web_snippet=(web_enrichment_snippet(q) if needs_prep_tips(q) else "")
)

                if st.session_state.get("user_name") and ans and not ans.lower().startswith(st.session_state["user_name"].lower()):
                    ans = friendly_prefix() + ans
                    ans_placeholder.markdown(ans)

        # debug panel
        if st.session_state.debug:
            elapsed = f"{(time.time() - start_ts):.2f}s"
            with st.expander(f"Debug: retrieved KB • {elapsed}"):
                st.code(kb or "(none)")
            if is_university_query(q):
                with st.expander("Debug: UNIV_KB view"):
                    st.code(univ_kb_blocks_for(q), language="markdown")
            st.caption(f"Answered in {elapsed} • Model: {MODEL_NAME} • k={TOP_K} • T={TEMPERATURE}")

    # persist assistant message
    st.session_state.messages.append({"role": "assistant", "content": ans})
    if st.session_state.get("current_chat_id"):
        save_message(st.session_state["current_chat_id"], "assistant", ans)
    st.session_state.is_generating = False

# ---------------------- Optional schedule builder panel ----------------------
if st.session_state.get("show_schedule", False):
    render_schedule_builder(rows_all, vs, bm25)
