"""Supabase authentication and chat persistence helpers."""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import streamlit as st

from .utils import get_secret

try:
    from supabase import create_client  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    create_client = None  # type: ignore

if 'SupabaseClient' not in globals():
    try:
        from supabase import Client as SupabaseClient  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        SupabaseClient = Any  # type: ignore

log = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_supabase() -> Optional[SupabaseClient]:
    url = get_secret("SUPABASE_URL")
    key = get_secret("SUPABASE_ANON_KEY")
    if not url or not key or create_client is None:
        st.info("Login disabled (no Supabase keys or SDK).")
        return None
    try:
        return create_client(url, key)
    except Exception as exc:  # pragma: no cover - external sdk
        st.warning(f"Could not init Supabase: {exc}. Running without auth.")
        return None


sb: Optional[SupabaseClient] = get_supabase()


def _pg_tbl(name: str):
    if sb is None:
        return None
    if hasattr(sb, "table"):
        try:
            return sb.table(name)
        except Exception:
            pass
    if hasattr(sb, "from_"):
        return sb.from_(name)
    if hasattr(sb, "postgrest") and hasattr(sb.postgrest, "from_"):
        return sb.postgrest.from_(name)
    raise RuntimeError("No compatible supabase/postgrest table accessor found.")


def _pg_select(qb, cols="*"):
    if hasattr(qb, "select"):
        return qb.select(cols)
    raise AttributeError("This PostgREST builder does not support .select(); please upgrade packages.")


def _pg_order(qb, col: str, desc: bool = False):
    if hasattr(qb, "order"):
        return qb.order(col, desc=desc)
    return qb


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
    return qb.single() if hasattr(qb, "single") else qb


def _pg_exec(qb):
    res = qb.execute() if hasattr(qb, "execute") else qb
    if hasattr(res, "data"):
        return res.data
    if isinstance(res, dict) and "data" in res:
        return res["data"]
    return res


def _as_rows(obj) -> List[Dict]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    if hasattr(obj, "model_dump"):
        dumped = obj.model_dump()
        return [dumped] if isinstance(dumped, dict) else list(dumped or [])
    return [obj]


def _first_row(obj) -> Dict:
    rows = _as_rows(obj)
    return rows[0] if rows else {}


def _auth_enabled() -> bool:
    return sb is not None


def _auth_user() -> Tuple[Optional[str], Optional[str]]:
    user = st.session_state.get("_auth_user")
    if isinstance(user, dict):
        return user.get("id"), user.get("email")
    if hasattr(user, "id") or hasattr(user, "email"):
        return getattr(user, "id", None), getattr(user, "email", None)
    if isinstance(user, list) and user:
        first = user[0]
        if isinstance(first, dict):
            return first.get("id"), first.get("email")
        if hasattr(first, "id") or hasattr(first, "email"):
            return getattr(first, "id", None), getattr(first, "email", None)
    try:
        if sb is not None and hasattr(sb, "auth") and hasattr(sb.auth, "get_user"):
            resp = sb.auth.get_user()
            supabase_user = getattr(resp, "user", None)
            if supabase_user:
                uid = getattr(supabase_user, "id", None)
                email = getattr(supabase_user, "email", None)
                if uid:
                    st.session_state._auth_user = {"id": uid, "email": email}
                    return uid, email
    except Exception:
        pass
    return None, None


def sign_in(email: str, password: str):
    if sb is None:
        return
    sb.auth.sign_out()
    res = sb.auth.sign_in_with_password({"email": email, "password": password})
    user = getattr(res, "user", None)
    uid = getattr(user, "id", None)
    em = getattr(user, "email", None)
    st.session_state._auth_user = {"id": uid, "email": em}
    try:
        sb.table("profiles").upsert({"id": uid, "email": em}).execute()
    except Exception:
        pass


def sign_up(email: str, password: str):
    if sb is None:
        return
    sb.auth.sign_up({"email": email, "password": password})
    st.success("Check your email to verify your account, then sign in.")


def sign_out():
    if sb is None:
        return
    sb.auth.sign_out()
    st.session_state._auth_user = None
    st.session_state.pop("current_chat_id", None)
    st.session_state.pop("messages", None)


def list_chats(uid: str):
    if not _auth_enabled():
        return []
    qb = _pg_tbl("chats")
    data = _pg_exec(_pg_order(_pg_eq(_pg_select(qb, "*"), "user_id", uid), "created_at", desc=True))
    return _as_rows(data)


def create_chat(uid: str, title: str = "New chat") -> Optional[str]:
    if not _auth_enabled():
        return None
    qb = _pg_tbl("chats")
    ins = _pg_insert(qb, {"user_id": uid, "title": title})
    row = _first_row(_pg_exec(_pg_single(ins)))
    return row.get("id")


def rename_chat(chat_id: str, title: str):
    if not _auth_enabled():
        return
    qb = _pg_tbl("chats")
    _pg_exec(_pg_eq(_pg_update(qb, {"title": title}), "id", chat_id))


def delete_chat(chat_id: str):
    if not _auth_enabled():
        return
    qb = _pg_tbl("chats")
    _pg_exec(_pg_eq(_pg_delete(qb), "id", chat_id))


def load_messages(chat_id: str):
    if not _auth_enabled():
        return []
    qb = _pg_tbl("messages")
    rows = _pg_exec(_pg_order(_pg_eq(_pg_select(qb, "*"), "chat_id", chat_id), "created_at", desc=False))
    return [{"role": r.get("role"), "content": r.get("content")} for r in _as_rows(rows)]


def save_message(chat_id: str, role: str, content: str):
    if not _auth_enabled():
        return
    qb = _pg_tbl("messages")
    _pg_exec(_pg_insert(qb, {"chat_id": chat_id, "role": role, "content": content}))


def ensure_current_chat():
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
    except Exception as exc:
        st.warning(f"Could not initialize chats: {exc}")


def load_chat_messages_into_ui(chat_id: str):
    if not _auth_enabled() or not chat_id:
        return
    try:
        msgs = load_messages(chat_id) or []
        st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in msgs]
    except Exception as exc:
        st.warning(f"Could not load messages: {exc}")


def current_user() -> Tuple[Optional[str], Optional[str]]:
    return _auth_user()


__all__ = [
    "sb",
    "get_supabase",
    "sign_in",
    "sign_up",
    "sign_out",
    "list_chats",
    "create_chat",
    "rename_chat",
    "delete_chat",
    "load_messages",
    "save_message",
    "ensure_current_chat",
    "load_chat_messages_into_ui",
    "current_user",
]
