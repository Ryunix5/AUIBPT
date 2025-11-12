"""Responsive Streamlit input helpers."""
from __future__ import annotations

from typing import Iterable, List, Optional

import streamlit as st


def _opt_index(options: Iterable, value) -> int:
    try:
        return list(options).index(value)
    except Exception:
        return 0


def ui_select(label: str, options: List, *, default=None, key=None, help: Optional[str] = None):
    """Select helper that renders touch-friendly radios on mobile."""
    if st.session_state.get("mobile_mode", False):
        idx = _opt_index(options, default)
        return st.radio(label, options, index=idx, horizontal=True, key=key, help=help)
    idx = _opt_index(options, default)
    return st.selectbox(label, options, index=idx, key=key, help=help)


def ui_multi(label: str, options: List, *, default=None, key=None, help: Optional[str] = None):
    """Multiselect helper that swaps to pill controls on mobile."""
    default = default or []
    if st.session_state.get("mobile_mode", False):
        if hasattr(st, "pills"):
            return st.pills(label, options, default=default, selection_mode="multi", key=key, help=help)
        st.caption(label)
        cols = st.columns(3)
        picked = []
        for i, opt in enumerate(options):
            if cols[i % 3].checkbox(opt, value=(opt in default), key=f"{key or label}_{i}"):
                picked.append(opt)
        return picked
    return st.multiselect(label, options, default=default, key=key, help=help)


def ui_int(label: str, *, min_value: int, max_value: int, value: int, step: int = 1, key=None, help: Optional[str] = None):
    """Integer input that stays usable on touch devices."""
    if st.session_state.get("mobile_mode", False):
        return st.select_slider(
            label,
            options=list(range(min_value, max_value + 1, step)),
            value=value,
            key=key,
            help=help,
        )
    return st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
        key=key,
        help=help,
    )
