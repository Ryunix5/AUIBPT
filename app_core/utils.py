"""Utility helpers for the Streamlit app."""
from __future__ import annotations

import os
import re
from typing import Optional

import streamlit as st


def get_secret(key: str) -> Optional[str]:
    """Fetch a secret from Streamlit or the environment."""
    value = os.getenv(key)
    if value:
        return value
    try:
        return st.secrets[key]
    except Exception:
        return None


def to_str(value) -> str:
    """Best-effort conversion that understands LangChain messages."""
    try:
        from langchain_core.messages import AIMessage

        if isinstance(value, AIMessage):
            return value.content or ""
    except Exception:
        pass
    return value if isinstance(value, str) else str(value)


THINK_TAG_RE = re.compile(r"<think\b[^>]*>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"</?think\b[^>]*>|</?final\b[^>]*>", flags=re.IGNORECASE)
FINAL_BLOCK_RE = re.compile(r"<final>(.*?)</final>", flags=re.IGNORECASE | re.DOTALL)
FINAL_OPEN_RE = re.compile(r"<final>(.*)$", flags=re.IGNORECASE | re.DOTALL)


def clean_output(text: str) -> str:
    """Strip hidden reasoning tags while preserving Markdown formatting."""
    if not text:
        return "I don't know from the provided data."

    finals = FINAL_BLOCK_RE.findall(text)
    if finals:
        for block in reversed(finals):
            block = block.strip()
            if block:
                text = block
                break
        else:
            text = finals[-1].strip()
    else:
        open_match = FINAL_OPEN_RE.search(text)
        if open_match:
            text = open_match.group(1).strip()

    text = THINK_TAG_RE.sub("", text)
    text = TAG_RE.sub("", text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)

    if " - " in text and "\n- " not in text:
        text = text.replace(" - ", "\n- ")

    text = text.strip()
    return text or "I don't know from the provided data."
