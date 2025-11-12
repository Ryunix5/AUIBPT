"""LLM creation and streaming utilities."""
from __future__ import annotations

import logging
from typing import Any, Optional

import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler

from .utils import clean_output, get_secret, to_str

log = logging.getLogger(__name__)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.text = ""

    def on_llm_new_token(self, token: str, **_):
        self.text += token
        self.placeholder.markdown(self.text)


def _make_openai_llm(model_name: str, temperature: float, max_tokens: int, callbacks=None):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
        callbacks=callbacks or [],
        max_retries=8,
        timeout=60.0,
    )


def _make_groq_llm(model_name: str, temperature: float, max_tokens: int, callbacks=None):
    from langchain_groq import ChatGroq

    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
        callbacks=callbacks or [],
        max_retries=8,
        timeout=60.0,
    )


def _make_ollama_llm(model_name: str, temperature: float, max_tokens: int, callbacks=None):
    from langchain_ollama import OllamaLLM

    return OllamaLLM(
        model=model_name,
        temperature=temperature,
        num_predict=max_tokens,
        stop=["</final>"],
        callbacks=callbacks or [],
    )


def make_llm(model_name: str, temperature: float, max_tokens: int, *, use_openai: bool, callbacks=None):
    openai_key = get_secret("OPENAI_API_KEY")
    groq_key = get_secret("GROQ_API_KEY")
    if openai_key and use_openai:
        try:
            return _make_openai_llm(model_name, temperature, max_tokens, callbacks)
        except Exception as exc:  # pragma: no cover - external SDK
            st.warning(f"OpenAI init failed: {exc}. Trying Groq…")
    if groq_key:
        try:
            groq_model = model_name
            if "gpt-" in model_name.lower():
                groq_model = "llama-3.1-8b-instant"
            return _make_groq_llm(groq_model, temperature, max_tokens, callbacks)
        except Exception as exc:  # pragma: no cover - external SDK
            st.warning(f"Groq init failed: {exc}. Trying Ollama…")
    try:
        return _make_ollama_llm(model_name, temperature, max_tokens, callbacks)
    except Exception:
        from langchain_openai import ChatOpenAI

        st.warning("No Groq/Ollama available. Using OpenAI mini as last resort.")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            callbacks=callbacks or [],
            max_retries=8,
            timeout=60.0,
        )


def ask_llm_stream(
    chain,
    *,
    kb: str,
    history_text: str,
    question: str,
    answer_lang: str,
    student_context: str,
    placeholder,
    univ_kb: str = "",
    groq_fallback_chain: Optional[Any] = None,
    general_kb: str = "",
    web_snippet: str = "",
) -> str:
    handler = StreamHandler(placeholder)
    payload = {
        "kb": kb,
        "univ_kb": univ_kb,
        "general_kb": general_kb,
        "web_snippet": web_snippet,
        "history": history_text,
        "question": question,
        "answer_lang": answer_lang,
        "student_context": student_context,
    }
    try:
        raw = chain.invoke(payload, config={"callbacks": [handler]})
        final_text = clean_output(to_str(raw).strip())
        placeholder.markdown(final_text)
        return final_text
    except Exception as exc:
        if groq_fallback_chain is not None:
            try:
                raw = groq_fallback_chain.invoke(payload, config={"callbacks": [handler]})
                final_text = clean_output(to_str(raw).strip())
                placeholder.markdown(final_text)
                return final_text
            except Exception as groq_exc:
                log.error("LLM failure: %s", groq_exc)
        log.error("LLM failure: %s", exc)
        message = "We’re a bit busy right now. Please try again shortly."
        placeholder.warning(message)
        return message


__all__ = ["make_llm", "ask_llm_stream", "StreamHandler"]
