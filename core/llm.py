# LLM functionality for AUIBPT

import os
import re
import logging
from typing import Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

log = logging.getLogger("app")

# Try to import OpenAI
try:
    import openai
    OpenAIRateLimitError = openai.RateLimitError
except Exception:
    class OpenAIRateLimitError(Exception): ...

def _get_secret(key: str) -> str | None:
    """Get secret from environment or streamlit secrets."""
    import streamlit as st
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets[key]
    except Exception:
        return None

# Get API keys
OPENAI_KEY = _get_secret("OPENAI_API_KEY")
GROQ_KEY = _get_secret("GROQ_API_KEY")

if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY
if GROQ_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_KEY

class StreamHandler(BaseCallbackHandler):
    """Stream handler for LLM responses."""
    def __init__(self, placeholder): 
        self.placeholder = placeholder
        self.text = ""
    
    def on_llm_new_token(self, token, **_): 
        self.text += token
        self.placeholder.markdown(self.text)

def _to_str(x) -> str:
    """Convert various types to string."""
    try:
        from langchain_core.messages import AIMessage
        if isinstance(x, AIMessage):
            return x.content or ""
    except Exception:
        pass
    return x if isinstance(x, str) else str(x)

def _clean_output(text: str) -> str:
    """Clean and normalize LLM output."""
    if not text:
        return "I don't know from the provided data."
    
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
    text = re.sub(r'(?is)\brules:\b.*', '', text)
    text = re.sub(r'(?is)\b(knowledge base|kb|instructions)\b.*', '', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "I don't know from the provided data."

def _make_openai_llm(model_name: str, temperature: float, num_predict: int, callbacks=None):
    """Create OpenAI LLM."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=num_predict,
        streaming=True,
        callbacks=callbacks or [],
        max_retries=8,
        timeout=60.0,
    )

def _make_groq_llm(model_name: str, temperature: float, num_predict: int, callbacks=None):
    """Create Groq LLM."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
        max_tokens=num_predict,
        streaming=True,
        callbacks=callbacks or [],
        max_retries=8,
        timeout=60.0,
    )

def _make_ollama_llm(model_name: str, temperature: float, num_predict: int, callbacks=None):
    """Create Ollama LLM."""
    from langchain_ollama import OllamaLLM
    return OllamaLLM(
        model=model_name,
        temperature=temperature,
        num_predict=num_predict,
        stop=["</final>"],
        callbacks=callbacks or [],
    )

def make_llm(model_name: str, temperature: float, num_predict: int, callbacks=None, use_openai: bool = True):
    """Create LLM with fallback chain."""
    effective_use_openai = bool(OPENAI_KEY) and use_openai
    if effective_use_openai:
        try:
            return _make_openai_llm(model_name, temperature, num_predict, callbacks)
        except Exception as e:
            import streamlit as st
            st.warning(f"OpenAI init failed: {e}. Trying Groq…")
    
    if GROQ_KEY:
        try:
            groq_model = model_name
            if "gpt-" in model_name.lower():
                groq_model = "llama-3.1-8b-instant"
            return _make_groq_llm(groq_model, temperature, num_predict, callbacks)
        except Exception as e:
            import streamlit as st
            st.warning(f"Groq init failed: {e}. Trying Ollama…")
    
    try:
        return _make_ollama_llm(model_name, temperature, num_predict, callbacks)
    except Exception:
        from langchain_openai import ChatOpenAI
        import streamlit as st
        st.warning("No Groq key and Ollama not available. Using OpenAI mini as last resort.")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            max_tokens=num_predict,
            streaming=True,
            callbacks=callbacks or [],
            max_retries=8,
            timeout=60.0,
        )

def ask_llm_stream(chain, kb: str, history_text: str, q: str, answer_lang: str, student_context: str, placeholder, univ_kb: str = "", groq_fallback_chain=None) -> str:
    """Ask LLM with streaming and fallback."""
    import streamlit as st
    import time
    
    handler = StreamHandler(placeholder)
    payload = {
        "kb": kb,
        "univ_kb": univ_kb,
        "history": history_text,
        "question": q,
        "answer_lang": answer_lang,
        "student_context": student_context
    }

    def _invoke(c, stream=True):
        if stream:
            return c.invoke(payload, config={"callbacks": [handler]})
        else:
            return c.invoke(payload)

    try:
        raw = _invoke(chain, stream=True)
        final_text = _clean_output(_to_str(raw).strip())
        placeholder.markdown(final_text)
        return final_text
    except OpenAIRateLimitError as e:
        if groq_fallback_chain is not None:
            st.info("Switching to Groq due to OpenAI rate limit.")
            try:
                raw = _invoke(groq_fallback_chain, stream=True)
                final_text = _clean_output(_to_str(raw).strip())
                placeholder.markdown(final_text)
                return final_text
            except Exception as ee:
                last_err = ee
        else:
            last_err = e
    except Exception as e:
        try:
            raw = _invoke(chain, stream=False)
            final_text = _clean_output(_to_str(raw).strip())
            placeholder.markdown(final_text)
            return final_text
        except Exception as ee:
            last_err = ee

    if groq_fallback_chain is not None:
        try:
            raw = _invoke(groq_fallback_chain, stream=False)
            final_text = _clean_output(_to_str(raw).strip())
            placeholder.markdown(final_text)
            return final_text
        except Exception as ee:
            last_err = ee

    log.error(f"LLM failure (after Groq fallback if any): {last_err}")
    msg = "We're a bit busy right now. Please try again in ~30–60s."
    placeholder.warning(msg)
    return msg
