"""
Model runner module - handles execution of Groq LLM models only.
Unified RAG retrieval is handled separately by rag_retriever module.
"""
import asyncio
import os
from typing import Optional

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# Global Groq client - set by main.py
_groq_client = None


GEN_DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("GEN_DEFAULT_MAX_OUTPUT_TOKENS", "1200"))
GEN_MIN_OUTPUT_TOKENS = int(os.getenv("GEN_MIN_OUTPUT_TOKENS", "256"))

def set_clients(groq_client=None):
    """Set the shared Groq client from main.py"""
    global _groq_client
    if groq_client is not None:
        _groq_client = groq_client

def initialize_clients():
    """Initialize Groq client if not already set."""
    global _groq_client
    if _groq_client is None and GROQ_AVAILABLE:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                _groq_client = Groq(api_key=groq_api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize Groq client: {e}")
                _groq_client = None

async def run_model(model_id: str, prompt: str, req=None) -> str:
    """
    Execute a prompt on a Groq model.
    
    Args:
        model_id: Groq model identifier (groq-llama-8b, groq-llama-70b, etc.)
        prompt: The prompt text to send to the model
        req: Optional request object for token budget calculation
    
    Returns:
        Raw text output from the model
    """
    initialize_clients()
    
    if not GROQ_AVAILABLE or _groq_client is None:
        raise ValueError("Groq client not initialized. Please set GROQ_API_KEY environment variable.")

    model_map = {
        "groq-llama-8b": ("llama-3.1-8b-instant", 65536),
        "groq-llama-70b": ("llama-3.3-70b-versatile", 32768),
        "groq-qwen-32b": ("qwen/qwen3-32b", 32768),
        "groq-llama-guard": ("meta-llama/llama-guard-4-12b", 1024),
        "groq-gpt-oss-120b": ("openai/gpt-oss-120b", 65536),
        "groq-gpt-oss-20b": ("openai/gpt-oss-20b", 65536),
    }

    if model_id not in model_map:
        return f"<Question>Model '{model_id}' not found. Available: {', '.join(model_map.keys())}</Question><Answer>N/A</Answer>"

    groq_model, hard_cap_tokens = model_map[model_id]
    max_tokens = get_completion_token_budget(model_id, req=req)
    max_tokens = max(GEN_MIN_OUTPUT_TOKENS, min(max_tokens, hard_cap_tokens))

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: _groq_client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens
        )
    )
    return response.choices[0].message.content


def get_completion_token_budget(model_id: str, req=None) -> int:
    """Return a safer output-token budget to avoid oversized token-rate requests.

    Historical behavior requested full model caps (e.g. 32k), which can trip Groq
    token rate limits even for tiny prompts. This budget scales by request size and
    stays configurable via env vars.
    """
    if model_id == "groq-llama-guard":
        return 512

    n = 1
    if req is not None and hasattr(req, "num_questions"):
        try:
            n = max(1, int(getattr(req, "num_questions") or 1))
        except Exception:
            n = 1

    qtype = ""
    if req is not None and hasattr(req, "qType"):
        qtype = str(getattr(req, "qType") or "")

    suggested = 500 + (n * 350)
    if "Multiple Choice" in qtype:
        suggested += 120

    return int(max(GEN_MIN_OUTPUT_TOKENS, min(suggested, GEN_DEFAULT_MAX_OUTPUT_TOKENS)))
