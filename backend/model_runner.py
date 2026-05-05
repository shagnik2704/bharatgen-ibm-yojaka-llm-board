"""
Model runner module - handles execution of Groq LLM models only.
Unified RAG retrieval is handled separately by rag_retriever module.
"""
import asyncio
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Determine the active provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

try:
    from openai import AsyncOpenAI
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    AsyncOpenAI = None

_groq_client = None
_ollama_client = None

GEN_DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("GEN_DEFAULT_MAX_OUTPUT_TOKENS", "8192"))
GEN_MIN_OUTPUT_TOKENS = int(os.getenv("GEN_MIN_OUTPUT_TOKENS", "256"))
MODEL_REQUEST_TIMEOUT_S = float(os.getenv("MODEL_REQUEST_TIMEOUT_S", "120"))

def set_clients(groq_client=None):
    global _groq_client
    if groq_client is not None:
        _groq_client = groq_client

def initialize_clients():
    global _groq_client, _ollama_client
    
    # Initialize Groq if needed
    if _groq_client is None and GROQ_AVAILABLE and LLM_PROVIDER == "groq":
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                _groq_client = Groq(api_key=groq_api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize Groq client: {e}")
                
    # Initialize Ollama if needed
    if _ollama_client is None and OLLAMA_AVAILABLE and LLM_PROVIDER == "ollama":
        try:
            # Ollama requires an API key string, but it doesn't validate it.
            _ollama_client = AsyncOpenAI(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key="ollama" 
            )
        except Exception as e:
            print(f"Warning: Failed to initialize Ollama client: {e}")

async def run_model(model_id: str, prompt: str, req=None, temperature: Optional[float] = None) -> str:
    initialize_clients()
    
    # Updated Model Map containing your specific Ollama models
    model_map = {
        # Groq Models
        "groq-llama-8b": ("llama-3.1-8b-instant", 65536),
        "groq-llama-70b": ("llama-3.3-70b-versatile", 32768),
        "groq-qwen-32b": ("qwen/qwen3-32b", 32768),
        "groq-llama-guard": ("meta-llama/llama-guard-4-12b", 1024),
        "groq-gpt-oss-120b": ("openai/gpt-oss-120b", 65536),
        "groq-gpt-oss-20b": ("openai/gpt-oss-20b", 65536),
        
        # Ollama Models
        "ollama-gemma4-e4b": ("gemma4:e4b", 8192),
        "ollama-olmo-3-7b": ("olmo-3:7b", 8192),
        "ollama-phi4-mini": ("phi4-mini:3.8b", 8192),
        "ollama-qwen-2b": ("qwen3.5:2b", 8192),
        "ollama-gemma4-e2b": ("gemma4:e2b", 8192),
        "ollama-qwen-4b": ("qwen3.5:4b", 8192),
        "ollama-gemma4-31b": ("gemma4:31b", 8192),
    }

    if model_id not in model_map:
        return f"<Question>Model '{model_id}' not found. Available: {', '.join(model_map.keys())}</Question><Answer>N/A</Answer>"

    target_model, hard_cap_tokens = model_map[model_id]
    max_tokens = get_completion_token_budget(model_id, req=req)
    max_tokens = max(GEN_MIN_OUTPUT_TOKENS, min(max_tokens, hard_cap_tokens))

    # Temperature fallback logic
    temp = 0.7
    if temperature is not None:
        try: temp = float(temperature)
        except Exception: pass
    elif req is not None and hasattr(req, "temperature"):
        try: temp = float(getattr(req, "temperature"))
        except Exception: pass
    temp = max(0.0, min(1.0, temp))

    # Route request based on active provider
    if LLM_PROVIDER == "ollama":
        if _ollama_client is None:
            raise ValueError("Ollama client not initialized. Ensure openai package is installed and URL is correct.")

        try:
            response = await asyncio.wait_for(
                _ollama_client.chat.completions.create(
                    model=target_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=max_tokens
                ),
                timeout=MODEL_REQUEST_TIMEOUT_S,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Ollama request timed out after {MODEL_REQUEST_TIMEOUT_S}s for model '{target_model}'. "
                "Check whether Ollama is reachable, the model is loaded, or increase MODEL_REQUEST_TIMEOUT_S."
            ) from exc
        return response.choices[0].message.content
        
    else:  # Default to Groq
        if _groq_client is None:
            raise ValueError("Groq client not initialized. Please set GROQ_API_KEY.")
            
        loop = asyncio.get_event_loop()
        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: _groq_client.chat.completions.create(
                        model=target_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temp,
                        max_tokens=max_tokens
                    )
                ),
                timeout=MODEL_REQUEST_TIMEOUT_S,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Groq request timed out after {MODEL_REQUEST_TIMEOUT_S}s for model '{target_model}'. "
                "Check connectivity or increase MODEL_REQUEST_TIMEOUT_S."
            ) from exc
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

    suggested = 6000 + (n * 350)
    if "Multiple Choice" in qtype:
        suggested += 120

    return int(max(GEN_MIN_OUTPUT_TOKENS, min(suggested, GEN_DEFAULT_MAX_OUTPUT_TOKENS)))
