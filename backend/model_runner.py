"""
Model runner module - handles execution of Groq LLM models only.
"""
import os, requests, re
from typing import Optional
import asyncio
import ncert_rag_pipe.main as ncert_rag

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# Global Groq client - set by main.py
_groq_client = None

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

def call_vllm(model_url, prompt: str) -> str:
    print("Here ")
    data = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
            }

    resp = requests.post(model_url, json=data, verify=False)
    resp=resp.json()
    resp=resp['choices'][0]['message']['content']
    remove_think = lambda s: re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    resp=remove_think(resp)
    return resp.strip()

async def run_model(model_id: str, prompt: str, context_chunks: tuple = None) -> str:
    """
    Execute a prompt on a Groq model.
    
    Args:
        model_id: Groq model identifier (groq-llama-8b, groq-llama-70b, etc.)
        prompt: The prompt text to send to the model
        context_chunks: Optional tuple of (topic_chunk, theme_chunk) for RAG models
    
    Returns:
        Raw text output from the model
    """
    initialize_clients()
    
    if not GROQ_AVAILABLE or _groq_client is None:
        raise ValueError("Groq client not initialized. Please set GROQ_API_KEY environment variable.")

    model_map = {
        "groq-llama-8b": ("llama-3.1-8b-instant", 65536),
        "groq-llama-70b": ("llama-3.3-70b-versatile", 32768),
        "Qwen3-32B":("https://qwen32b.impactsummit.nxtgen.cloud/v1/chat/completions",0),
        "rag-piped-groq-70b": ("llama-3.3-70b-versatile", 32768),
        "groq-llama-guard": ("meta-llama/llama-guard-4-12b", 1024),
        "groq-gpt-oss-120b": ("openai/gpt-oss-120b", 65536),
        "groq-gpt-oss-20b": ("openai/gpt-oss-20b", 65536),
    }
    
    if('Qwen' in model_id):
        url,_ = model_map[model_id]
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: call_vllm(url, prompt)
        )
    if model_id not in model_map:
        return f"<Question>Model '{model_id}' not found. Available: {', '.join(model_map.keys())}</Question><Answer>N/A</Answer>"

    groq_model, max_tokens = model_map[model_id]
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

def needs_rag(model_id: str) -> bool:
    """Check if a model requires RAG context."""
    return model_id == "rag-piped-groq-70b"

def get_rag_context(chapter: str, theme: str, language: str = "en") -> tuple:
    """
    Retrieve RAG context chunks for a given chapter and theme, scoped by language.
    Returns (topic_chunk, theme_chunk, topic_meta, theme_meta).
    """
    return ncert_rag.main(chapter, theme, language=language)
