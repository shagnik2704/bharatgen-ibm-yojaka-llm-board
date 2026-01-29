"""
Model runner module - handles execution of different LLM models.
Extracted from main.py to be reusable by both single-model and council flows.
"""
import os
import ollama
import torch
from google import genai
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from typing import Optional
import asyncio
import ncert_rag_pipe.main as ncert_rag

# Try to import Groq (optional)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# Global clients - will be set by main.py
_gemini_client = None
_openai_client = None
_groq_client = None
_tokenizer_29 = None
_model_29 = None

def set_clients(gemini_client=None, openai_client=None, groq_client=None, tokenizer_29=None, model_29=None):
    """Set the shared clients from main.py"""
    global _gemini_client, _openai_client, _groq_client, _tokenizer_29, _model_29
    if gemini_client is not None:
        _gemini_client = gemini_client
    if openai_client is not None:
        _openai_client = openai_client
    if groq_client is not None:
        _groq_client = groq_client
    if tokenizer_29 is not None:
        _tokenizer_29 = tokenizer_29
    if model_29 is not None:
        _model_29 = model_29

def initialize_clients():
    """Initialize model clients if not already set. Should be called once at startup."""
    global _gemini_client, _openai_client, _groq_client, _tokenizer_29, _model_29
    
    if _gemini_client is None:
        gemini_api_key = os.getenv("GEMINI_API_KEY_21")
        if gemini_api_key:
            try:
                _gemini_client = genai.Client(api_key=gemini_api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize Gemini client: {e}")
                _gemini_client = None
    if _openai_client is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            try:
                _openai_client = OpenAI(api_key=openai_api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")
                _openai_client = None
    if _groq_client is None and GROQ_AVAILABLE:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                _groq_client = Groq(api_key=groq_api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize Groq client: {e}")
                _groq_client = None
    # Initialize Param-1-2.9B-Instruct (HuggingFace: bharatgenai/Param-1-2.9B-Instruct)
    if _tokenizer_29 is None or _model_29 is None:
        model_29_id = os.getenv("PARAM1_2_9B_INSTRUCT_MODEL", "bharatgenai/Param-1-2.9B-Instruct")
        use_4bit = os.getenv("PARAM_2_9B_4BIT", "").lower() in ("1", "true", "yes")
        try:
            print(f"Loading Param-1-2.9B-Instruct from: {model_29_id}")
            _tokenizer_29 = AutoTokenizer.from_pretrained(model_29_id, trust_remote_code=False)
            load_kw = {"device_map": "auto", "trust_remote_code": True}
            if use_4bit:
                from transformers import BitsAndBytesConfig
                load_kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            _model_29 = AutoModelForCausalLM.from_pretrained(model_29_id, **load_kw)
            print("Successfully loaded Param-1-2.9B-Instruct")
        except Exception as e:
            print(f"Warning: Failed to initialize Param-1-2.9B-Instruct: {e}")
            _tokenizer_29 = None
            _model_29 = None

async def run_model(model_id: str, prompt: str, context_chunks: tuple = None) -> str:
    """
    Execute a prompt on a specified model.
    
    Args:
        model_id: The model identifier (e.g., "gemini", "chatgpt", "local-llama", etc.)
        prompt: The prompt text to send to the model
        context_chunks: Optional tuple of (topic_chunk, theme_chunk) for RAG models
    
    Returns:
        Raw text output from the model
    """
    global _gemini_client, _openai_client, _tokenizer_29, _model_29
    
    # Ensure clients are initialized
    initialize_clients()
    
    if model_id == "gemini":
        if _gemini_client is None:
            raise ValueError("Gemini client not initialized. Please set GEMINI_API_KEY_21 environment variable.")
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: _gemini_client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
        )
        return response.text
    elif model_id == "chatgpt":
        if _openai_client is None:
            raise ValueError("OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: _openai_client.chat.completions.create(
                model="gpt-4o", 
                messages=[{"role": "user", "content": prompt}]
            )
        )
        return response.choices[0].message.content
    elif model_id == "groq-llama-8b":
        if not GROQ_AVAILABLE:
            raise ValueError("Groq library not installed. Please install it with: pip install groq")
        if _groq_client is None:
            raise ValueError("Groq client not initialized. Please set GROQ_API_KEY environment variable.")
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: _groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=65536
            )
        )
        return response.choices[0].message.content
    elif model_id == "groq-llama-70b":
        if not GROQ_AVAILABLE:
            raise ValueError("Groq library not installed. Please install it with: pip install groq")
        if _groq_client is None:
            raise ValueError("Groq client not initialized. Please set GROQ_API_KEY environment variable.")
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: _groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=32768
            )
        )
        return response.choices[0].message.content
    elif model_id == "rag-piped-groq-70b":
        # RAG-piped Groq 70B: same API, prompt already contains RAG context from council/main
        if not GROQ_AVAILABLE:
            raise ValueError("Groq library not installed. Please install it with: pip install groq")
        if _groq_client is None:
            raise ValueError("Groq client not initialized. Please set GROQ_API_KEY environment variable.")
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: _groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=32768
            )
        )
        return response.choices[0].message.content
    elif model_id == "groq-llama-guard":
        if not GROQ_AVAILABLE:
            raise ValueError("Groq library not installed. Please install it with: pip install groq")
        if _groq_client is None:
            raise ValueError("Groq client not initialized. Please set GROQ_API_KEY environment variable.")
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: _groq_client.chat.completions.create(
                model="meta-llama/llama-guard-4-12b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024
            )
        )
        return response.choices[0].message.content
    elif model_id == "groq-gpt-oss-120b":
        if not GROQ_AVAILABLE:
            raise ValueError("Groq library not installed. Please install it with: pip install groq")
        if _groq_client is None:
            raise ValueError("Groq client not initialized. Please set GROQ_API_KEY environment variable.")
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: _groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=65536
            )
        )
        return response.choices[0].message.content
    elif model_id == "groq-gpt-oss-20b":
        if not GROQ_AVAILABLE:
            raise ValueError("Groq library not installed. Please install it with: pip install groq")
        if _groq_client is None:
            raise ValueError("Groq client not initialized. Please set GROQ_API_KEY environment variable.")
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: _groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=65536
            )
        )
        return response.choices[0].message.content
    elif model_id == "local-llama":
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        )
        return response['message']['content']
    elif model_id == "qwen":
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: ollama.chat(model='qwen3:8b', messages=[{'role': 'user', 'content': prompt}])
        )
        return response['message']['content']
    elif model_id == "granite3.3:8b":
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: ollama.chat(model='granite3.3:8b', messages=[{'role': 'user', 'content': prompt}])
        )
        return response['message']['content']
    elif model_id == "rag-piped-llama":
        # Use provided context chunks or retrieve them
        if context_chunks:
            topic_chunk, theme_chunk = context_chunks
        else:
            # This shouldn't happen in practice, but fallback
            topic_chunk, theme_chunk = "", ""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        )
        return response['message']['content']
    elif model_id == "param-1-2.9b-instruct" or model_id == "rag-piped-param-instruct":
        if _tokenizer_29 is None or _model_29 is None:
            raise ValueError(
                "Param-1-2.9B-Instruct not loaded. Set PARAM1_2_9B_INSTRUCT_MODEL (optional) and ensure "
                "transformers can fetch bharatgenai/Param-1-2.9B-Instruct."
            )
        loop = asyncio.get_event_loop()

        def _generate():
            import time
            user_content = prompt
            if len(prompt) > 6000:
                user_content = prompt[:6000] + "\n\n[truncated]"
            conversation = [
                {"role": "system", "content": "You are an expert Academic Assessment Designer. Follow instructions carefully and output only what is asked."},
                {"role": "user", "content": user_content},
            ]
            inputs = _tokenizer_29.apply_chat_template(
                conversation,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            input_length = inputs.shape[1]
            device = next(_model_29.parameters()).device
            inputs = inputs.to(device)
            pad_id = _tokenizer_29.pad_token_id
            if pad_id is None:
                pad_id = _tokenizer_29.eos_token_id
            t0 = time.time()
            with torch.no_grad():
                out = _model_29.generate(
                    inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.6,
                    eos_token_id=_tokenizer_29.eos_token_id,
                    pad_token_id=pad_id,
                    use_cache=False,
                )
            elapsed = time.time() - t0
            gen = out[0][input_length:]
            result = _tokenizer_29.decode(gen, skip_special_tokens=True)
            print(f"[DEBUG] Param-2.9B-Instruct generation done in {elapsed:.2f}s, {len(result)} chars")
            return result.strip()

        return await loop.run_in_executor(None, _generate)
    else:
        return "<Question>Model not found.</Question><Answer>N/A</Answer>"

def needs_rag(model_id: str) -> bool:
    """Check if a model requires RAG context."""
    return model_id in ["rag-piped-llama", "rag-piped-param-instruct", "rag-piped-groq-70b"]

def get_rag_context(chapter: str, theme: str, language: str = "en") -> tuple:
    """Retrieve RAG context chunks for a given chapter and theme, scoped by language."""
    return ncert_rag.main(chapter, theme, language=language)
