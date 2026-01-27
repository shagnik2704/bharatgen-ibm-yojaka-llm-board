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
_tokenizer_moe = None
_model_moe = None

def set_clients(gemini_client=None, openai_client=None, groq_client=None, tokenizer_moe=None, model_moe=None):
    """Set the shared clients from main.py"""
    global _gemini_client, _openai_client, _groq_client, _tokenizer_moe, _model_moe
    if gemini_client is not None:
        _gemini_client = gemini_client
    if openai_client is not None:
        _openai_client = openai_client
    if groq_client is not None:
        _groq_client = groq_client
    if tokenizer_moe is not None:
        _tokenizer_moe = tokenizer_moe
    if model_moe is not None:
        _model_moe = model_moe

def initialize_clients():
    """Initialize model clients if not already set. Should be called once at startup."""
    global _gemini_client, _openai_client, _groq_client, _tokenizer_moe, _model_moe
    
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
    # Initialize Param-1-7B-MoE from local path
    if _tokenizer_moe is None or _model_moe is None:
        param_moe_path = os.getenv("PARAM1_7B_MOE_PATH")
        if param_moe_path:
            model_path = Path(param_moe_path)
            if model_path.exists():
                try:
                    print(f"Loading Param-1-7B-MoE from: {model_path}")
                    from transformers import BitsAndBytesConfig
                    _tokenizer_moe = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4"
                    )
                    _model_moe = AutoModelForCausalLM.from_pretrained(
                        str(model_path),
                        quantization_config=quant_config,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    print("Successfully loaded Param-1-7B-MoE")
                except Exception as e:
                    print(f"Warning: Failed to initialize Param-1-7B-MoE: {e}")
                    _tokenizer_moe = None
                    _model_moe = None
            else:
                print(f"Warning: Param-1-7B-MoE path does not exist: {model_path}")

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
    global _gemini_client, _openai_client, _tokenizer_moe, _model_moe
    
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
    elif model_id == "param-1-7b-moe":
        if _tokenizer_moe is None or _model_moe is None:
            raise ValueError("Param-1-7B-MoE model not initialized. Please set PARAM1_7B_MOE_PATH environment variable and ensure the model path exists.")
        loop = asyncio.get_event_loop()
        def generate():
            print(f"[DEBUG] Starting generation for param-1-7b-moe")
            print(f"[DEBUG] Prompt length: {len(prompt)} characters")
            
            # Tokenize with truncation to avoid issues (limit to 1500 tokens)
            max_input_tokens = 1500
            inputs = _tokenizer_moe(prompt, return_tensors="pt", return_token_type_ids=False, truncation=True, max_length=max_input_tokens)
            input_length = inputs['input_ids'].shape[1]
            print(f"[DEBUG] Input token length: {input_length}")
            
            # Move to device
            device = next(_model_moe.parameters()).device
            print(f"[DEBUG] Model device: {device}")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            print(f"[DEBUG] Starting model.generate()...")
            import time
            start_time = time.time()
            
            with torch.no_grad():
                try:
                    output = _model_moe.generate(
                        **inputs,
                        max_new_tokens=150,  # Reduced for faster generation
                        do_sample=False,  # Greedy decoding is faster
                        eos_token_id=_tokenizer_moe.eos_token_id,
                        use_cache=True,
                        pad_token_id=_tokenizer_moe.pad_token_id if _tokenizer_moe.pad_token_id is not None else _tokenizer_moe.eos_token_id,
                        repetition_penalty=1.1  # Help prevent repetition/stuck loops
                    )
                    elapsed = time.time() - start_time
                    print(f"[DEBUG] Generation complete in {elapsed:.2f}s. Output length: {output.shape[1]}")
                except Exception as e:
                    elapsed = time.time() - start_time
                    print(f"[ERROR] Generation failed after {elapsed:.2f}s: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            # Decode only the new tokens
            generated_tokens = output[0][input_length:]
            result = _tokenizer_moe.decode(generated_tokens, skip_special_tokens=True)
            print(f"[DEBUG] Decoded result length: {len(result)} characters")
            print(f"[DEBUG] First 500 chars of output: {result[:500]}")
            return result
        return await loop.run_in_executor(None, generate)
    elif model_id == "rag-piped-param-moe":
        if _tokenizer_moe is None or _model_moe is None:
            raise ValueError("Param-1-7B-MoE model not initialized. Please set PARAM1_7B_MOE_PATH environment variable and ensure the model path exists.")
        # Use provided context chunks or retrieve them
        if context_chunks:
            topic_chunk, theme_chunk = context_chunks
        else:
            # This shouldn't happen in practice, but fallback
            topic_chunk, theme_chunk = "", ""
        loop = asyncio.get_event_loop()
        def generate():
            print(f"[DEBUG] Starting generation for rag-piped-param-moe")
            print(f"[DEBUG] Prompt length: {len(prompt)} characters")
            
            # More aggressive truncation - Param models typically have 2048 context, but we need room for generation
            # Limit to 1500 tokens to leave room for generation
            max_input_tokens = 1500
            inputs = _tokenizer_moe(prompt, return_tensors="pt", return_token_type_ids=False, truncation=True, max_length=max_input_tokens)
            input_length = inputs['input_ids'].shape[1]
            print(f"[DEBUG] Input token length: {input_length} (truncated from original)")
            
            # Move to device
            device = next(_model_moe.parameters()).device
            print(f"[DEBUG] Model device: {device}")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            print(f"[DEBUG] Starting model.generate()...")
            import time
            start_time = time.time()
            
            with torch.no_grad():
                try:
                    # Use shorter generation with timeout protection
                    output = _model_moe.generate(
                        **inputs,
                        max_new_tokens=150,  # Further reduced for faster generation
                        do_sample=False,  # Greedy decoding is faster
                        eos_token_id=_tokenizer_moe.eos_token_id,
                        use_cache=True,
                        pad_token_id=_tokenizer_moe.pad_token_id if _tokenizer_moe.pad_token_id is not None else _tokenizer_moe.eos_token_id,
                        repetition_penalty=1.1  # Help prevent repetition/stuck loops
                    )
                    elapsed = time.time() - start_time
                    print(f"[DEBUG] Generation complete in {elapsed:.2f}s. Output length: {output.shape[1]}")
                except Exception as e:
                    elapsed = time.time() - start_time
                    print(f"[ERROR] Generation failed after {elapsed:.2f}s: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            # Decode only the new tokens
            generated_tokens = output[0][input_length:]
            result = _tokenizer_moe.decode(generated_tokens, skip_special_tokens=True)
            print(f"[DEBUG] Decoded result length: {len(result)} characters")
            print(f"[DEBUG] First 500 chars of output: {result[:500]}")
            return result
        return await loop.run_in_executor(None, generate)
    else:
        return "<Question>Model not found.</Question><Answer>N/A</Answer>"

def needs_rag(model_id: str) -> bool:
    """Check if a model requires RAG context."""
    return model_id in ["rag-piped-llama", "rag-piped-param-moe"]

def get_rag_context(chapter: str, theme: str) -> tuple:
    """Retrieve RAG context chunks for a given chapter and theme."""
    return ncert_rag.main(chapter, theme)
