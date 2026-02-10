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

# from gradio_client import Client
import re

def call_vllm(model_url, prompt: str,max_tokens=2048,context_chunks=None,req=None) -> str:
    # print(prompt)
    if(req):
        depth_str=""
        print(req.qType)
        if('level 1' in req.depth):
            depth_str="DOK 1 (Recall): Direct facts from the text. (e.g., 'What is...', 'Define...')"
        elif('level 2' in req.depth):
            depth_str="DOK 2 (Understand/Apply): Interpreting the text. (e.g., 'How does X affect Y?', 'Classify...')"
        elif('level 3' in req.depth):
            depth_str="DOK 3 (Analyze/Evaluate): Using the text to solve non-routine problems. (e.g., 'What would happen if...', 'Justify...')"
        elif('level 4' in req.depth):
            depth_str="DOK 4 (Create/Synthesis): Connecting this text to broader scientific/mathematical principles."
        question_text='Question Text'

        if('True' in req.qType):
            if(req.language=='hi'):
                question_text = "Question text starting with सही या गलत बताएं followed by the statement."
            else:
                question_text = "Question text starting with State True or False followed by the statement."
        elif('Fill' in req.qType):
            question_text = "Structure it as a sentence with a blank in it. Do not put options like an MCQ."
        elif('MCQ' in req.qType):
            question_text = "Question text along with all the four options A, B, C and D."
        
        system_prompt = f'''You are a helpful AI assistant. You think step-by-step. 
Generate {"an MCQ" if "MCQ" in req.qType else req.qType} type question. {"Structure it as a sentence with a blank in it. Do not put options like an MCQ." if 'Fill' in req.qType else "The question should conttain 4 options namely A,B, C and D along with the text." if 'MCQ' in req.qType else ""}
Generate each question in the following structure. Repeat this block for every question:
<Question>
{question_text}
</Question>
<Answer>
Correct answer with a 2-sentence explanation of the underlying concept]
</Answer>
'''
        if req.language == "hi":
            lang_block = (
               "Write all Questions and Answers in Hindi (Devanagari script), while preserving LaTeX/math notation as-is."
            )
        else:
            lang_block = (
                "Write all Questions and Answers in English."
            )
        user_prompt = f'''
{"Generate an MCQ with options in the question text." if 'MCQ' in req.qType else ""}
{lang_block}
SUBJECT: {req.subject}
CHAPTER: {req.chapter}
REQUIRED DEPTH: {depth_str}
QUANTITY: {req.num_questions}
{"Context chunk : "+context_chunks[0] if context_chunks else ""}
'''
        print("\n============System===============\n",system_prompt,"\n============END===============\n")
        print("\n============User===============\n",user_prompt,"\n============END===============\n")
        
        url = 'https://api.bharatgen.dev/v1/chat/completions'
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer bharatgen-secret-token-123" }
        payload = {
            "model": "param-17B SFT S1",
            "temperature": 0,
            "max_length": 2048,
            "chat_template_kwargs": {
                "enable_thinking": True
            },
            "messages": [
                {
                "role": "system",
                "content": system_prompt
                },
                {
                "role": "user",
                "content": user_prompt
                }
            ]
            }
        resp = requests.post(url, headers=headers, json=payload, verify=False)
    else:   
        # client = Client("https://1df79b03590242911b.gradio.live/")
        # result = client.predict(
        #     message=user_prompt,
        #     history=[],
        #     system_prompt=system_prompt,
        #     temp=0.7,
        #     max_tok=2048,
        #     top_p=0.9,
        #     top_k=50,
        #     api_name="/chat_fn"
        # )
        # result = result[0][-1]['content'][-1]['text']
        # try:
        #     result = re.sub(r"<think>[\s\S]*?</think>", "", result)
        #     result = result.split("<details style='margin-top:10px; font-size:0.85em; opacity:0.6;'><summary>🔍 Debug: Raw Response</summary>")[-1]
        # except Exception as e:
        #     print(e)
        #     pass 
        # print("\n============START===============\n",result,"\n============END===============\n")
        # return result
        data = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                }
        resp = requests.post(model_url, json=data, verify=False)
        print(resp)
    try:
        resp=resp.json()
        resp=resp['choices'][0]['message']['content']
        remove_think = lambda s: re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
        resp=remove_think(resp)
    except Exception as e:
        print("Exception : ",e)
        resp = "<Question>The model generated thoughts, but not words. 🤔</Question> \n <Answer>Thinking...🤔</Answer>"
    return resp.strip()

async def run_model(model_id: str, prompt: str, context_chunks: tuple = None, req=None) -> str:
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
        "groq-Qwen3-32B":('qwen/qwen3-32b',32768),
        "Qwen3-32B":(" https://model-serve-qwen3-32b.impactsummit.nxtgen.cloud/v1/chat/completions",32768),
        "Param-17B":("https://param5b.impactsummit.nxtgen.cloud/v1/chat/completions",2048),
        "rag-piped-groq-70b": ("llama-3.3-70b-versatile", 32768),
        "groq-llama-guard": ("meta-llama/llama-guard-4-12b", 1024),
        "groq-gpt-oss-120b": ("openai/gpt-oss-120b", 65536),
        "groq-gpt-oss-20b": ("openai/gpt-oss-20b", 65536),
    }
    
    if(('Qwen' in model_id or 'Param' in model_id) and 'groq' not in model_id):
        url,token_limit = model_map[model_id]
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: call_vllm(url, prompt + ("\nHere's some context on the topic : \n"+context_chunks[0] if(context_chunks) else ""),max_tokens=token_limit,context_chunks=context_chunks,req=req)
        )
    if model_id not in model_map:
        return f"<Question>Model '{model_id}' not found. Available: {', '.join(model_map.keys())}</Question><Answer>N/A</Answer>"

    groq_model, max_tokens = model_map[model_id]
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: _groq_client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt+ ("\nHere's some context on the topic : \n"+context_chunks[0] if(context_chunks) else "")}],
            temperature=0.7,
            max_tokens=max_tokens
        )
    )
    return response.choices[0].message.content

def needs_rag(model_id: str) -> bool:
    """Check if a model requires RAG context."""
    # return model_id in ["rag-piped-llama", "rag-piped-param-instruct", "rag-piped-groq-70b"]
    return True
    # return model_id == "rag-piped-groq-70b"

def get_rag_context(subject:str, class_level:str, chapter: str, theme: str, language: str = "en") -> tuple:
    """
    Retrieve RAG context chunks for a given chapter and theme, scoped by language.
    Returns (topic_chunk, theme_chunk, topic_meta, theme_meta).
    """
    prompt = f"{chapter}"
    print(prompt)
    return ncert_rag.main_ibm(prompt, language = language, subject=subject, class_level=class_level)
