# import os
# import json # Added to parse the string response
# import uvicorn
# import ollama
# import re
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from google import genai
# from openai import OpenAI
# from pathlib import Path
# import NCERT_RAG_PIPE.main as ncert_rag

# BASE_DIR = Path(__file__).resolve().parent

# load_dotenv()
# app = FastAPI()

# # Clients
# gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_1"))
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# class QueryRequest(BaseModel):
#     model_id: str
#     depth: str
#     theme: str
#     topic: str
#     qType: str

# def parse_ai_output(raw_text):
#     # 1. Check if raw_text is actually a string
#     if raw_text is None:
#         return {"question": "Error: Model returned no data.", "answer": "Check local server status."}
    
#     # 2. Extract tags
#     question_match = re.search(r'<Question>(.*?)</Question>', raw_text, re.DOTALL)
#     answer_match = re.search(r'<Answer>(.*?)</Answer>', raw_text, re.DOTALL)

#     # 3. Fallback: If tags aren't found, give the user the raw output in the question box
#     # instead of just crashing or saying "Not Found"
#     question = question_match.group(1).strip() if question_match else raw_text.strip()
#     answer = answer_match.group(1).strip() if answer_match else "No tags detected. Logic may be embedded in text above."

#     return {
#         "question": question,
#         "answer": answer
#     }

# @app.get("/")
# async def serve_index():
#     return FileResponse(BASE_DIR / "../frontend/index.html")

# @app.post("/ask")
# async def ask_llm(req: QueryRequest):
#     # The strictly defined prompt for diverse questions
#     # Refined implementation for your code
#     topic_chunk, theme_chunk  = ncert_rag.main(req.theme, req.topic)
#     prompt = (
#         "### ROLE\n"
#         "Act as an expert Academic Assessment Designer specializing in curriculum development.\n\n"
        
#         "### TASK\n"
#         f"Generate a high-quality question based on:\n"
#         f"- QUESTION TYPE: {req.qType}\n"
#         f"- TOPIC: {req.topic}\n"
#         f"- THEME: {req.theme}\n"
#         f"- DEPTH: {req.depth}\n\n"
        
#         "### CONSTRAINTS\n"
#         "1. If depth is 'High' or 'Application', include a situational scenario.\n"
#         "2. Ensure distractors are plausible and logically related.\n"
#         "3. Use professional academic language.\n\n"
        
#         "### OUTPUT FORMAT\n"
#         "Strictly wrap the content in these tags:\n"
#         "<Question> [Question text and options] </Question>\n"
#         "<Answer> [Correct answer] </Answer>"
#     )
#     prompt2 = (
#         "### ROLE\n"
#         "Act as an expert Academic Assessment Designer specializing in psychometrics and Evidence-Centered Design. "
#         "Your goal is to synthesize specific reference materials into a high-validity assessment item.\n\n"
        
#         "### INPUT PARAMETERS\n"
#         f"- QUESTION TYPE: {req.qType}\n"
#         f"- TOPIC: {req.topic}\n"
#         f"- THEME: {req.theme}\n"
#         f"- COGNITIVE DEPTH: {req.depth}\n\n"
        
#         "### REFERENCE MATERIAL (RAG CONTEXT)\n"
#         "The following chunks are provided as potential source material. "
#         "**Crucial:** Evaluate these chunks for relevance to the TOPIC and THEME. "
#         "If a chunk is relevant, prioritize its details. If a chunk is irrelevant or contains 'noise,' ignore it and rely on your internal expert knowledge to maintain academic accuracy.\n\n"
#         f"--- [TOPIC CHUNK] ---\n{topic_chunk}\n--- [END TOPIC CHUNK] ---\n\n"
#         f"--- [THEME CHUNK] ---\n{theme_chunk}\n--- [END THEME CHUNK] ---\n\n"
        
#         "### DESIGN CONSTRAINTS\n"
#         "1. **Relevance Assessment:** Before generating, determine if [TOPIC CHUNK] contains the core facts for the question and if [THEME CHUNK] provides a usable narrative setting. "
#         "If the chunks are not helpful for the requested {req.topic}, prioritize factual correctness over context adherence.\n"
#         "2. **Context Integration:** Where relevant, the question should assess the [TOPIC CHUNK] principles while using the [THEME CHUNK] as the situational backdrop.\n"
#         "3. **Depth Handling:**\n"
#         "   - If depth is 'Low/Recall': Focus on defining terms or facts. Use the [TOPIC CHUNK] if it contains the definitions; otherwise, use standard academic definitions.\n"
#         "   - If depth is 'High/Application': Create a situational scenario. Use the [THEME CHUNK] to build the 'story' of the scenario ONLY if it aligns with the {req.topic}.\n"
#         "4. **Distractor Quality:** Distractors must be 'plausible distractors'—common misconceptions related to the topic, not random or obviously wrong choices.\n"
#         "5. **Professionalism:** Use formal academic language and ensure the question is clear and unambiguous.\n\n"
        
#         "### OUTPUT FORMAT\n"
#         "Strictly wrap the content in these tags. Do not include introductory text, reasoning, or meta-comments on context relevance:\n"
#         "<Question>\n"
#         "[Question Stem/Scenario]\n"
#         "[A. Option]\n"
#         "[B. Option]\n"
#         "[C. Option]\n"
#         "[D. Option]\n"
#         "</Question>\n"
#         "<Answer> [Correct Answer Letter and Text] </Answer>"
#     )
#     try:
#         raw_output = ""
        
#         if req.model_id == "gemini":
#             response = gemini_client.models.generate_content(
#                 model="gemini-3-flash-preview", 
#                 contents=prompt
#             )
#             raw_output = response.text
        
#         elif req.model_id == "local-llama":
#             response = ollama.chat(model='llama3', messages=[
#                 {
#                     'role': 'user',
#                     'content': prompt,
#                 },
#             ])
#             raw_output = response['message']['content']

#         elif req.model_id == "qwen":
#             response = ollama.chat(model='qwen3:8b', messages=[
#                 {'role': 'user', 'content': prompt}
#             ])
#             raw_output = response['message']['content']

#         elif req.model_id == "rag-piped-llama":
#             response = ollama.chat(model='llama3', messages=[
#                 {
#                     'role': 'user',
#                     'content': prompt2,
#                 },
#             ])
#             raw_output = response['message']['content']

#         elif req.model_id == "granite3.3:8b":
#             response = ollama.chat(model='granite3.3:8b', messages=[
#                 {'role': 'user', 'content': prompt}
#             ])
#             raw_output = response['message']['content']
        
#         else:
#             raw_output = "<Question> Local Mode Question <Question> <Answer> Local Answer <Answer>"

#         # Return the raw text directly in the 'question' key
#         # return {"question": raw_output}
#         formatted_response = parse_ai_output(raw_output)
#         return formatted_response

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


import os
import re
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from google import genai
from openai import OpenAI
from dotenv import load_dotenv
import NCERT_RAG_PIPE.main as ncert_rag

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
load_dotenv()
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent

# Clients
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_21"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# class QueryRequest(BaseModel):
#     model_id: str
#     depth: str
#     subject: str
#     chapter: str
#     topic: str
#     qType: str

# def parse_ai_output(raw_text):
#     if not raw_text:
#         return {"question": "Error: No data.", "answer": "N/A"}
#     question_match = re.search(r'<Question>(.*?)</Question>', raw_text, re.DOTALL)
#     answer_match = re.search(r'<Answer>(.*?)</Answer>', raw_text, re.DOTALL)
#     return {
#         "question": question_match.group(1).strip() if question_match else raw_text.strip(),
#         "answer": answer_match.group(1).strip() if answer_match else "Logic embedded in text."
#     }

# @app.get("/")
# async def serve_index():
#     return FileResponse(BASE_DIR / "../frontend/index.html")

# @app.post("/ask")
# async def ask_llm(req: QueryRequest):
#     # RAG Context Retrieval
    
    
#     # Standard Prompt
#     prompt = f"### ROLE: NCERT {req.subject} Expert. TASK: Generate a {req.qType} for {req.topic} (Chapter: {req.chapter}). DOK: {req.depth}. OUTPUT: <Question>...</Question><Answer>...</Answer>"
    
#     try:
#         raw_output = ""
#         if req.model_id == "gemini":
#             response = gemini_client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
#             raw_output = response.text
#         elif req.model_id == "chatgpt":
#             response = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
#             raw_output = response.choices[0].message.content
#         elif req.model_id == "local-llama":
#             response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
#             raw_output = response['message']['content']
#         elif req.model_id == "qwen":
#             response = ollama.chat(model='qwen3:8b', messages=[{'role': 'user', 'content': prompt}])
#             raw_output = response['message']['content']
#         elif req.model_id == "granite3.3:8b":
#             response = ollama.chat(model='granite3.3:8b', messages=[{'role': 'user', 'content': prompt}])
#             raw_output = response['message']['content']
#         elif req.model_id == "rag-piped-llama":
#             topic_chunk, theme_chunk = ncert_rag.main(req.chapter, req.topic)
#             # RAG-Specific Prompt (for rag-piped-llama)
#             prompt_rag = f"### RAG CONTEXT:\n{topic_chunk}\n\n### TASK: Generate {req.qType} for {req.topic} using context. DOK: {req.depth}. OUTPUT: <Question>...</Question><Answer>...</Answer>"
#             response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt_rag}])
#             raw_output = response['message']['content']
#         else:
#             raw_output = "<Question>Model not found.</Question><Answer>N/A</Answer>"

#         return parse_ai_output(raw_output)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
#     # ... (existing imports)

class QueryRequest(BaseModel):
    model_id: str
    depth: str
    subject: str
    chapter: str
    topic: str
    qType: str
    num_questions: int # Added this field

def parse_ai_output(raw_text):
    if not raw_text:
        return []

    # Regex explanation:
    # (?i) makes it case-insensitive
    # <question> matches the opening tag
    # (.*?) non-greedily captures everything until...
    # </question> the closing tag
    q_pattern = re.compile(r'<(?:[Qq]uestion)>(.*?)</(?:[Qq]uestion)>', re.DOTALL)
    a_pattern = re.compile(r'<(?:[Aa]nswer)>(.*?)</(?:[Aa]nswer)>', re.DOTALL)

    questions = q_pattern.findall(raw_text)
    answers = a_pattern.findall(raw_text)

    results = []
    for i in range(len(questions)):
        # Clean up any leftover markdown headers the AI might have snuck inside the tags
        clean_q = re.sub(r'(\*\*Question \d+\*\*|Question \d+:)', '', questions[i]).strip()
        clean_a = re.sub(r'(\*\*Answer\*\*|Answer:)', '', answers[i]).strip() if i < len(answers) else "No answer provided."

        results.append({
            "question": clean_q,
            "answer": clean_a
        })

    return results


@app.get("/")
async def serve_index():
    return FileResponse(BASE_DIR / os.getenv("FRONTEND_RELATIVE_PATH", ))

@app.post("/ask")
async def ask_llm(req: QueryRequest):
    # Your requested prompt structure
    prompt = (
        "### ROLE\n"
        "Act as an expert Academic Assessment Designer specializing in NCERT/CBSE curriculum development. "
        "Your goal is to create questions that move beyond simple memory and test true cognitive depth.\n\n"

        "### COGNITIVE DEPTH CONTEXT (Bloom's Taxonomy x DOK)\n"
        "You must adhere to the following definitions for the requested DEPTH:\n"
        "- DOK 1 (Recall/Remember): Recall of a fact, term, or property. (e.g., Define, List, State)\n"
        "- DOK 2 (Skills & Concepts/Understand & Apply): Use of information or conceptual knowledge. (e.g., Describe, Classify, Solve routine problems)\n"
        "- DOK 3 (Strategic Thinking/Analyze & Evaluate): Reasoning, planning, and using evidence. (e.g., Explain why, Non-routine problem solving, Compare/Contrast phenomena)\n"
        "- DOK 4 (Extended Thinking/Create): Complex synthesis and connection across chapters. (e.g., Create a model, Design an experiment, Critique a theoretical framework)\n\n"

        "### PARAMETERS\n"
        f"- SUBJECT: {req.subject}\n"
        f"- CHAPTER: {req.chapter}\n"
        f"- TOPIC: {req.topic}\n"
        f"- QUESTION TYPE: {req.qType}\n"
        f"- TARGET DEPTH: {req.depth}\n"
        f"- QUANTITY: {req.num_questions}\n\n"

        "### CONSTRAINTS\n"
        "1. Content must be strictly based on NCERT syllabus standards.\n"
        "2. Distractors for MCQs must be 'Common Misconceptions'—they should look correct to a student who has not understood the core concept.\n"
        "3. For numericals, provide a step-by-step logical breakdown in the Answer section.\n"
        "4. Use LaTeX for all mathematical formulas and chemical equations (e.g., $E=mc^2$).\n\n"

        "### OUTPUT FORMAT (FOLLOW EXACTLY)\n"
        "Generate each question in the following structure. Repeat this block for every question:\n"
        "<Question>\n[Question text here. If MCQ, include options A, B, C, D]\n</Question>\n"
        "<Answer>\n[Correct answer with a 2-sentence explanation of the underlying concept]\n</Answer>"
    )

    try:
        raw_output = ""
        if req.model_id == "gemini":
            response = gemini_client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
            raw_output = response.text
        elif req.model_id == "chatgpt":
            response = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
            raw_output = response.choices[0].message.content
        elif req.model_id == "local-llama":
            response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
            raw_output = response['message']['content']
        elif req.model_id == "qwen":
            response = ollama.chat(model='qwen3:8b', messages=[{'role': 'user', 'content': prompt}])
            raw_output = response['message']['content']
        elif req.model_id == "granite3.3:8b":
            response = ollama.chat(model='granite3.3:8b', messages=[{'role': 'user', 'content': prompt}])
            raw_output = response['message']['content']
        elif req.model_id == "rag-piped-llama":
            topic_chunk, theme_chunk = ncert_rag.main(req.chapter, req.topic)
            # RAG-Specific Prompt (for rag-piped-llama)
            prompt_rag = (
                "### ROLE\n"
                "Act as an expert NCERT Assessment Designer. Your task is to use the provided 'Source Material' "
                "to generate high-quality questions. You must strictly adhere to the requested Cognitive Depth.\n\n"

                "### SOURCE MATERIAL (RAG CONTEXT)\n"
                f"{topic_chunk}\n\n"

                "### COGNITIVE DEPTH FRAMEWORK (Bloom's x DOK)\n"
                "If the source material is simple, you must still elevate the question to meet these levels:\n"
                "- DOK 1 (Recall): Direct facts from the text. (e.g., 'What is...', 'Define...')\n"
                "- DOK 2 (Understand/Apply): Interpreting the text. (e.g., 'How does X affect Y?', 'Classify...')\n"
                "- DOK 3 (Analyze/Evaluate): Using the text to solve non-routine problems. (e.g., 'What would happen if...', 'Justify...')\n"
                "- DOK 4 (Create/Synthesis): Connecting this text to broader scientific/mathematical principles.\n\n"

                "### SESSION PARAMETERS\n"
                f"- SUBJECT: {req.subject}\n"
                f"- CHAPTER: {req.chapter}\n"
                f"- TOPIC: {req.topic}\n"
                f"- QUESTION TYPE: {req.qType}\n"
                f"- REQUIRED DEPTH: {req.depth}\n\n"

                "### INSTRUCTIONS\n"
                "1. Use the Source Material for factual accuracy. Do not hallucinate outside NCERT bounds.\n"
                "2. THE DEPTH IS PARAMOUNT: If the depth is DOK 3, do not provide a DOK 1 recall question even if the text is short.\n"
                "3. Use LaTeX for all technical notation (e.g., $H_2O$, $\sin(\theta)$).\n\n"

                "### OUTPUT FORMAT\n"
                "Strictly wrap each question and answer pair in these tags:\n"
                "<Question> [Text + Options if MCQ] </Question>\n"
                "<Answer> [Correct Answer + 1-sentence logic based on the Source Material] </Answer>"
            )
            response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt_rag}])
            raw_output = response['message']['content']
        elif req.model_id == "param.1:7b":
            model_name = BASE_DIR / os.getenv("PARAM1_7B_RELATIVE_PATH")
            print(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.bfloat32,
                device_map="auto"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.6,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False
                )
            raw_output = tokenizer.decode(output[0], skip_special_tokens=True)
        elif req.model_id == "rag-piped-param":
            topic_chunk, theme_chunk = ncert_rag.main(req.chapter, req.topic)
            # RAG-Specific Prompt (for rag-piped-param)
            prompt_rag = (
                "### ROLE\n"
                "Act as an expert NCERT Assessment Designer. Your task is to use the provided 'Source Material' "
                "to generate high-quality questions. You must strictly adhere to the requested Cognitive Depth.\n\n"

                "### SOURCE MATERIAL (RAG CONTEXT)\n"
                f"{topic_chunk}\n\n"

                "### COGNITIVE DEPTH FRAMEWORK (Bloom's x DOK)\n"
                "If the source material is simple, you must still elevate the question to meet these levels:\n"
                "- DOK 1 (Recall): Direct facts from the text. (e.g., 'What is...', 'Define...')\n"
                "- DOK 2 (Understand/Apply): Interpreting the text. (e.g., 'How does X affect Y?', 'Classify...')\n"
                "- DOK 3 (Analyze/Evaluate): Using the text to solve non-routine problems. (e.g., 'What would happen if...', 'Justify...')\n"
                "- DOK 4 (Create/Synthesis): Connecting this text to broader scientific/mathematical principles.\n\n"

                "### SESSION PARAMETERS\n"
                f"- SUBJECT: {req.subject}\n"
                f"- CHAPTER: {req.chapter}\n"
                f"- TOPIC: {req.topic}\n"
                f"- QUESTION TYPE: {req.qType}\n"
                f"- REQUIRED DEPTH: {req.depth}\n\n"

                "### INSTRUCTIONS\n"
                "1. Use the Source Material for factual accuracy. Do not hallucinate outside NCERT bounds.\n"
                "2. THE DEPTH IS PARAMOUNT: If the depth is DOK 3, do not provide a DOK 1 recall question even if the text is short.\n"
                "3. Use LaTeX for all technical notation (e.g., $H_2O$, $\sin(\theta)$).\n\n"

                "### OUTPUT FORMAT\n"
                "Strictly wrap each question and answer pair in these tags:\n"
                "<Question> [Text + Options if MCQ] </Question>\n"
                "<Answer> [Correct Answer + 1-sentence logic based on the Source Material] </Answer>"
            )
            inputs = tokenizer(prompt_rag, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.6,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False
                )
            raw_output = tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            raw_output = "<Question>Model not found.</Question><Answer>N/A</Answer>"
        print(raw_output+"\n")
        return parse_ai_output(raw_output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))