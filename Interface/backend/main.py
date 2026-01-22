import os
import json # Added to parse the string response
import uvicorn
import ollama
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from pathlib import Path
import NCERT_RAG_PIPE.main as ncert_rag

BASE_DIR = Path(__file__).resolve().parent

load_dotenv()
app = FastAPI()

# Clients
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_1"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class QueryRequest(BaseModel):
    model_id: str
    depth: str
    theme: str
    topic: str
    qType: str

def parse_ai_output(raw_text):
    # 1. Check if raw_text is actually a string
    if raw_text is None:
        return {"question": "Error: Model returned no data.", "answer": "Check local server status."}
    
    # 2. Extract tags
    question_match = re.search(r'<Question>(.*?)</Question>', raw_text, re.DOTALL)
    answer_match = re.search(r'<Answer>(.*?)</Answer>', raw_text, re.DOTALL)

    # 3. Fallback: If tags aren't found, give the user the raw output in the question box
    # instead of just crashing or saying "Not Found"
    question = question_match.group(1).strip() if question_match else raw_text.strip()
    answer = answer_match.group(1).strip() if answer_match else "No tags detected. Logic may be embedded in text above."

    return {
        "question": question,
        "answer": answer
    }

@app.get("/")
async def serve_index():
    return FileResponse(BASE_DIR / "../frontend/index.html")

@app.post("/ask")
async def ask_llm(req: QueryRequest):
    # The strictly defined prompt for diverse questions
    # Refined implementation for your code
    topic_chunk, theme_chunk = ncert_rag.main(req.theme, req.topic)
    prompt = (
        "### ROLE\n"
        "Act as an expert Academic Assessment Designer specializing in curriculum development.\n\n"
        
        "### TASK\n"
        f"Generate a high-quality question based on:\n"
        f"- QUESTION TYPE: {req.qType}\n"
        f"- TOPIC: {req.topic}\n"
        f"- THEME: {req.theme}\n"
        f"- DEPTH: {req.depth}\n\n"
        
        "### CONSTRAINTS\n"
        "1. If depth is 'High' or 'Application', include a situational scenario.\n"
        "2. Ensure distractors are plausible and logically related.\n"
        "3. Use professional academic language.\n\n"
        
        "### OUTPUT FORMAT\n"
        "Strictly wrap the content in these tags:\n"
        "<Question> [Question text and options] </Question>\n"
        "<Answer> [Correct answer] </Answer>"
    )
    prompt2 = (
        "### ROLE\n"
        "Act as an expert Academic Assessment Designer specializing in psychometrics and Evidence-Centered Design. "
        "Your goal is to synthesize specific reference materials into a high-validity assessment item.\n\n"
        
        "### INPUT PARAMETERS\n"
        f"- QUESTION TYPE: {req.qType}\n"
        f"- TOPIC: {req.topic}\n"
        f"- THEME: {req.theme}\n"
        f"- COGNITIVE DEPTH: {req.depth}\n\n"
        
        "### REFERENCE MATERIAL (RAG CONTEXT)\n"
        "Use the following two content chunks as the EXCLUSIVE source of truth for the question content. "
        "Do not use outside knowledge if it contradicts these chunks.\n"
        f"--- [TOPIC CHUNK] ---\n{topic_chunk}\n--- [END TOPIC CHUNK] ---\n\n"
        f"--- [THEME CHUNK] ---\n{theme_chunk}\n--- [END THEME CHUNK] ---\n\n"
        
        "### DESIGN CONSTRAINTS\n"
        "1. **Context Integration:** The question must assess understanding of the [TOPIC CHUNK] while using the [THEME CHUNK] as the setting or application context.\n"
        "2. **Depth Handling:**\n"
        "   - If depth is 'Low/Recall': Focus on defining terms or stating facts found directly in the [TOPIC CHUNK].\n"
        "   - If depth is 'High/Application': Create a situational scenario. The scenario must be built using the [THEME CHUNK], requiring the student to apply [TOPIC CHUNK] principles to solve it.\n"
        "3. **Distractor Quality:** Distractors must be 'plausible distractors'—common misconceptions or partial truths derived from the text, not random errors.\n"
        "4. **Formatting:** Use clear academic phrasing.\n\n"
        
        "### OUTPUT FORMAT\n"
        "Strictly wrap the content in these tags. Do not include introductory text or reasoning:\n"
        "<Question>\n"
        "[Question Stem/Scenario]\n"
        "[A. Option]\n"
        "[B. Option]\n"
        "[C. Option]\n"
        "[D. Option]\n"
        "</Question>\n"
        "<Answer> [Correct Answer Letter and Text] </Answer>"
    )
    
    try:
        raw_output = ""
        
        if req.model_id == "gemini":
            response = gemini_client.models.generate_content(
                model="gemini-3-flash-preview", 
                contents=prompt
            )
            raw_output = response.text
        
        elif req.model_id == "local-llama":
            response = ollama.chat(model='llama3', messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])
            raw_output = response['message']['content']

        elif req.model_id == "qwen":
            response = ollama.chat(model='qwen3:8b', messages=[
                {'role': 'user', 'content': prompt}
            ])
            raw_output = response['message']['content']

        elif req.model_id == "rag-piped-param":
            response = ollama.chat(model='llama3', messages=[
                {
                    'role': 'user',
                    'content': prompt2,
                },
            ])
            raw_output = response['message']['content']
        
        else:
            raw_output = "<Question> Local Mode Question <Question> <Answer> Local Answer <Answer>"

        # Return the raw text directly in the 'question' key
        # return {"question": raw_output}
        formatted_response = parse_ai_output(raw_output)
        return formatted_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
