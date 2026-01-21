import os
import json # Added to parse the string response
import uvicorn
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

load_dotenv()
app = FastAPI()

# Clients
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_21"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class QueryRequest(BaseModel):
    model_id: str
    depth: str
    theme: str
    topic: str

@app.get("/")
async def serve_index():
    return FileResponse(BASE_DIR / "../frontend/index.html")

@app.post("/ask")
async def ask_llm(req: QueryRequest):
    # The strictly defined prompt for diverse questions
    prompt = (
        f"Act as an expert academic assessment designer. Your goal is to generate 12-15 diverse "
        f"questions for the topic '{req.topic}' themed as '{req.theme}' at {req.depth} level.\n\n"
        
        "### INSTRUCTION ###\n"
        "You MUST include a mix of ALL the following question types:\n"
        "1. Multiple Choice (MCQ) - with 4 distinct options (A, B, C, D)\n"
        "2. Fill-in-the-Blanks - using underscores for the missing word\n"
        "3. True/False - factual statements based on the theme\n"
        "4. Short Answer - requiring 1-2 sentence explanations\n"
        "5. Matching - list items to be paired (e.g., 1-A, 2-B)\n"
        "6. Scenario-Based - a mini-story followed by a question\n\n"
        
        "### OUTPUT FORMAT RULES ###\n"
        "Strictly wrap every question and every answer in these exact tags:\n"
        "<Question> [The question text, including options if it is an MCQ] <Question>\n"
        "<Answer> [The correct answer and a brief explanation] <Answer>\n\n"
        
        "### EXAMPLE (Follow this structure) ###\n"
        "<Question> What is the powerhouse of the cell?\nA) Nucleus\nB) Mitochondria\nC) Ribosome\nD) Golgi Body <Question>\n"
        "<Answer> B) Mitochondria. It is responsible for ATP production. <Answer>\n"
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

        elif req.model_id == "chatgpt":
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            raw_output = response.choices[0].message.content
        
        else:
            raw_output = "<Question> Local Mode Question <Question> <Answer> Local Answer <Answer>"

        # Return the raw text directly in the 'question' key
        return {"question": raw_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
