import os
import json # Added to parse the string response
import uvicorn
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
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_1"))
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
        f"Generate a variety of academic questions (such as MCQs, Fill-in-the-blanks, and True/False) "
        f"for {req.topic} themed as '{req.theme}' at {req.depth} level.\n\n"
        "You are to strictly follow this output format -\n"
        "<Question> The generated question. <Question>\n"
        "<Answer> The reference answer <Answer>"
    )
    
    try:
        raw_output = ""
        
        if req.model_id == "gemini":
            response = gemini_client.models.generate_content(
                model="gemini-3-flash-preview", 
                contents=prompt
            )
            raw_output = response.text
        
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
