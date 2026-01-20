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
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
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
    # JSON-specific instructions
    prompt = (
        f"Generate an academic question for {req.topic} ({req.theme}) at {req.depth} level.\n"
        "Include any necessary scenario or context within the question field.\n"
        "Return the result EXCLUSIVELY as a JSON object with these exact keys: "
        "'question' and 'reference_answer'. Do not include any other text or markdown headers."
    )
    
    try:
        if req.model_id == "gemini":
            response = gemini_client.models.generate_content(
                model="gemini-3-flash-preview", 
                contents=prompt,
                config={
                    # This tells Gemini to ONLY output valid JSON
                    'response_mime_type': 'application/json' 
                }
            )
            # response.text is a JSON string, e.g., '{"question": "...", "reference_answer": "..."}'
            data = json.loads(response.text)
            return {
                "question": data.get("question"),
                "answer": data.get("reference_answer")
            }
        
        elif req.model_id == "chatgpt":
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                # OpenAI also supports JSON mode
                response_format={"type": "json_object"}, 
                messages=[{"role": "user", "content": prompt}]
            )
            data = json.loads(response.choices[0].message.content)
            return {
                "question": data.get("question"),
                "answer": data.get("reference_answer")
            }
        
        else:
            return {"question": "Local Mode", "answer": "Local Response"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="localhost", port=8000)
    except KeyboardInterrupt:
        print("Stopping server...")
    finally:
        print("Port 8000 has been released.")