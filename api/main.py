from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import sys
import json
from pathlib import Path
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import after setting up paths
try:
    from chat.service import generate_response
except ImportError:
    # Fallback if imports fail
    async def generate_response(message: str):
        return "Sorry, I'm having trouble loading my data right now.", []

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    try:
        reply, suggestions = await generate_response(chat_message.message)
        return {"reply": reply, "suggestions": suggestions}
    except Exception as e:
        return {"reply": f"Sorry, I encountered an error: {str(e)}", "suggestions": []}

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Johnny's API is running!"}