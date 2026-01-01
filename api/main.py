from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# Load environment variables
load_dotenv()

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

# Create FastAPI app
app = FastAPI(title="Johnny Chatbot")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import chat service (with fallback)
try:
    from chat.service import generate_response
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback function
    async def generate_response(message: str):
        return f"Hi! I'm Johnny. You asked: '{message}'. I'm still setting up my full responses, but I'm excited to chat with you!", [
            "What's your favorite project? 🚀",
            "Tell me about your skills",
            "What makes you unique?",
            "What are you learning?"
        ]

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    try:
        reply, suggestions = await generate_response(chat_message.message)
        return {"reply": reply, "suggestions": suggestions}
    except Exception as e:
        return {
            "reply": f"Sorry, I encountered an error: {str(e)}", 
            "suggestions": [
                "What's your favorite project? 🚀",
                "Tell me about your skills",
                "What makes you unique?",
                "What are you learning?"
            ]
        }

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Johnny's API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# Serverless adapter for Vercel
from mangum import Mangum
handler = Mangum(app)