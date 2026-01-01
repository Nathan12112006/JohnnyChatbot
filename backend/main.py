from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# Add backend directory to Python path for imports
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Load environment variables
load_dotenv()

from chat.router import router as chat_router

app = FastAPI(title="Internship Chatbot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

# Include API routes with /api prefix
app.include_router(chat_router, prefix="/api")

@app.get("/api/")
def health_check():
    return {"status": "ok"}

# Vercel serverless function handler
handler = app
