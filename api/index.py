from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(chat_message: ChatMessage):
    # Simple Johnny responses for now
    message = chat_message.message.lower()
    
    if "project" in message:
        reply = "🚀 I've worked on some cool projects! I built a web scraper that helped automate data collection, and a chat application using React and Node.js. My favorite was probably the machine learning project where I predicted stock prices - it taught me so much about data analysis!"
    elif "skill" in message:
        reply = "💻 I'm proficient in Python, JavaScript, React, Node.js, and SQL. I'm also learning machine learning with TensorFlow and love working with APIs. I pick up new technologies quickly - I learned React in just two weeks for my last project!"
    elif "experience" in message:
        reply = "📚 I'm a Computer Science student with hands-on experience from personal projects and coursework. I've built full-stack applications, worked with databases, and even contributed to open-source projects. I may be early in my career, but I'm passionate and eager to learn!"
    elif "funny" in message or "joke" in message:
        reply = "😄 Well, I once spent 3 hours debugging code only to realize I forgot a semicolon. Classic! But hey, that's how I learned to pay attention to details. Now I'm like a semicolon detective! 🕵️‍♂️"
    elif "different" in message or "unique" in message:
        reply = "🌟 What makes me different? I bring genuine enthusiasm and a fresh perspective. I'm not just looking for any job - I'm looking for a place where I can grow and contribute. Plus, I make pretty good coffee and I'm told my debugging skills are legendary! ☕"
    else:
        reply = f"Hi! I'm Johnny, and I'm excited you're interested in learning more! You asked about '{chat_message.message}' - I'd love to tell you more about my background, projects, and what makes me a great candidate. What specifically would you like to know?"
    
    suggestions = [
        "What's your favorite project you've worked on? 🚀",
        "Tell me something funny that happened while coding",
        "What makes you different from other candidates?",
        "What's the coolest thing you've learned recently?"
    ]
    
    return {"reply": reply, "suggestions": suggestions}

@app.get("/api/")
def health_check():
    return {"status": "ok", "message": "Johnny's API is running!"}

# For Vercel
from mangum import Mangum
handler = Mangum(app)