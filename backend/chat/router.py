from fastapi import APIRouter
from pydantic import BaseModel
from chat.service import generate_response

router = APIRouter(prefix="/chat")

class ChatRequest(BaseModel):
    message: str

@router.post("")
async def chat(req: ChatRequest):
    reply, suggestions = await generate_response(req.message)
    return {
        "reply": reply,
        "suggestions": suggestions
    }
