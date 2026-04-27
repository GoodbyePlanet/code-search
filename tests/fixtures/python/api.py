from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ChatRequest(BaseModel):
    """A request payload for the chat endpoint."""

    message: str


@router.post("/api/chat")
async def chat(req: ChatRequest) -> dict:
    """Echo the message back."""
    return {"reply": req.message}


def helper(value: int) -> int:
    return value * 2
