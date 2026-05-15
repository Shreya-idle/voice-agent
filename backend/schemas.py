from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    uid: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    audio_url: Optional[str] = None
    remaining_credits: Optional[int] = None
