from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=500)
    uid: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    audio_url: Optional[str] = None
    remaining_credits: Optional[int] = None
