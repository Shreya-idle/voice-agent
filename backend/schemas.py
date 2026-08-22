from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=500)
    uid: Optional[str] = None
    conversation_id: Optional[str] = Field(None, alias="conversationId")

class ChatResponse(BaseModel):
    response: str
    audio_url: Optional[str] = None
    conversation_id: Optional[str] = None
    audio_path: Optional[str] = None
    remaining_credits: Optional[int] = None
