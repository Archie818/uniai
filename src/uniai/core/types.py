"""Type definitions for UniAI."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Message role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str

    def to_dict(self) -> dict:
        """Convert message to dictionary format for API calls."""
        return {"role": self.role.value, "content": self.content}


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """Response from a chat completion request."""

    content: str
    model: str
    usage: Optional[Usage] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[dict] = Field(default=None, exclude=True)

    def __str__(self) -> str:
        return self.content


class StreamChunk(BaseModel):
    """A single chunk from a streaming response."""

    content: str
    finish_reason: Optional[str] = None
    is_final: bool = False

    def __str__(self) -> str:
        return self.content

