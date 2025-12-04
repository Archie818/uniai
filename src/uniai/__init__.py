"""
UniAI - A Universal AI Interface Layer

Seamlessly connect to and switch between different LLM providers
with unified context management and streaming support.
"""

from uniai.client import UniAI
from uniai.core.base import BaseProvider
from uniai.core.config import ProviderConfig
from uniai.core.types import Message, Role, ChatResponse, StreamChunk
from uniai.context.memory import Memory
from uniai.exceptions import (
    UniAIError,
    ProviderError,
    ConfigurationError,
    APIError,
)

__version__ = "0.1.0"

__all__ = [
    # Main client
    "UniAI",
    # Base classes
    "BaseProvider",
    # Config
    "ProviderConfig",
    # Types
    "Message",
    "Role",
    "ChatResponse",
    "StreamChunk",
    # Memory
    "Memory",
    # Exceptions
    "UniAIError",
    "ProviderError",
    "ConfigurationError",
    "APIError",
]

