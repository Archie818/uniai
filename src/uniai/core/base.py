"""Abstract base class for all LLM providers."""

from abc import ABC, abstractmethod
from typing import Iterator

from uniai.core.config import ProviderConfig
from uniai.core.types import ChatResponse, Message, StreamChunk


class BaseProvider(ABC):
    """
    Abstract base class that defines the interface for all LLM providers.

    All provider implementations must inherit from this class and implement
    the required abstract methods.
    """

    # Provider identifier (override in subclasses)
    name: str = "base"

    def __init__(self, config: ProviderConfig):
        """
        Initialize the provider with configuration.

        Args:
            config: Provider configuration containing API key, model, etc.
        """
        self.config = config
        self._client = None

    @abstractmethod
    def _init_client(self) -> None:
        """Initialize the underlying API client."""
        pass

    @abstractmethod
    def chat(self, messages: list[Message]) -> ChatResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of messages in the conversation.

        Returns:
            ChatResponse containing the assistant's reply.
        """
        pass

    @abstractmethod
    def stream_chat(self, messages: list[Message]) -> Iterator[StreamChunk]:
        """
        Send a streaming chat completion request.

        Args:
            messages: List of messages in the conversation.

        Yields:
            StreamChunk objects containing partial responses.
        """
        pass

    def _prepare_messages(self, messages: list[Message]) -> list[dict]:
        """
        Convert Message objects to API-compatible format.

        Args:
            messages: List of Message objects.

        Returns:
            List of message dictionaries.
        """
        result = []

        # Add system prompt if configured
        if self.config.system_prompt:
            result.append({"role": "system", "content": self.config.system_prompt})

        # Add conversation messages
        for msg in messages:
            result.append(msg.to_dict())

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model!r})"

