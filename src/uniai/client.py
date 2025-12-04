"""Main UniAI client class."""

from typing import Iterator, Optional

from pydantic import SecretStr

from uniai.context.memory import Memory
from uniai.core.base import BaseProvider
from uniai.core.config import ProviderConfig
from uniai.core.types import ChatResponse, StreamChunk
from uniai.exceptions import ConfigurationError
from uniai.providers import get_provider


class UniAI:
    """
    Universal AI interface for seamless LLM provider switching.

    Provides a unified API for interacting with different LLM providers
    (OpenAI, DeepSeek, etc.) with automatic context management.

    Example:
        >>> bot = UniAI(provider="openai", api_key="sk-...", model="gpt-4o-mini")
        >>> response = bot.chat("Hello, who are you?")
        >>> print(response)

        >>> # Streaming
        >>> for chunk in bot.stream("Tell me a story"):
        ...     print(chunk, end="", flush=True)

        >>> # Switch provider easily
        >>> bot.switch_provider("deepseek", api_key="sk-...", model="deepseek-chat")
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        max_history: Optional[int] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize the UniAI client.

        Args:
            provider: Provider name ('openai', 'deepseek', etc.).
            api_key: API key for the provider.
            model: Model name (uses provider default if not specified).
            base_url: Custom base URL for the API.
            system_prompt: System prompt to prepend to all conversations.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens in response.
            max_history: Maximum messages to retain in memory.
            timeout: Request timeout in seconds.
            max_retries: Number of retry attempts.
        """
        self._provider_name = provider.lower()
        self._memory = Memory(max_messages=max_history, system_prompt=system_prompt)

        # Build provider config
        config = ProviderConfig(
            api_key=SecretStr(api_key),
            model=model or "",  # Provider will use its default
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            system_prompt=system_prompt,
        )

        # Initialize provider
        self._provider = self._create_provider(self._provider_name, config)

    def _create_provider(self, name: str, config: ProviderConfig) -> BaseProvider:
        """
        Create a provider instance.

        Args:
            name: Provider name.
            config: Provider configuration.

        Returns:
            Initialized provider instance.
        """
        try:
            provider_class = get_provider(name)
            return provider_class(config)
        except ValueError as e:
            raise ConfigurationError(str(e))

    @property
    def provider(self) -> str:
        """Get the current provider name."""
        return self._provider_name

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._provider.config.model

    @property
    def memory(self) -> Memory:
        """Get the memory/context manager."""
        return self._memory

    def chat(self, message: str) -> str:
        """
        Send a message and get a response.

        Automatically manages conversation context.

        Args:
            message: User message to send.

        Returns:
            Assistant's response as a string.
        """
        # Add user message to memory
        self._memory.add_user_message(message)

        # Get response from provider
        response = self._provider.chat(self._memory.messages)

        # Add assistant response to memory
        self._memory.add_assistant_message(response.content)

        return response.content

    def chat_with_response(self, message: str) -> ChatResponse:
        """
        Send a message and get a full response object.

        Like chat(), but returns the full ChatResponse with metadata.

        Args:
            message: User message to send.

        Returns:
            ChatResponse object with content, usage, etc.
        """
        # Add user message to memory
        self._memory.add_user_message(message)

        # Get response from provider
        response = self._provider.chat(self._memory.messages)

        # Add assistant response to memory
        self._memory.add_assistant_message(response.content)

        return response

    def stream(self, message: str) -> Iterator[str]:
        """
        Send a message and stream the response.

        Automatically manages conversation context.

        Args:
            message: User message to send.

        Yields:
            Response chunks as strings.
        """
        # Add user message to memory
        self._memory.add_user_message(message)

        # Collect full response for memory
        full_response = []

        # Stream response from provider
        for chunk in self._provider.stream_chat(self._memory.messages):
            if chunk.content:
                full_response.append(chunk.content)
                yield chunk.content

        # Add complete response to memory
        self._memory.add_assistant_message("".join(full_response))

    def stream_with_chunks(self, message: str) -> Iterator[StreamChunk]:
        """
        Send a message and stream response chunks.

        Like stream(), but yields StreamChunk objects with metadata.

        Args:
            message: User message to send.

        Yields:
            StreamChunk objects with content and metadata.
        """
        # Add user message to memory
        self._memory.add_user_message(message)

        # Collect full response for memory
        full_response = []

        # Stream response from provider
        for chunk in self._provider.stream_chat(self._memory.messages):
            if chunk.content:
                full_response.append(chunk.content)
            yield chunk

        # Add complete response to memory
        self._memory.add_assistant_message("".join(full_response))

    def switch_provider(
        self,
        provider: str,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        keep_history: bool = True,
        **kwargs,
    ) -> None:
        """
        Switch to a different provider.

        Args:
            provider: New provider name.
            api_key: API key for the new provider.
            model: Model name (uses provider default if not specified).
            base_url: Custom base URL for the API.
            keep_history: Whether to preserve conversation history.
            **kwargs: Additional provider configuration options.
        """
        # Build new config
        config = ProviderConfig(
            api_key=SecretStr(api_key),
            model=model or "",
            base_url=base_url,
            temperature=kwargs.get("temperature", self._provider.config.temperature),
            max_tokens=kwargs.get("max_tokens", self._provider.config.max_tokens),
            timeout=kwargs.get("timeout", self._provider.config.timeout),
            max_retries=kwargs.get("max_retries", self._provider.config.max_retries),
            system_prompt=kwargs.get("system_prompt", self._memory.system_prompt),
        )

        # Create new provider
        self._provider_name = provider.lower()
        self._provider = self._create_provider(self._provider_name, config)

        # Clear history if requested
        if not keep_history:
            self._memory.clear()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._memory.clear()

    def get_history(self) -> list[dict]:
        """
        Get conversation history as a list of dictionaries.

        Returns:
            List of message dictionaries with 'role' and 'content'.
        """
        return [msg.to_dict() for msg in self._memory.messages]

    def __repr__(self) -> str:
        return (
            f"UniAI(provider={self._provider_name!r}, "
            f"model={self.model!r}, "
            f"history={len(self._memory)})"
        )

