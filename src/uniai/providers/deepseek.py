"""DeepSeek provider implementation."""

from typing import Iterator

from openai import OpenAI

from uniai.core.base import BaseProvider
from uniai.core.config import DeepSeekConfig, ProviderConfig
from uniai.core.types import ChatResponse, Message, StreamChunk, Usage
from uniai.exceptions import APIError, AuthenticationError, RateLimitError


class DeepSeekProvider(BaseProvider):
    """
    Provider implementation for DeepSeek's API.

    DeepSeek uses an OpenAI-compatible API, so we use the OpenAI client
    with a custom base URL.
    """

    name = "deepseek"

    def __init__(self, config: ProviderConfig):
        """
        Initialize DeepSeek provider.

        Args:
            config: Provider configuration. Will use DeepSeekConfig defaults
                    if not already a DeepSeekConfig instance.
        """
        # Convert to DeepSeekConfig if needed to get defaults
        if not isinstance(config, DeepSeekConfig):
            config = DeepSeekConfig(
                api_key=config.api_key,
                model=config.model or DeepSeekConfig.model_fields["model"].default,
                base_url=config.base_url or DeepSeekConfig.model_fields["base_url"].default,
                timeout=config.timeout,
                max_retries=config.max_retries,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                system_prompt=config.system_prompt,
            )
        super().__init__(config)
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the DeepSeek client (using OpenAI-compatible client)."""
        self._client = OpenAI(
            api_key=self.config.api_key.get_secret_value(),
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

    def chat(self, messages: list[Message]) -> ChatResponse:
        """
        Send a chat completion request to DeepSeek.

        Args:
            messages: List of messages in the conversation.

        Returns:
            ChatResponse containing the assistant's reply.
        """
        prepared_messages = self._prepare_messages(messages)

        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=prepared_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            choice = response.choices[0]
            usage = None
            if response.usage:
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            return ChatResponse(
                content=choice.message.content or "",
                model=response.model,
                usage=usage,
                finish_reason=choice.finish_reason,
                raw_response=response.model_dump(),
            )

        except Exception as e:
            self._handle_error(e)

    def stream_chat(self, messages: list[Message]) -> Iterator[StreamChunk]:
        """
        Send a streaming chat completion request to DeepSeek.

        Args:
            messages: List of messages in the conversation.

        Yields:
            StreamChunk objects containing partial responses.
        """
        prepared_messages = self._prepare_messages(messages)

        try:
            stream = self._client.chat.completions.create(
                model=self.config.model,
                messages=prepared_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta

                    if delta.content:
                        yield StreamChunk(
                            content=delta.content,
                            finish_reason=choice.finish_reason,
                            is_final=choice.finish_reason is not None,
                        )
                    elif choice.finish_reason:
                        yield StreamChunk(
                            content="",
                            finish_reason=choice.finish_reason,
                            is_final=True,
                        )

        except Exception as e:
            self._handle_error(e)

    def _handle_error(self, error: Exception) -> None:
        """
        Handle DeepSeek API errors.

        Args:
            error: The exception to handle.

        Raises:
            Appropriate UniAI exception.
        """
        from openai import (
            APIConnectionError,
            AuthenticationError as OpenAIAuthError,
            RateLimitError as OpenAIRateLimitError,
        )

        if isinstance(error, OpenAIAuthError):
            raise AuthenticationError(
                "Invalid API key or unauthorized access",
                status_code=401,
            )
        elif isinstance(error, OpenAIRateLimitError):
            raise RateLimitError(
                "Rate limit exceeded. Please try again later.",
                status_code=429,
            )
        elif isinstance(error, APIConnectionError):
            raise APIError(
                f"Failed to connect to DeepSeek API: {error}",
                status_code=None,
            )
        else:
            raise APIError(f"DeepSeek API error: {error}")

