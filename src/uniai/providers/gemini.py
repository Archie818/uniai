"""Google Gemini provider implementation."""

from typing import Iterator

import google.generativeai as genai

from uniai.core.base import BaseProvider
from uniai.core.config import GeminiConfig, ProviderConfig
from uniai.core.types import ChatResponse, Message, StreamChunk, Usage
from uniai.exceptions import APIError, AuthenticationError, RateLimitError


class GeminiProvider(BaseProvider):
    """
    Provider implementation for Google's Gemini API.

    Supports models like gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash, etc.
    """

    name = "gemini"

    def __init__(self, config: ProviderConfig):
        """
        Initialize Gemini provider.

        Args:
            config: Provider configuration. Will use GeminiConfig defaults
                    if not already a GeminiConfig instance.
        """
        if not isinstance(config, GeminiConfig):
            config = GeminiConfig(
                api_key=config.api_key,
                model=config.model or GeminiConfig.model_fields["model"].default,
                base_url=config.base_url,
                timeout=config.timeout,
                max_retries=config.max_retries,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                system_prompt=config.system_prompt,
            )
        super().__init__(config)
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Gemini client."""
        genai.configure(api_key=self.config.api_key.get_secret_value())
        # Base URL is managed by the SDK; we ignore config.base_url here.
        self._client = genai.GenerativeModel(self.config.model)

    def _build_prompt(self, messages: list[Message]) -> str:
        """
        Build a text prompt from conversation messages.

        Gemini supports structured messages, but for simplicity and broad
        compatibility we serialize the conversation into a single text prompt.
        """
        parts: list[str] = []

        if self.config.system_prompt:
            parts.append(f"System: {self.config.system_prompt}")

        for msg in messages:
            role = msg.role.value
            if role == "system":
                parts.append(f"System: {msg.content}")
            elif role == "user":
                parts.append(f"User: {msg.content}")
            elif role == "assistant":
                parts.append(f"Assistant: {msg.content}")
            else:
                parts.append(msg.content)

        return "\n".join(parts)

    def _get_generation_config(self) -> genai.types.GenerationConfig:
        """Build generation config for Gemini requests."""
        kwargs: dict[str, object] = {"temperature": self.config.temperature}
        if self.config.max_tokens is not None:
            # google-generativeai uses max_output_tokens for response length
            kwargs["max_output_tokens"] = self.config.max_tokens

        return genai.types.GenerationConfig(**kwargs)

    def chat(self, messages: list[Message]) -> ChatResponse:
        """
        Send a chat completion request to Gemini.

        Args:
            messages: List of messages in the conversation.

        Returns:
            ChatResponse containing the assistant's reply.
        """
        prompt = self._build_prompt(messages)

        try:
            response = self._client.generate_content(
                prompt,
                generation_config=self._get_generation_config(),
            )

            usage = None
            usage_metadata = getattr(response, "usage_metadata", None)
            if usage_metadata is not None:
                usage = Usage(
                    prompt_tokens=getattr(
                        usage_metadata,
                        "prompt_token_count",
                        0,
                    ),
                    completion_tokens=getattr(
                        usage_metadata,
                        "candidates_token_count",
                        0,
                    ),
                    total_tokens=getattr(
                        usage_metadata,
                        "total_token_count",
                        0,
                    ),
                )

            content = getattr(response, "text", "") or ""

            return ChatResponse(
                content=content,
                model=self.config.model,
                usage=usage,
                finish_reason=None,
                raw_response=None,
            )

        except Exception as error:  # noqa: BLE001
            self._handle_error(error)

    def stream_chat(self, messages: list[Message]) -> Iterator[StreamChunk]:
        """
        Send a streaming chat completion request to Gemini.

        Args:
            messages: List of messages in the conversation.

        Yields:
            StreamChunk objects containing partial responses.
        """
        prompt = self._build_prompt(messages)

        try:
            stream = self._client.generate_content(
                prompt,
                generation_config=self._get_generation_config(),
                streaming=True,
            )

            for chunk in stream:
                text = getattr(chunk, "text", "")
                if text:
                    yield StreamChunk(
                        content=text,
                        finish_reason=None,
                        is_final=False,
                    )

            # Emit a final chunk to signal completion
            yield StreamChunk(
                content="",
                finish_reason="stop",
                is_final=True,
            )

        except Exception as error:  # noqa: BLE001
            self._handle_error(error)

    def _handle_error(self, error: Exception) -> None:
        """
        Handle Gemini API errors.

        Args:
            error: The exception to handle.

        Raises:
            Appropriate UniAI exception.
        """
        try:
            from google.api_core import exceptions as google_exceptions
        except Exception:  # noqa: BLE001
            # If google.api_core is not available, fall back to a generic error.
            raise APIError(f"Gemini API error: {error}") from error

        if isinstance(error, google_exceptions.Unauthenticated):
            raise AuthenticationError(
                "Invalid Gemini API key or unauthorized access",
                status_code=401,
            ) from error
        if isinstance(error, google_exceptions.ResourceExhausted):
            raise RateLimitError(
                "Gemini rate limit exceeded. Please try again later.",
                status_code=429,
            ) from error
        if isinstance(error, google_exceptions.GoogleAPIError):
            status = getattr(error, "code", None)
            raise APIError(
                f"Gemini API error: {error}",
                status_code=status,
            ) from error

        raise APIError(f"Gemini API error: {error}") from error


