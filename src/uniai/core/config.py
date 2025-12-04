"""Configuration models for UniAI."""

from typing import Optional

from pydantic import BaseModel, Field, SecretStr


class ProviderConfig(BaseModel):
    """Base configuration for all providers."""

    api_key: SecretStr
    model: str
    base_url: Optional[str] = None
    timeout: float = Field(default=60.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    system_prompt: Optional[str] = None

    model_config = {"frozen": False}


class OpenAIConfig(ProviderConfig):
    """Configuration specific to OpenAI provider."""

    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"


class DeepSeekConfig(ProviderConfig):
    """Configuration specific to DeepSeek provider."""

    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"


class ClaudeConfig(ProviderConfig):
    """Configuration specific to Claude/Anthropic provider."""

    base_url: str = "https://api.anthropic.com"
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = Field(default=4096, ge=1)  # Claude requires max_tokens

