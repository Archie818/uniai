"""LLM Provider implementations."""

from uniai.providers.openai import OpenAIProvider
from uniai.providers.deepseek import DeepSeekProvider

# Registry of available providers
PROVIDERS: dict[str, type] = {
    "openai": OpenAIProvider,
    "deepseek": DeepSeekProvider,
}


def get_provider(name: str) -> type:
    """
    Get a provider class by name.

    Args:
        name: Provider name (e.g., 'openai', 'deepseek').

    Returns:
        The provider class.

    Raises:
        ValueError: If provider is not found.
    """
    provider = PROVIDERS.get(name.lower())
    if provider is None:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider: {name}. Available: {available}")
    return provider


def register_provider(name: str, provider_class: type) -> None:
    """
    Register a custom provider.

    Args:
        name: Provider name.
        provider_class: Provider class (must inherit from BaseProvider).
    """
    PROVIDERS[name.lower()] = provider_class


__all__ = [
    "OpenAIProvider",
    "DeepSeekProvider",
    "PROVIDERS",
    "get_provider",
    "register_provider",
]

