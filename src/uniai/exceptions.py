"""Custom exceptions for UniAI."""


class UniAIError(Exception):
    """Base exception for all UniAI errors."""

    pass


class ConfigurationError(UniAIError):
    """Raised when there is a configuration error."""

    pass


class ProviderError(UniAIError):
    """Raised when there is an error with a provider."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        self.message = message
        super().__init__(f"[{provider}] {message}")


class APIError(UniAIError):
    """Raised when an API call fails."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response: dict | None = None,
    ):
        self.status_code = status_code
        self.response = response
        super().__init__(message)


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""

    pass


class AuthenticationError(APIError):
    """Raised when authentication fails."""

    pass


class InvalidRequestError(APIError):
    """Raised when the request is invalid."""

    pass

