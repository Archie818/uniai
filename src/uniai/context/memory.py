"""Memory management for conversation context."""

from typing import Optional

from uniai.core.types import Message, Role


class Memory:
    """
    Manages conversation history for multi-turn dialogues.

    Provides automatic context management with optional length limiting
    to prevent context overflow.
    """

    def __init__(
        self,
        max_messages: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize memory storage.

        Args:
            max_messages: Maximum number of messages to retain (None for unlimited).
                          System prompt is not counted in this limit.
            system_prompt: Optional system prompt to prepend to conversations.
        """
        self._messages: list[Message] = []
        self._max_messages = max_messages
        self._system_prompt = system_prompt

    @property
    def messages(self) -> list[Message]:
        """Get all messages in memory."""
        return self._messages.copy()

    @property
    def system_prompt(self) -> Optional[str]:
        """Get the system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: Optional[str]) -> None:
        """Set the system prompt."""
        self._system_prompt = value

    def add_user_message(self, content: str) -> Message:
        """
        Add a user message to memory.

        Args:
            content: The message content.

        Returns:
            The created Message object.
        """
        message = Message(role=Role.USER, content=content)
        self._add_message(message)
        return message

    def add_assistant_message(self, content: str) -> Message:
        """
        Add an assistant message to memory.

        Args:
            content: The message content.

        Returns:
            The created Message object.
        """
        message = Message(role=Role.ASSISTANT, content=content)
        self._add_message(message)
        return message

    def add_system_message(self, content: str) -> Message:
        """
        Add a system message to memory.

        Args:
            content: The message content.

        Returns:
            The created Message object.
        """
        message = Message(role=Role.SYSTEM, content=content)
        self._add_message(message)
        return message

    def _add_message(self, message: Message) -> None:
        """
        Internal method to add a message and enforce limits.

        Args:
            message: The message to add.
        """
        self._messages.append(message)
        self._enforce_limit()

    def _enforce_limit(self) -> None:
        """Remove oldest messages if limit is exceeded."""
        if self._max_messages is not None and len(self._messages) > self._max_messages:
            # Calculate how many messages to remove
            excess = len(self._messages) - self._max_messages
            self._messages = self._messages[excess:]

    def get_context(self) -> list[Message]:
        """
        Get the full context including system prompt.

        Returns:
            List of messages ready for API call.
        """
        result = []

        # Add system prompt as first message if present
        if self._system_prompt:
            result.append(Message(role=Role.SYSTEM, content=self._system_prompt))

        result.extend(self._messages)
        return result

    def clear(self) -> None:
        """Clear all messages from memory (keeps system prompt)."""
        self._messages.clear()

    def pop_last(self) -> Optional[Message]:
        """
        Remove and return the last message.

        Returns:
            The last message, or None if memory is empty.
        """
        if self._messages:
            return self._messages.pop()
        return None

    def __len__(self) -> int:
        """Return the number of messages in memory."""
        return len(self._messages)

    def __repr__(self) -> str:
        return f"Memory(messages={len(self._messages)}, max={self._max_messages})"

