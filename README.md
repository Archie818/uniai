# UniAI

**A Universal AI Interface Layer for Python**

UniAI simplifies working with multiple LLM providers by offering a unified interface with automatic context management and streaming support.

## Features

- 🔄 **Unified Interface**: Single API for multiple LLM providers
- 🔀 **Easy Provider Switching**: Switch between OpenAI, DeepSeek, and more with one line
- 💬 **Automatic Context Management**: Built-in conversation history tracking
- 🌊 **Streaming Support**: Stream responses in real-time
- 🛡️ **Type Safety**: Full type hints and Pydantic validation
- 🔧 **Extensible**: Easy to add custom providers

## Installation

```bash
pip install uniai
```

Or install from source:

```bash
git clone https://github.com/Archie818/uniai.git
cd uniai
pip install -e .
```

## Quick Start

```python
from uniai import UniAI

# Initialize with your preferred provider
bot = UniAI(
    provider="openai",
    api_key="sk-your-api-key",
    model="gpt-4o-mini"
)

# Simple chat
response = bot.chat("Hello, who are you?")
print(response)

# Multi-turn conversation (context is automatically managed)
response = bot.chat("What can you help me with?")
print(response)

response = bot.chat("Tell me more about the first thing you mentioned")
print(response)
```

## Streaming Responses

```python
from uniai import UniAI

bot = UniAI(provider="openai", api_key="sk-...", model="gpt-4o-mini")

# Stream response chunks
for chunk in bot.stream("Tell me a story about a brave knight"):
    print(chunk, end="", flush=True)
print()  # New line at the end
```

## Switching Providers

```python
from uniai import UniAI

# Start with OpenAI
bot = UniAI(provider="openai", api_key="sk-openai-key", model="gpt-4o-mini")
response = bot.chat("Hello!")

# Switch to DeepSeek (preserves conversation history by default)
bot.switch_provider(
    provider="deepseek",
    api_key="sk-deepseek-key",
    model="deepseek-chat"
)
response = bot.chat("Continue our conversation")

# Switch and clear history
bot.switch_provider(
    provider="openai",
    api_key="sk-openai-key",
    model="gpt-4o",
    keep_history=False
)
```

## Supported Providers

| Provider | Models                                    | Status       |
| -------- | ----------------------------------------- | ------------ |
| OpenAI   | gpt-4o, gpt-4o-mini, gpt-4, gpt-3.5-turbo | ✅ Supported |
| DeepSeek | deepseek-chat, deepseek-coder             | ✅ Supported |
| Claude   | Coming soon                               | 🚧 Planned   |

## Advanced Usage

### System Prompt

```python
bot = UniAI(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o-mini",
    system_prompt="You are a helpful coding assistant. Always provide code examples."
)
```

### Conversation History Management

```python
# Limit history to prevent context overflow
bot = UniAI(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o-mini",
    max_history=20  # Keep only last 20 messages
)

# Get conversation history
history = bot.get_history()
print(history)

# Clear history
bot.clear_history()

# Access memory directly
print(f"Messages in memory: {len(bot.memory)}")
```

### Full Response Object

```python
# Get detailed response with metadata
response = bot.chat_with_response("What is Python?")

print(f"Content: {response.content}")
print(f"Model: {response.model}")
print(f"Tokens used: {response.usage.total_tokens}")
print(f"Finish reason: {response.finish_reason}")
```

### Configuration Options

```python
bot = UniAI(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o-mini",
    base_url=None,           # Custom API endpoint
    system_prompt=None,      # System instructions
    temperature=1.0,         # Randomness (0.0-2.0)
    max_tokens=None,         # Max response length
    max_history=None,        # Message history limit
    timeout=60.0,            # Request timeout (seconds)
    max_retries=3,           # Retry attempts
)
```

## Architecture

```
uniai/
├── __init__.py          # Package exports
├── client.py            # Main UniAI class
├── exceptions.py        # Custom exceptions
├── core/
│   ├── base.py          # Abstract BaseProvider
│   ├── config.py        # Pydantic config models
│   └── types.py         # Type definitions
├── context/
│   └── memory.py        # Conversation memory
└── providers/
    ├── openai.py        # OpenAI implementation
    └── deepseek.py      # DeepSeek implementation
```

## Adding Custom Providers

```python
from uniai.core.base import BaseProvider
from uniai.providers import register_provider

class MyCustomProvider(BaseProvider):
    name = "custom"

    def _init_client(self):
        # Initialize your client
        pass

    def chat(self, messages):
        # Implement chat
        pass

    def stream_chat(self, messages):
        # Implement streaming
        pass

# Register the provider
register_provider("custom", MyCustomProvider)

# Now you can use it
bot = UniAI(provider="custom", api_key="...", model="...")
```

## Error Handling

```python
from uniai import UniAI
from uniai.exceptions import (
    UniAIError,
    AuthenticationError,
    RateLimitError,
    APIError,
)

bot = UniAI(provider="openai", api_key="sk-...", model="gpt-4o-mini")

try:
    response = bot.chat("Hello!")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded, please wait")
except APIError as e:
    print(f"API error: {e}")
except UniAIError as e:
    print(f"UniAI error: {e}")
```

## Roadmap

- [x] v0.1: Core functionality
  - [x] Unified interface
  - [x] OpenAI & DeepSeek support
  - [x] Context management
  - [x] Streaming
- [ ] v0.2: Extended provider support
  - [ ] Claude/Anthropic
  - [ ] Google Gemini
  - [ ] Local models (Ollama)
- [ ] v0.3: Advanced features
  - [ ] Async support
  - [ ] Function calling
  - [ ] Token counting
- [ ] v1.0: Agent framework
  - [ ] Agent development
  - [ ] Multi-model collaboration
  - [ ] Tool integration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.
