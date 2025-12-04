# UniAI

[![PyPI version](https://badge.fury.io/py/uniai.svg)](https://badge.fury.io/py/uniai)
[![Python Versions](https://img.shields.io/pypi/pyversions/uniai.svg)](https://pypi.org/project/uniai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/github/stars/Archie818/uniai?style=social)](https://github.com/Archie818/uniai)

**A Universal AI Interface Layer for Python**

> Seamlessly connect to and switch between LLM providers with unified context management and streaming support.

## ✨ Features

| Feature                   | Description                                             |
| ------------------------- | ------------------------------------------------------- |
| 🔄 **Unified Interface**  | Single API for multiple LLM providers                   |
| 🔀 **Easy Switching**     | Switch between OpenAI, DeepSeek, and more with one line |
| 💬 **Context Management** | Built-in conversation history tracking                  |
| 🌊 **Streaming**          | Stream responses in real-time                           |
| 🛡️ **Type Safety**        | Full type hints and Pydantic validation                 |
| 🔧 **Extensible**         | Easy to add custom providers                            |

## 📦 Installation

```bash
pip install uniai
```

**From source:**

```bash
git clone https://github.com/Archie818/uniai.git
cd uniai
pip install -e .
```

## 🚀 Quick Start

```python
from uniai import UniAI

# Initialize with your preferred provider
bot = UniAI(provider="openai", api_key="sk-...", model="gpt-4o-mini")

# Simple chat
response = bot.chat("Hello, who are you?")
print(response)

# Multi-turn conversation (context is automatically managed)
response = bot.chat("What can you help me with?")
response = bot.chat("Tell me more about the first thing you mentioned")
```

## 🌊 Streaming

```python
for chunk in bot.stream("Tell me a story about a brave knight"):
    print(chunk, end="", flush=True)
```

## 🔀 Switch Providers

```python
# Start with OpenAI
bot = UniAI(provider="openai", api_key="sk-openai-key", model="gpt-4o-mini")
bot.chat("Hello!")

# Switch to DeepSeek (preserves history)
bot.switch_provider(provider="deepseek", api_key="sk-deepseek-key", model="deepseek-chat")
bot.chat("Continue our conversation")

# Switch and clear history
bot.switch_provider(provider="openai", api_key="sk-...", model="gpt-4o", keep_history=False)
```

## 📋 Supported Providers

| Provider     | Models                                    | Status         |
| ------------ | ----------------------------------------- | -------------- |
| **OpenAI**   | gpt-4o, gpt-4o-mini, gpt-4, gpt-3.5-turbo | ✅ Supported   |
| **DeepSeek** | deepseek-chat, deepseek-coder             | ✅ Supported   |
| **Claude**   | claude-3.5-sonnet, claude-3-opus          | 🚧 Coming Soon |
| **Gemini**   | gemini-pro                                | 🚧 Planned     |

## ⚙️ Configuration

```python
bot = UniAI(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o-mini",
    system_prompt="You are a helpful assistant.",  # System instructions
    temperature=0.7,         # Randomness (0.0-2.0)
    max_tokens=2048,         # Max response length
    max_history=20,          # Conversation history limit
    timeout=60.0,            # Request timeout (seconds)
    max_retries=3,           # Retry attempts
)
```

## 📖 Advanced Usage

### Full Response Object

```python
response = bot.chat_with_response("What is Python?")

print(f"Content: {response.content}")
print(f"Model: {response.model}")
print(f"Tokens: {response.usage.total_tokens}")
print(f"Finish reason: {response.finish_reason}")
```

### History Management

```python
history = bot.get_history()      # Get conversation history
bot.clear_history()              # Clear history
print(len(bot.memory))           # Messages count
```

### Custom Providers

```python
from uniai.core.base import BaseProvider
from uniai.providers import register_provider

class MyProvider(BaseProvider):
    name = "custom"

    def _init_client(self):
        pass

    def chat(self, messages):
        # Your implementation
        pass

    def stream_chat(self, messages):
        # Your implementation
        pass

register_provider("custom", MyProvider)
bot = UniAI(provider="custom", api_key="...", model="...")
```

## 🛡️ Error Handling

```python
from uniai.exceptions import AuthenticationError, RateLimitError, APIError

try:
    response = bot.chat("Hello!")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded")
except APIError as e:
    print(f"API error: {e}")
```

## 🏗️ Architecture

```
uniai/
├── client.py            # Main UniAI class
├── exceptions.py        # Custom exceptions
├── core/
│   ├── base.py          # Abstract BaseProvider
│   ├── config.py        # Pydantic config models
│   └── types.py         # Type definitions
├── context/
│   └── memory.py        # Conversation memory
└── providers/
    ├── openai.py        # OpenAI provider
    └── deepseek.py      # DeepSeek provider
```

## 🗺️ Roadmap

- [x] **v0.1** - Core functionality (Unified interface, OpenAI & DeepSeek, Context management, Streaming)
- [ ] **v0.2** - Extended providers (Claude, Gemini, Ollama)
- [ ] **v0.3** - Advanced features (Async support, Function calling, Token counting)
- [ ] **v1.0** - Agent framework (Multi-model collaboration, Tool integration)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[MIT License](LICENSE) © 2024 UniAI Contributors
