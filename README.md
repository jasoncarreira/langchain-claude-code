# langchain-claude-code

[![PyPI version](https://badge.fury.io/py/langchain-claude-code.svg)](https://badge.fury.io/py/langchain-claude-code)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

LangChain chat model wrapper for the Claude Code Agent SDK. Use Claude Code as a drop-in LangChain `BaseChatModel` with full tool support, streaming, and LangGraph compatibility.

## Prerequisites

**Claude Code CLI** (required):

```bash
npm install -g @anthropic-ai/claude-code
```

## Installation

```bash
pip install langchain-claude-code
```

With example dependencies:

```bash
pip install langchain-claude-code[examples]
```

## Authentication

### Option 1: API Key (for Anthropic API users)

```python
from langchain_claude_code import ClaudeCodeChatModel

model = ClaudeCodeChatModel(api_key="sk-ant-...")
```

Or set the environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Option 2: OAuth Token (for Claude Max subscribers)

```python
from langchain_claude_code import ClaudeCodeChatModel

model = ClaudeCodeChatModel(oauth_token="...")
```

Or set the environment variable:

```bash
export CLAUDE_CODE_OAUTH_TOKEN="..."
```

## Quick Start

### Basic Usage

```python
import asyncio
from langchain_claude_code import ClaudeCodeChatModel
from langchain_core.messages import HumanMessage

async def main():
    model = ClaudeCodeChatModel(model="sonnet")
    response = await model.ainvoke([HumanMessage(content="What is 2 + 2?")])
    print(response.content)

asyncio.run(main())
```

### With Tool Binding

```python
from langchain_claude_code import ClaudeCodeChatModel
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchTool

model = ClaudeCodeChatModel(model="haiku")
model = model.bind_tools([DuckDuckGoSearchTool()])

response = await model.ainvoke([
    HumanMessage(content="Search for the latest news about AI")
])
```

### Using Claude Code's Built-in Tools

```python
from langchain_claude_code import ClaudeCodeChatModel, ClaudeTool

model = ClaudeCodeChatModel(
    model="sonnet",
    allowed_tools=[
        ClaudeTool.WEB_SEARCH,
        ClaudeTool.WEB_FETCH,
        ClaudeTool.BASH,
        ClaudeTool.READ,
        ClaudeTool.WRITE,
    ],
)
```

### Streaming

```python
async for chunk in model.astream([HumanMessage(content="Write a poem")]):
    print(chunk.content, end="", flush=True)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"opus"` | Model to use: `opus`, `sonnet`, `haiku` |
| `permission_mode` | `str` | `"default"` | Permission mode: `default`, `acceptEdits`, `plan`, `bypassPermissions` |
| `allowed_tools` | `list` | `[]` | List of allowed Claude Code tools |
| `disallowed_tools` | `list` | `[]` | List of disallowed tools |
| `system_prompt` | `str` | `None` | Custom system prompt |
| `max_turns` | `int` | `None` | Maximum conversation turns |
| `max_budget_usd` | `float` | `None` | Maximum budget in USD |
| `cwd` | `str` | `None` | Working directory for file operations |
| `api_key` | `str` | `None` | Anthropic API key |
| `oauth_token` | `str` | `None` | Claude Code OAuth token |

## Permission Modes

| Mode | Description |
|------|-------------|
| `default` | Prompts for confirmation on potentially dangerous operations |
| `acceptEdits` | Automatically accepts file edits without confirmation |
| `plan` | Planning mode - generates plans without executing |
| `bypassPermissions` | Bypasses all permission checks (use with caution) |

> **Security Note**: When using `acceptEdits` or `bypassPermissions`, Claude Code will modify your filesystem without confirmation. Use these modes only in sandboxed environments or when you trust the input.

## Available Tools

```python
from langchain_claude_code import ClaudeTool

# File operations
ClaudeTool.READ
ClaudeTool.WRITE
ClaudeTool.EDIT
ClaudeTool.GLOB
ClaudeTool.GREP

# Shell
ClaudeTool.BASH
ClaudeTool.BASH_OUTPUT

# Web
ClaudeTool.WEB_SEARCH
ClaudeTool.WEB_FETCH

# Other
ClaudeTool.TASK
ClaudeTool.TODO_WRITE
```

## Examples

See the [examples/](examples/) directory for complete examples:

- `bind_tools_example.py` - Using LangChain tools with Claude Code
- `deepagents_example.py` - Integration with DeepAgents/LangGraph

## License

MIT License - see [LICENSE](LICENSE) for details.
