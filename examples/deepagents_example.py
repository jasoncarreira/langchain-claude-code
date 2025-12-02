"""Example: Using ClaudeCodeChatModel with DeepAgents/LangGraph.

This example demonstrates integration with the deepagents library for
building complex agent workflows.

Prerequisites:
    pip install langchain-claude-code[examples]
    npm install -g @anthropic-ai/claude-code
"""

import json

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage

from langchain_claude_code import ClaudeCodeChatModel, ClaudeTool

system_prompt = (
    "You are an expert researcher. Your job is to conduct thorough "
    "research and then write a polished report."
)

model = ClaudeCodeChatModel(
    model="haiku",
    permission_mode="acceptEdits",
    allowed_tools=[
        ClaudeTool.WEB_FETCH,
        ClaudeTool.WEB_SEARCH,
        ClaudeTool.BASH,
        ClaudeTool.BASH_OUTPUT,
    ],
)

agent = create_deep_agent(
    model=model,
    name="Claude Code Agent",
    system_prompt=system_prompt,
)

if __name__ == "__main__":
    for chunk, meta in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What is langgraph? If you could not perform web searches "
                        "you must explicitly say so."
                    ),
                }
            ]
        },
        stream_mode="messages",
    ):
        if (
            type(chunk) == AIMessage
            and chunk.tool_calls is not None
            and len(chunk.tool_calls)
        ):
            for tc in chunk.tool_calls:
                print(f"Tool call: {json.dumps(tc)}")
        else:
            chunk.pretty_print()
