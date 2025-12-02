"""Example: Binding LangChain tools to ClaudeCodeChatModel.

This example demonstrates how to use DuckDuckGo search with Claude Code.

Prerequisites:
    pip install langchain-claude-code[examples]
    npm install -g @anthropic-ai/claude-code
"""

import asyncio

from langchain_claude_code import ClaudeCodeChatModel
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchTool
from langchain_core.messages import HumanMessage, SystemMessage


async def main():
    model = ClaudeCodeChatModel(
        model="haiku",
        permission_mode="acceptEdits",
    )
    model = model.bind_tools([DuckDuckGoSearchTool()])

    messages = [
        SystemMessage(content="You are a helpful web search assistant."),
        HumanMessage(
            content="Using DuckDuckGo, find out the exact release date of Gemini 3 Pro"
        ),
    ]

    res = await model.ainvoke(messages)

    print("\n=== Tool Calls ===")
    for tc in (
        res.response_metadata.get("internal_tool_calls", [])
        if res.response_metadata
        else []
    ):
        print(tc)

    print("\n=== Content (includes tool output) ===")
    print(res.content)

    if res.response_metadata:
        cost = res.response_metadata.get("total_cost_usd")
        if cost is not None:
            print(f"\nCost: ${cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
