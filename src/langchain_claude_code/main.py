import asyncio
from langchain_claude_code import ClaudeCodeChatModel
from langchain_core.messages import HumanMessage, SystemMessage


async def main():
    model = ClaudeCodeChatModel(
        model="sonnet",
        permission_mode="acceptEdits",
    )

    messages = [
        SystemMessage(content="You are a helpful coding assistant."),
        HumanMessage(content="What is 2 + 2?"),
    ]

    result = await model.ainvoke(messages)
    print(f"Response: {result.content}")

    if model.last_result:
        print(f"Cost: ${model.last_result.total_cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
