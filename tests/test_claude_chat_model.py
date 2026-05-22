import asyncio
import unittest
from typing import Any, Callable

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
from claude_agent_sdk import ToolResultBlock
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import BaseModel, PrivateAttr
from unittest.mock import AsyncMock, patch

from langchain_claude_code import ClaudeCodeChatModel
from langchain_claude_code.claude_code_tools import ClaudeTool, normalize_tools


class StubClaudeSDKClient:
    """Minimal async stub for ClaudeSDKClient used in tests."""

    preset_responses: list[Any] = []
    instances: list["StubClaudeSDKClient"] = []

    def __init__(self, options=None, transport=None):
        self.options = options
        self.transport = transport
        self.queries: list[str] = []
        self.responses = list(type(self).preset_responses)
        type(self).instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.preset_responses = []
        cls.instances = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def query(self, prompt: str):
        self.queries.append(prompt)

    async def receive_response(self):
        for msg in self.responses:
            yield msg


class AsyncArgs(BaseModel):
    text: str


class AsyncFirstTool(BaseTool):
    name: str = "echo"
    description: str = "Echo the provided text"
    args_schema: type[BaseModel] = AsyncArgs

    _called: str | None = PrivateAttr(default=None)

    def __init__(self):
        super().__init__()

    async def _arun(self, text: str) -> str:
        self._called = "arun"
        return text.upper()

    def _run(self, text: str) -> str:  # pragma: no cover - not expected to be used
        self._called = "run"
        return text


class ClaudeChatModelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        StubClaudeSDKClient.reset()

    async def test_session_resume_from_configurable(self):
        StubClaudeSDKClient.preset_responses = [
            AssistantMessage(content=[TextBlock(text="hello")], model="test", parent_tool_use_id=None, error=None),
            ResultMessage(
                subtype="result",
                duration_ms=10,
                duration_api_ms=8,
                is_error=False,
                num_turns=1,
                session_id="sess-123",
                total_cost_usd=0.01,
                usage={"input_tokens": 5},
                result=None,
                structured_output=None,
            ),
        ]

        model = ClaudeCodeChatModel(model="sonnet")
        messages = [HumanMessage(content="hi")]

        with patch(
            "langchain_claude_code.claude_chat_model.ClaudeSDKClient",
            StubClaudeSDKClient,
        ):
            response = await model.ainvoke(
                messages, config={"configurable": {"session_id": "sess-123"}}
            )

        client = StubClaudeSDKClient.instances[-1]
        self.assertEqual(client.options.resume, "sess-123")
        self.assertTrue(client.options.continue_conversation)
        self.assertEqual(model.last_result.session_id, "sess-123")
        self.assertEqual(response.content, "hello")

    def test_wrap_langchain_tool_prefers_arun(self):
        tool = AsyncFirstTool()
        model = ClaudeCodeChatModel()
        wrapped = model._wrap_langchain_tool(
            tool, tool.args_schema.model_json_schema()
        )

        result = asyncio.run(wrapped.handler({"text": "ping"}))

        self.assertEqual(tool._called, "arun")
        self.assertEqual(result["content"][0]["text"], "PING")

    async def test_resume_from_thread_sets_session_id(self):
        StubClaudeSDKClient.preset_responses = [
            AssistantMessage(content=[TextBlock(text="ok")], model="test", parent_tool_use_id=None, error=None),
            ResultMessage(
                subtype="result",
                duration_ms=5,
                duration_api_ms=3,
                is_error=False,
                num_turns=1,
                session_id="thread-99",
                total_cost_usd=0.0,
                usage=None,
                result=None,
                structured_output=None,
            ),
        ]

        base = ClaudeCodeChatModel(model="sonnet")
        model = base.resume_from_thread("thread-99")

        with patch(
            "langchain_claude_code.claude_chat_model.ClaudeSDKClient",
            StubClaudeSDKClient,
        ):
            msg = await model.ainvoke([HumanMessage(content="hi")])

        client = StubClaudeSDKClient.instances[-1]
        self.assertEqual(client.options.resume, "thread-99")
        self.assertEqual(client.options.continue_conversation, True)
        self.assertEqual(msg.content, "ok")

    def test_tool_schema_falls_back_on_invalid_json_schema(self):
        class BadArgs(BaseModel):
            fn: Callable[[str], str]

        class BadTool(BaseTool):
            name: str = "bad"
            description: str = "bad"
            args_schema: type[BaseModel] = BadArgs

            def _run(self, fn):  # pragma: no cover
                return fn("x")

        model = ClaudeCodeChatModel()
        schema = model._get_tool_schema(BadTool())

        self.assertEqual(schema.get("properties", {}), {})

    def test_build_options_filters_private_kwargs(self):
        model = ClaudeCodeChatModel()
        opts = model._build_options(_mcp_servers={"bad": "value"}, bogus=1)

        self.assertFalse(hasattr(opts, "_mcp_servers"))
        self.assertFalse(hasattr(opts, "bogus"))

    def test_enable_tools_and_enum_normalization(self):
        model = ClaudeCodeChatModel(allowed_tools=[ClaudeTool.GREP])
        bound = model.enable_tools([ClaudeTool.WEB_FETCH, "Bash"])

        self.assertIn("Grep", bound.allowed_tools)
        self.assertIn("WebFetch", bound.allowed_tools)
        self.assertIn("Bash", bound.allowed_tools)
        self.assertEqual(len(bound.allowed_tools), 3)

    def test_normalize_tools_dedupes(self):
        names = normalize_tools([ClaudeTool.BASH, "Bash", "WebFetch", ClaudeTool.WEB_FETCH])
        self.assertEqual(names, ["Bash", "WebFetch"])

    async def test_bind_tools_sets_mcp_and_allowlist(self):
        class EchoTool(BaseTool):
            name: str = "echo"
            description: str = "echo"

            def _run(self, text: str = "hi"):  # pragma: no cover
                return text

        StubClaudeSDKClient.preset_responses = [
            AssistantMessage(content=[TextBlock(text="ok")], model="test", parent_tool_use_id=None, error=None),
            ResultMessage(
                subtype="result",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="sess-allow",
                total_cost_usd=0,
                usage=None,
                result=None,
                structured_output=None,
            ),
        ]

        model = ClaudeCodeChatModel()
        bound = model.bind_tools([EchoTool()])

        with patch(
            "langchain_claude_code.claude_chat_model.ClaudeSDKClient",
            StubClaudeSDKClient,
        ):
            await bound.ainvoke([HumanMessage(content="hi")])

        client = StubClaudeSDKClient.instances[-1]
        self.assertIn("langchain-tools", client.options.mcp_servers)
        self.assertIn("mcp__langchain-tools__echo", client.options.allowed_tools)

    def test_bind_tools_skips_runtime_injected_args(self):
        """Tools whose underlying function takes a langgraph-injected
        parameter (notably ``ToolRuntime``) can't be bridged through
        MCP — the bridge calls ``tool._arun(**args)`` directly and
        the injection argument never gets populated. Such tools
        should be filtered out of the MCP allowlist; the framework's
        native tool path still serves them.

        The deepagents filesystem middleware's ``read_file`` tool is
        the motivating real-world case — it has signature
        ``async def async_read_file(file_path, runtime: ToolRuntime[...],
        ...)`` and was producing ``missing 1 required positional
        argument: 'runtime'`` on every invocation through the bridge.
        """
        try:
            from langgraph.prebuilt.tool_node import ToolRuntime
        except ImportError:
            self.skipTest("langgraph not installed")

        class InjectedTool(BaseTool):
            name: str = "needs_runtime"
            description: str = "tool that takes ToolRuntime injection"

            def _run(self, runtime: ToolRuntime[Any, dict[str, Any]], path: str = "") -> str:  # noqa: ARG002
                return path  # pragma: no cover

        class PlainTool(BaseTool):
            name: str = "echo"
            description: str = "plain tool, no injection"

            def _run(self, text: str = "hi") -> str:
                return text  # pragma: no cover

        from langchain_claude_code.claude_chat_model import (
            _has_runtime_injected_args,
        )
        # Detection at the underlying helper level.
        self.assertTrue(_has_runtime_injected_args(InjectedTool()))
        self.assertFalse(_has_runtime_injected_args(PlainTool()))

        # Detection wired into bind_tools — only the plain tool lands
        # in the allowlist; the injected one is silently skipped.
        # ``bind_tools`` returns a RunnableBinding(bound=ModelCopy,
        # kwargs=...) so allowlist mutations live on ``bound.bound``.
        model = ClaudeCodeChatModel()
        bound = model.bind_tools([InjectedTool(), PlainTool()])
        underlying = bound.bound
        self.assertNotIn(
            "mcp__langchain-tools__needs_runtime",
            underlying.allowed_tools,
        )
        self.assertIn(
            "mcp__langchain-tools__echo",
            underlying.allowed_tools,
        )

    def test_generate_uses_asyncio_run_when_no_loop(self):
        model = ClaudeCodeChatModel()
        messages = [HumanMessage(content="hi")]
        fake_result = ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="ok"), generation_info={})
            ]
        )

        with patch.object(model, "_agenerate", AsyncMock(return_value=fake_result)) as agenerate_mock:
            with patch("langchain_claude_code.claude_chat_model.asyncio.run") as run_mock:
                def run_coro(coro):
                    loop = asyncio.new_event_loop()
                    try:
                        return loop.run_until_complete(coro)
                    finally:
                        loop.close()

                run_mock.side_effect = run_coro
                result = model._generate(messages)

        self.assertIs(result, fake_result)
        agenerate_mock.assert_called_once()
        run_mock.assert_called_once()

    async def test_astream_emits_usage_metadata(self):
        StubClaudeSDKClient.preset_responses = [
            AssistantMessage(content=[TextBlock(text="chunk")], model="test", parent_tool_use_id=None, error=None),
            ResultMessage(
                subtype="result",
                duration_ms=12,
                duration_api_ms=7,
                is_error=False,
                num_turns=1,
                session_id="sess-meta",
                total_cost_usd=0.02,
                usage={"output_tokens": 3},
                result=None,
                structured_output=None,
            ),
        ]

        model = ClaudeCodeChatModel()
        with patch(
            "langchain_claude_code.claude_chat_model.ClaudeSDKClient",
            StubClaudeSDKClient,
        ):
            chunks = [
                chunk async for chunk in model._astream([HumanMessage(content="hi")])
            ]

        self.assertEqual(chunks[0].message.content, "chunk")
        final_info = chunks[-1].generation_info
        self.assertEqual(final_info["duration_api_ms"], 7)
        self.assertEqual(final_info["usage"], {"output_tokens": 3})
        self.assertEqual(model.last_result.session_id, "sess-meta")
        self.assertEqual(chunks[-1].message.chunk_position, "last")

    async def test_tool_result_blocks_append_to_content(self):
        StubClaudeSDKClient.preset_responses = [
            AssistantMessage(
                content=[
                    ToolResultBlock(tool_use_id="t1", content="raw result", is_error=None),
                    TextBlock(text="done"),
                ],
                model="test",
                parent_tool_use_id=None,
                error=None,
            ),
            ResultMessage(
                subtype="result",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="sess-tool",
                total_cost_usd=0.0,
                usage=None,
                result=None,
                structured_output=None,
            ),
        ]

        model = ClaudeCodeChatModel()
        with patch(
            "langchain_claude_code.claude_chat_model.ClaudeSDKClient",
            StubClaudeSDKClient,
        ):
            res = await model.ainvoke([HumanMessage(content="hi")])

        self.assertIn("raw result", res.content)
        self.assertTrue(res.content.endswith("done"))
        self.assertIn("tool_results", res.response_metadata)
        self.assertEqual(res.response_metadata["tool_results"][0]["tool_use_id"], "t1")


if __name__ == "__main__":
    unittest.main()
