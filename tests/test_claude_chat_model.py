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

    async def test_astream_preserves_full_result_message_fields(self):
        """``_astream``'s result chunk's ``generation_info`` must
        carry the full SDK ``ResultMessage`` field set so callers
        reading ``finished_message.response_metadata`` see the same
        information they'd get from the non-streaming ``_generate``
        path. Previously the streaming path dropped ``stop_reason``,
        ``num_turns``, and ``is_error`` entirely — collapsing
        ``is_error`` into a binary ``finish_reason`` — which made
        the two code paths produce asymmetric AIMessage shapes.

        ``stop_reason`` is set via ``setattr`` because the SDK
        ``>= 0.1.10`` floor this package declares predates that
        field's addition to ``ResultMessage``; on newer SDKs the
        helper picks it up via ``getattr`` without an SDK bump.
        """
        rm = ResultMessage(
            subtype="result",
            duration_ms=10,
            duration_api_ms=5,
            is_error=False,
            num_turns=3,
            session_id="sess-stream-full",
            total_cost_usd=0.001,
            usage={"input_tokens": 10, "output_tokens": 5},
            result=None,
            structured_output=None,
        )
        # Simulate newer-SDK stop_reason field via setattr.
        try:
            rm.stop_reason = "end_turn"  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass
        StubClaudeSDKClient.preset_responses = [
            AssistantMessage(
                content=[TextBlock(text="ok")],
                model="test", parent_tool_use_id=None, error=None,
            ),
            rm,
        ]

        model = ClaudeCodeChatModel()
        with patch(
            "langchain_claude_code.claude_chat_model.ClaudeSDKClient",
            StubClaudeSDKClient,
        ):
            chunks = [
                chunk async for chunk in model._astream([HumanMessage(content="hi")])
            ]

        final = chunks[-1].generation_info
        # Granular SDK fields all present:
        self.assertEqual(final["num_turns"], 3)
        self.assertEqual(final["is_error"], False)
        # LangChain convention finish_reason also present (binary):
        self.assertEqual(final["finish_reason"], "stop")
        # stop_reason is preserved when present on the SDK message.
        # On SDK 0.1.10 (no stop_reason field) the key is omitted —
        # that's a forward-compatibility-only assertion; gate it.
        if hasattr(StubClaudeSDKClient.preset_responses[1], "stop_reason"):
            self.assertEqual(final.get("stop_reason"), "end_turn")
        # Existing fields untouched:
        self.assertEqual(final["total_cost_usd"], 0.001)
        self.assertEqual(final["session_id"], "sess-stream-full")
        self.assertEqual(final["usage"], {"input_tokens": 10, "output_tokens": 5})

    async def test_astream_finish_reason_error_on_is_error_true(self):
        """When ``ResultMessage.is_error`` is True, ``finish_reason``
        renders as ``"error"`` (LangChain convention) and ``is_error``
        is preserved as its own field for callers who need the boolean
        directly."""
        rm = ResultMessage(
            subtype="result",
            duration_ms=1, duration_api_ms=1,
            is_error=True,
            num_turns=50,
            session_id="sess-err",
            total_cost_usd=None,
            usage=None,
            result=None,
            structured_output=None,
        )
        try:
            rm.stop_reason = "max_turns"  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass
        StubClaudeSDKClient.preset_responses = [rm]
        model = ClaudeCodeChatModel()
        with patch(
            "langchain_claude_code.claude_chat_model.ClaudeSDKClient",
            StubClaudeSDKClient,
        ):
            chunks = [
                chunk async for chunk in model._astream([HumanMessage(content="x")])
            ]
        final = chunks[-1].generation_info
        self.assertEqual(final["is_error"], True)
        self.assertEqual(final["finish_reason"], "error")
        self.assertEqual(final["num_turns"], 50)
        if hasattr(StubClaudeSDKClient.preset_responses[0], "stop_reason"):
            self.assertEqual(final.get("stop_reason"), "max_turns")

    async def test_generate_preserves_full_result_message_fields(self):
        """Non-streaming ``_generate`` path mirrors the streaming
        path's generation_info shape — the same helper builds both,
        so the field set is symmetric. Pre-fix, ``_generate``
        preserved ``num_turns`` / ``is_error`` but emitted no
        ``finish_reason``, while ``_astream`` did the opposite."""
        rm = ResultMessage(
            subtype="result",
            duration_ms=8, duration_api_ms=4,
            is_error=False,
            num_turns=2,
            session_id="sess-gen-full",
            total_cost_usd=0.005,
            usage={"output_tokens": 7},
            result=None,
            structured_output=None,
        )
        try:
            rm.stop_reason = "end_turn"  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass
        StubClaudeSDKClient.preset_responses = [
            AssistantMessage(
                content=[TextBlock(text="done")],
                model="test", parent_tool_use_id=None, error=None,
            ),
            rm,
        ]
        model = ClaudeCodeChatModel()
        with patch(
            "langchain_claude_code.claude_chat_model.ClaudeSDKClient",
            StubClaudeSDKClient,
        ):
            res = await model.ainvoke([HumanMessage(content="hi")])
        md = res.response_metadata
        # All SDK fields preserved + finish_reason added per
        # LangChain convention.
        self.assertEqual(md["num_turns"], 2)
        self.assertEqual(md["is_error"], False)
        self.assertEqual(md["finish_reason"], "stop")
        self.assertEqual(md["session_id"], "sess-gen-full")
        if hasattr(rm, "stop_reason"):
            self.assertEqual(md.get("stop_reason"), "end_turn")

    def test_generation_info_from_result_helper(self):
        """Direct unit test of the helper that both paths share:
        produces a consistent generation_info dict from any
        ResultMessage. Omits ``stop_reason`` when the SDK doesn't
        carry it (older SDK) or when present-but-``None``. Omits
        ``usage`` only when empty."""
        from langchain_claude_code.claude_chat_model import (
            _generation_info_from_result,
        )

        msg = ResultMessage(
            subtype="result",
            duration_ms=1, duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s1",
            total_cost_usd=0.0,
            usage=None,                  # falsy → key omitted
            result=None,
            structured_output=None,
        )
        info = _generation_info_from_result(msg)
        self.assertEqual(info["num_turns"], 1)
        self.assertEqual(info["is_error"], False)
        self.assertEqual(info["finish_reason"], "stop")
        self.assertNotIn("stop_reason", info)  # absent on SDK 0.1.10
        self.assertNotIn("usage", info)         # None → omitted



    # ── _install_tool_event_hooks ────────────────────────────────────

    def test_install_tool_event_hooks_registers_three_events(self):
        """The helper must register PreToolUse, PostToolUse, and
        PostToolUseFailure callbacks on the provided options. Each
        event gets exactly one HookMatcher with one callback (ours)
        when options.hooks was None."""
        from claude_agent_sdk import ClaudeAgentOptions

        model = ClaudeCodeChatModel()
        options = ClaudeAgentOptions(model="test")
        events = model._install_tool_event_hooks(options)

        self.assertEqual(events, [])  # No events fired yet
        self.assertIsNotNone(options.hooks)
        self.assertIn("PreToolUse", options.hooks)
        self.assertIn("PostToolUse", options.hooks)
        self.assertIn("PostToolUseFailure", options.hooks)
        for event in ("PreToolUse", "PostToolUse", "PostToolUseFailure"):
            self.assertEqual(len(options.hooks[event]), 1)

    def test_install_tool_event_hooks_preserves_existing(self):
        """User-supplied hooks must be preserved — our callbacks are
        appended to the existing matcher list, not replaced. Critical
        for operators who supply permission gates via PreToolUse."""
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

        async def user_pre(*_a, **_kw):
            return {}

        model = ClaudeCodeChatModel()
        options = ClaudeAgentOptions(
            model="test",
            hooks={"PreToolUse": [HookMatcher(hooks=[user_pre])]},
        )
        model._install_tool_event_hooks(options)

        # PreToolUse should have BOTH the user's hook AND ours.
        self.assertEqual(len(options.hooks["PreToolUse"]), 2)
        # PostToolUse and PostToolUseFailure get ours only.
        self.assertEqual(len(options.hooks["PostToolUse"]), 1)
        self.assertEqual(len(options.hooks["PostToolUseFailure"]), 1)

    async def test_install_tool_event_hooks_callbacks_record_events(self):
        """Invoke the registered callbacks directly with synthetic
        SDK inputs and verify they append correctly-shaped events to
        the returned list, paired by tool_use_id, with monotonic
        timestamps."""
        from claude_agent_sdk import ClaudeAgentOptions

        model = ClaudeCodeChatModel()
        options = ClaudeAgentOptions(model="test")
        events = model._install_tool_event_hooks(options)

        pre_cb = options.hooks["PreToolUse"][0].hooks[0]
        post_cb = options.hooks["PostToolUse"][0].hooks[0]
        fail_cb = options.hooks["PostToolUseFailure"][0].hooks[0]

        await pre_cb(
            {"tool_name": "Bash", "tool_input": {"command": "ls"}},
            "toolu_a", None,
        )
        await post_cb(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "ls"},
                "tool_response": {"output": "f.txt"},
            },
            "toolu_a", None,
        )
        await fail_cb(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "/bin/false"},
                "error": "exited 1",
            },
            "toolu_b", None,
        )

        self.assertEqual(len(events), 3)
        call, ok_result, fail_result = events
        self.assertEqual(call["type"], "tool_call")
        self.assertEqual(call["tool_use_id"], "toolu_a")
        self.assertEqual(call["name"], "Bash")
        self.assertEqual(call["input"], {"command": "ls"})

        self.assertEqual(ok_result["type"], "tool_result")
        self.assertEqual(ok_result["tool_use_id"], "toolu_a")
        self.assertEqual(ok_result["result"], {"output": "f.txt"})
        self.assertFalse(ok_result["is_error"])

        self.assertEqual(fail_result["type"], "tool_result")
        self.assertEqual(fail_result["tool_use_id"], "toolu_b")
        self.assertEqual(fail_result["error"], "exited 1")
        self.assertTrue(fail_result["is_error"])

        # Monotonic ordering — events recorded in registration order.
        self.assertGreaterEqual(ok_result["ts_mono_ns"], call["ts_mono_ns"])
        self.assertGreaterEqual(fail_result["ts_mono_ns"], ok_result["ts_mono_ns"])

    async def test_aquery_attaches_tool_events_to_generation_info(self):
        """End-to-end with stubbed SDK: after _aquery, the captured
        tool_events list must be on generation_info. We manually invoke
        the registered callbacks via a stub that captures the hooks
        at __init__ time and fires them between the AssistantMessage
        and ResultMessage."""
        class HookFiringStub(StubClaudeSDKClient):
            async def receive_response(self):
                # Fire hooks the way the real SDK would, mid-loop.
                pre = self.options.hooks["PreToolUse"][0].hooks[0]
                post = self.options.hooks["PostToolUse"][0].hooks[0]
                await pre(
                    {"tool_name": "Read", "tool_input": {"file_path": "/a"}},
                    "toolu_r1", None,
                )
                await post(
                    {
                        "tool_name": "Read",
                        "tool_input": {"file_path": "/a"},
                        "tool_response": "contents",
                    },
                    "toolu_r1", None,
                )
                for msg in self.responses:
                    yield msg

        HookFiringStub.preset_responses = [
            AssistantMessage(
                content=[TextBlock(text="done")], model="test",
                parent_tool_use_id=None, error=None,
            ),
            ResultMessage(
                subtype="result", duration_ms=1, duration_api_ms=1,
                is_error=False, num_turns=1, session_id="s",
                total_cost_usd=0.0, usage=None,
                result=None, structured_output=None,
            ),
        ]

        model = ClaudeCodeChatModel()
        with patch(
            "langchain_claude_code.claude_chat_model.ClaudeSDKClient",
            HookFiringStub,
        ):
            res = await model.ainvoke([HumanMessage(content="hi")])

        tool_events = res.response_metadata.get("tool_events")
        self.assertIsNotNone(tool_events)
        self.assertEqual(len(tool_events), 2)
        self.assertEqual(tool_events[0]["type"], "tool_call")
        self.assertEqual(tool_events[0]["name"], "Read")
        self.assertEqual(tool_events[1]["type"], "tool_result")
        self.assertEqual(tool_events[1]["tool_use_id"], "toolu_r1")



if __name__ == "__main__":
    unittest.main()
