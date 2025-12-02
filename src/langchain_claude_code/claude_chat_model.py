from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any, Callable

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage as LCSystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import Field

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool as sdk_tool,
)


class ClaudeCodeChatModel(BaseChatModel):
    """LangChain chat model wrapping Claude Code Agent SDK.
    
    Uses ClaudeSDKClient for multi-turn conversations with full tool support.
    """

    model: str = Field(default="opus", description="Model to use (opus, sonnet, haiku)")
    fallback_model: str | None = Field(default=None, description="Fallback model")
    system_prompt: str | None = Field(default=None, description="System prompt")
    permission_mode: str = Field(
        default="acceptEdits",
        description="Permission mode: default, acceptEdits, plan, bypassPermissions",
    )
    allowed_tools: list[str] = Field(default_factory=list, description="Allowed tools")
    disallowed_tools: list[str] = Field(default_factory=list, description="Disallowed tools")
    max_turns: int | None = Field(default=None, description="Max conversation turns")
    max_budget_usd: float | None = Field(default=None, description="Max budget in USD")
    cwd: str | Path | None = Field(default=None, description="Working directory")
    include_partial_messages: bool = Field(
        default=False, description="Enable partial message streaming"
    )

    _mcp_servers: dict[str, Any] = {}
    _bound_tools: list[BaseTool] = []
    _sessions: dict[str, ClaudeSDKClient] = {}
    _last_result: ResultMessage | None = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "claude-code-agent"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "permission_mode": self.permission_mode,
            "system_prompt": self.system_prompt,
            "allowed_tools": self.allowed_tools,
        }

    def _build_options(self, **overrides: Any) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions from model config."""
        opts = {
            "model": self.model,
            "permission_mode": self.permission_mode,
            "allowed_tools": list(self.allowed_tools),
            "disallowed_tools": list(self.disallowed_tools),
        }

        if self.fallback_model:
            opts["fallback_model"] = self.fallback_model
        if self.system_prompt:
            opts["system_prompt"] = self.system_prompt
        if self.max_turns is not None:
            opts["max_turns"] = self.max_turns
        if self.max_budget_usd is not None:
            opts["max_budget_usd"] = self.max_budget_usd
        if self.cwd:
            opts["cwd"] = self.cwd
        if self._mcp_servers:
            opts["mcp_servers"] = self._mcp_servers
        if self.include_partial_messages:
            opts["include_partial_messages"] = True

        opts.update(overrides)
        return ClaudeAgentOptions(**opts)

    def _convert_messages(
        self, messages: list[BaseMessage]
    ) -> tuple[str, str | None]:
        """Convert LangChain messages to prompt and system prompt.
        
        Returns:
            Tuple of (prompt, system_prompt)
        """
        system_parts: list[str] = []
        conversation_parts: list[str] = []

        for msg in messages:
            if isinstance(msg, LCSystemMessage):
                system_parts.append(str(msg.content))
            elif isinstance(msg, HumanMessage):
                conversation_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                content = str(msg.content) if msg.content else ""
                if getattr(msg, "tool_calls", None):
                    tool_info = ", ".join(
                        f"{tc['name']}({tc['args']})" for tc in msg.tool_calls
                    )
                    content = f"{content}\n[Tool calls: {tool_info}]" if content else f"[Tool calls: {tool_info}]"
                conversation_parts.append(f"Assistant: {content}")
            elif isinstance(msg, ToolMessage):
                conversation_parts.append(
                    f"Tool ({msg.name}): {msg.content}"
                )

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        prompt = "\n\n".join(conversation_parts) if conversation_parts else ""

        return prompt, system_prompt

    def _parse_assistant_message(
        self, message: AssistantMessage
    ) -> tuple[str, list[dict[str, Any]]]:
        """Parse AssistantMessage content blocks.
        
        Returns:
            Tuple of (text_content, tool_calls)
        """
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in message.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "args": block.input,
                })

        return "\n".join(text_parts), tool_calls

    def _create_ai_message(
        self,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        generation_info: dict[str, Any] | None = None,
    ) -> AIMessage:
        """Create AIMessage with optional tool calls."""
        kwargs: dict[str, Any] = {"content": content}
        if tool_calls:
            kwargs["tool_calls"] = tool_calls
        if generation_info:
            kwargs["response_metadata"] = generation_info
        return AIMessage(**kwargs)

    async def _aquery(
        self,
        prompt: str,
        config: RunnableConfig | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Execute query and return parsed response.
        
        Returns:
            Tuple of (content, tool_calls, generation_info)
        """
        session_id = (config or {}).get("configurable", {}).get("session_id")
        options = self._build_options(**kwargs)

        if session_id and session_id in self._sessions:
            options.continue_conversation = True

        all_text: list[str] = []
        all_tool_calls: list[dict[str, Any]] = []
        generation_info: dict[str, Any] = {}

        async with ClaudeSDKClient(options=options) as client:
            if session_id:
                self._sessions[session_id] = client

            await client.query(prompt)

            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    text, tool_calls = self._parse_assistant_message(msg)
                    if text:
                        all_text.append(text)
                        if run_manager:
                            await run_manager.on_llm_new_token(text)
                    all_tool_calls.extend(tool_calls)

                elif isinstance(msg, ResultMessage):
                    self._last_result = msg
                    generation_info = {
                        "total_cost_usd": msg.total_cost_usd,
                        "duration_ms": msg.duration_ms,
                        "duration_api_ms": msg.duration_api_ms,
                        "num_turns": msg.num_turns,
                        "session_id": msg.session_id,
                        "is_error": msg.is_error,
                    }
                    if msg.usage:
                        generation_info["usage"] = msg.usage

        return "\n".join(all_text), all_tool_calls, generation_info

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation - runs async in event loop."""
        return asyncio.get_event_loop().run_until_complete(
            self._agenerate(messages, stop=stop, run_manager=None, **kwargs)
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation - primary implementation."""
        prompt, system_prompt = self._convert_messages(messages)

        if system_prompt and not self.system_prompt:
            kwargs["system_prompt"] = system_prompt

        content, tool_calls, generation_info = await self._aquery(
            prompt, run_manager=run_manager, **kwargs
        )

        ai_message = self._create_ai_message(content, tool_calls, generation_info)

        if run_manager and ai_message.id is None:
            ai_message.id = f"run-{run_manager.run_id}"

        generation = ChatGeneration(
            message=ai_message,
            generation_info=generation_info,
        )
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Synchronous streaming."""
        loop = asyncio.new_event_loop()
        try:
            async_gen = self._astream(messages, stop=stop, run_manager=None, **kwargs)
            while True:
                try:
                    chunk = loop.run_until_complete(async_gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async streaming - yields chunks as they arrive."""
        prompt, system_prompt = self._convert_messages(messages)

        if system_prompt and not self.system_prompt:
            kwargs["system_prompt"] = system_prompt

        kwargs["include_partial_messages"] = True
        options = self._build_options(**kwargs)

        tool_calls_buffer: list[dict[str, Any]] = []

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)

            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    text, tool_calls = self._parse_assistant_message(msg)

                    if text:
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=text)
                        )
                        if run_manager:
                            await run_manager.on_llm_new_token(text, chunk=chunk)
                        yield chunk

                    tool_calls_buffer.extend(tool_calls)

                elif isinstance(msg, ResultMessage):
                    self._last_result = msg

                    if tool_calls_buffer:
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(
                                content="",
                                tool_calls=tool_calls_buffer,
                            )
                        )

                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content=""),
                        generation_info={
                            "total_cost_usd": msg.total_cost_usd,
                            "duration_ms": msg.duration_ms,
                            "session_id": msg.session_id,
                            "finish_reason": "stop" if not msg.is_error else "error",
                        },
                    )

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        *,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Runnable:
        """Bind LangChain tools to the model via MCP server."""
        sdk_tools = []
        tool_names = []

        for lc_tool in tools:
            schema = self._get_tool_schema(lc_tool)
            sdk_func = self._wrap_langchain_tool(lc_tool, schema)
            sdk_tools.append(sdk_func)
            tool_names.append(lc_tool.name)

        server = create_sdk_mcp_server(
            name="langchain-tools",
            version="1.0.0",
            tools=sdk_tools,
        )

        allowed = [f"mcp__langchain-tools__{name}" for name in tool_names]

        return self.bind(
            _mcp_servers={"langchain-tools": server},
            allowed_tools=list(self.allowed_tools) + allowed,
            _bound_tools=list(tools),
            **kwargs,
        )

    def _get_tool_schema(self, tool: BaseTool) -> dict[str, Any]:
        """Extract JSON schema from LangChain tool."""
        if hasattr(tool, "args_schema") and tool.args_schema:
            return tool.args_schema.model_json_schema()
        return {"type": "object", "properties": {}, "required": []}

    def _wrap_langchain_tool(
        self, tool: BaseTool, schema: dict[str, Any]
    ) -> Callable[..., Any]:
        """Wrap LangChain tool as SDK tool function."""
        props = schema.get("properties", {})
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        param_types = {}
        for name, prop in props.items():
            json_type = prop.get("type", "string")
            param_types[name] = type_map.get(json_type, str)

        @sdk_tool(tool.name, tool.description or "", param_types)
        async def wrapped_tool(args: dict[str, Any]) -> dict[str, Any]:
            try:
                if asyncio.iscoroutinefunction(tool._run):
                    result = await tool._arun(**args)
                else:
                    result = tool._run(**args)
                return {"content": [{"type": "text", "text": str(result)}]}
            except Exception as e:
                return {
                    "content": [{"type": "text", "text": f"Error: {e}"}],
                    "is_error": True,
                }

        return wrapped_tool

    @property
    def last_result(self) -> ResultMessage | None:
        """Get the last ResultMessage with cost/usage info."""
        return self._last_result
