from __future__ import annotations

import asyncio
import contextvars
import queue
import threading
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
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr
from pydantic.errors import PydanticInvalidForJsonSchema

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    create_sdk_mcp_server,
    tool as sdk_tool,
)
from langchain_claude_code.claude_code_tools import ClaudeTool, normalize_tools


def _generation_info_from_result(msg: "ResultMessage") -> dict[str, Any]:
    """Build a ``generation_info`` dict from an SDK ``ResultMessage``.

    Mirrors the SDK's per-request metadata onto LangChain's
    ``generation_info`` so downstream consumers can read it off the
    final AIMessage's ``response_metadata``. Used by both ``_generate``
    (non-streaming) and ``_astream`` (streaming) so the two paths
    surface the same field set — previously they diverged: the
    non-streaming path preserved ``num_turns`` / ``is_error`` but
    missed ``finish_reason``, while the streaming path emitted
    ``finish_reason`` but dropped ``num_turns`` / ``is_error``
    entirely. Neither path preserved ``stop_reason``.

    Field shape (all keys present unless explicitly noted):

      - ``total_cost_usd``, ``duration_ms``, ``duration_api_ms``,
        ``session_id``                              — SDK fields
      - ``num_turns``, ``is_error``                 — SDK fields
      - ``finish_reason``                           — LangChain
        convention; ``"error"`` when ``msg.is_error`` else ``"stop"``
      - ``stop_reason``                             — granular SDK
        reason (``"end_turn"``, ``"max_turns"``, ``"max_tokens"``,
        etc.); only included when present (newer SDK only) AND
        non-``None``. Access is ``getattr``-guarded so the helper
        works on the SDK ``>= 0.1.10`` floor this package declares.
      - ``usage``                                   — only when
        ``msg.usage`` is non-empty
    """
    info: dict[str, Any] = {
        "total_cost_usd": msg.total_cost_usd,
        "duration_ms": msg.duration_ms,
        "duration_api_ms": msg.duration_api_ms,
        "session_id": msg.session_id,
        "num_turns": msg.num_turns,
        "is_error": msg.is_error,
        "finish_reason": "error" if msg.is_error else "stop",
    }
    # ``stop_reason`` was added to ResultMessage in a later SDK
    # release; use getattr so the helper stays compatible with the
    # >= 0.1.10 floor pinned in pyproject.toml.
    stop_reason = getattr(msg, "stop_reason", None)
    if stop_reason is not None:
        info["stop_reason"] = stop_reason
    if msg.usage:
        info["usage"] = msg.usage
    return info


class ClaudeCodeChatModel(BaseChatModel):
    """LangChain chat model wrapping Claude Code Agent SDK. 
    
    Uses ClaudeSDKClient for multi-turn conversations with full tool support.
    """

    model: str = Field(default="opus", description="Model to use (opus, sonnet, haiku)")
    fallback_model: str | None = Field(default=None, description="Fallback model")
    system_prompt: str | None = Field(default=None, description="System prompt")
    permission_mode: str = Field(
        default="default",
        description="Permission mode: default, acceptEdits, plan, bypassPermissions",
    )
    allowed_tools: list[str | ClaudeTool] = Field(default_factory=list, description="Allowed tools")
    disallowed_tools: list[str | ClaudeTool] = Field(default_factory=list, description="Disallowed tools")
    max_turns: int | None = Field(default=None, description="Max conversation turns")
    max_budget_usd: float | None = Field(default=None, description="Max budget in USD")
    cwd: str | Path | None = Field(default=None, description="Working directory")
    include_partial_messages: bool = Field(
        default=False, description="Enable partial message streaming"
    )
    api_key: str | None = Field(default=None, description="Anthropic API key")
    oauth_token: str | None = Field(default=None, description="OAuth token")

    _mcp_servers: dict[str, Any] = {}
    _bound_tools: list[BaseTool] = []
    _last_result: ResultMessage | None = None
    _tool_results_var: contextvars.ContextVar | None = PrivateAttr(
        default_factory=lambda: contextvars.ContextVar("claude_code_tool_results")
    )

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

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Ensure RunnableConfig is available to downstream calls."""
        config = ensure_config(config)
        kwargs.setdefault("_config", config)
        return super().invoke(input, config=config, stop=stop, **kwargs)

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Async invoke that propagates RunnableConfig via context."""
        config = ensure_config(config)
        kwargs.setdefault("_config", config)
        return await super().ainvoke(input, config=config, stop=stop, **kwargs)

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        """Stream while keeping config in context for session support."""
        config = ensure_config(config)
        kwargs.setdefault("_config", config)
        yield from super().stream(input, config=config, stop=stop, **kwargs)

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        """Async stream while propagating config context."""
        config = ensure_config(config)
        kwargs.setdefault("_config", config)
        async for chunk in super().astream(
            input, config=config, stop=stop, **kwargs
        ):
            yield chunk

    def _build_options(self, **overrides: Any) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions from model config."""
        opts = {
            "model": self.model,
            "permission_mode": self.permission_mode,
            "allowed_tools": normalize_tools(list(self.allowed_tools)),
            "disallowed_tools": normalize_tools(list(self.disallowed_tools)),
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

        env: dict[str, str] = {}
        if self.api_key:
            env["ANTHROPIC_API_KEY"] = self.api_key
        if self.oauth_token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = self.oauth_token
        if env:
            opts["env"] = env

        if self.include_partial_messages:
            opts["include_partial_messages"] = True
        # When tools are allowed/bound, ensure partial messages so tool results stream back.
        if not opts.get("include_partial_messages") and (
            self._bound_tools or opts.get("allowed_tools")
        ):
            opts["include_partial_messages"] = True

        allowed_keys = set(ClaudeAgentOptions.__dataclass_fields__.keys())
        for key, value in overrides.items():
            if key.startswith("_"):
                if key == "_mcp_servers":
                    opts["mcp_servers"] = value
                continue
            if key in allowed_keys:
                if key in {"allowed_tools", "disallowed_tools"}:
                    opts[key] = normalize_tools(list(value or []))
                else:
                    opts[key] = value

        return ClaudeAgentOptions(**opts)

    def _convert_messages(
        self,
        messages: list[BaseMessage],
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
        self,
        message: AssistantMessage,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        """Parse AssistantMessage content blocks.
        
        Returns:
            Tuple of (text_content, tool_calls, tool_results)
        """
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for block in message.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ToolResultBlock):
                content_items: list[str] = []
                if isinstance(block.content, str):
                    content_items.append(block.content)
                elif isinstance(block.content, list):
                    for item in block.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            content_items.append(str(item.get("text", "")))
                        else:
                            content_items.append(str(item))

                if content_items:
                    text_parts.append("\n".join(content_items))

                tool_results.append(
                    {
                        "tool_use_id": getattr(block, "tool_use_id", None),
                        "content": block.content,
                        "is_error": block.is_error,
                    }
                )
            elif isinstance(block, ToolUseBlock):
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "args": block.input,
                })

        return "\n".join(text_parts), tool_calls, tool_results

    def _create_ai_message(
        self,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        generation_info: dict[str, Any] | None = None,
    ) -> AIMessage:
        """Create AIMessage with optional tool calls.
        
        Note: tool_calls are stored in response_metadata, NOT as AIMessage.tool_calls.
        Claude Code executes tools internally, so exposing tool_calls would cause
        LangGraph to attempt re-execution of already-completed tool calls.
        """
        kwargs: dict[str, Any] = {"content": content}
        if generation_info:
            kwargs["response_metadata"] = generation_info
        if tool_calls:
            kwargs.setdefault("response_metadata", {})
            kwargs["response_metadata"]["internal_tool_calls"] = tool_calls
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
        cfg = ensure_config(config) if config is not None else None
        session_id = kwargs.pop("session_id", None) or kwargs.pop("resume", None)
        if cfg:
            session_id = session_id or cfg.get("configurable", {}).get("session_id")

        options = self._build_options(**kwargs)

        if session_id:
            options.resume = session_id
            options.continue_conversation = True

        all_text: list[str] = []
        all_tool_calls: list[dict[str, Any]] = []
        all_tool_results: list[dict[str, Any]] = []
        generation_info: dict[str, Any] = {}
        tool_results_token = self._tool_results_var.set([])

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)

            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    text, tool_calls, tool_results = self._parse_assistant_message(msg)
                    if text:
                        all_text.append(text)
                        if run_manager:
                            await run_manager.on_llm_new_token(text)
                    all_tool_calls.extend(tool_calls)
                    all_tool_results.extend(tool_results)

                elif isinstance(msg, ResultMessage):
                    self._last_result = msg
                    generation_info = _generation_info_from_result(msg)

        captured = self._tool_results_var.get()
        if captured:
            all_tool_results.extend(captured)
        self._tool_results_var.reset(tool_results_token)

        if all_tool_results:
            generation_info["tool_results"] = all_tool_results

        return "\n".join(all_text), all_tool_calls, generation_info

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation - runs async in event loop."""
        return self._run_sync(self._agenerate(messages, stop=stop, run_manager=None, **kwargs))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation - primary implementation."""
        config = kwargs.pop("_config", None)
        prompt, system_prompt = self._convert_messages(messages)

        if system_prompt and not self.system_prompt:
            kwargs["system_prompt"] = system_prompt

        content, tool_calls, generation_info = await self._aquery(
            prompt, config=config, run_manager=run_manager, **kwargs
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
        """Synchronous streaming using background thread to preserve anyio task affinity."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "Cannot use synchronous streaming while an event loop is running. Use 'astream' instead."
            )

        chunk_queue: queue.Queue[ChatGenerationChunk | BaseException | None] = queue.Queue()

        def run_async_stream() -> None:
            async def produce() -> None:
                try:
                    async for chunk in self._astream(
                        messages, stop=stop, run_manager=None, **kwargs
                    ):
                        chunk_queue.put(chunk)
                    chunk_queue.put(None)
                except BaseException as exc:
                    chunk_queue.put(exc)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(produce())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

        thread = threading.Thread(target=run_async_stream, daemon=True)
        thread.start()

        try:
            while True:
                item = chunk_queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            thread.join(timeout=5.0)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async streaming - yields chunks as they arrive."""
        config = kwargs.pop("_config", None)
        prompt, system_prompt = self._convert_messages(messages)

        if system_prompt and not self.system_prompt:
            kwargs["system_prompt"] = system_prompt

        kwargs["include_partial_messages"] = True
        cfg = ensure_config(config) if config is not None else None
        session_id = kwargs.pop("session_id", None) or kwargs.pop("resume", None)
        if cfg:
            session_id = session_id or cfg.get("configurable", {}).get("session_id")

        options = self._build_options(**kwargs)

        if session_id:
            options.resume = session_id
            options.continue_conversation = True

        tool_calls_buffer: list[dict[str, Any]] = []
        tool_results_buffer: list[dict[str, Any]] = []

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)

            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    text, tool_calls, tool_results = self._parse_assistant_message(msg)

                    if text:
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=text)
                        )
                        if run_manager:
                            await run_manager.on_llm_new_token(text, chunk=chunk)
                        yield chunk

                    tool_calls_buffer.extend(tool_calls)
                    tool_results_buffer.extend(tool_results)

                elif isinstance(msg, ResultMessage):
                    self._last_result = msg

                    generation_info: dict[str, Any] = (
                        _generation_info_from_result(msg)
                    )
                    if tool_calls_buffer:
                        generation_info["internal_tool_calls"] = tool_calls_buffer
                    if tool_results_buffer:
                        generation_info["internal_tool_results"] = tool_results_buffer

                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content="", chunk_position="last"),
                        generation_info=generation_info,
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

        return self.model_copy(
            update={
                "_mcp_servers": {"langchain-tools": server},
                "allowed_tools": normalize_tools(list(self.allowed_tools)) + allowed,
                "_bound_tools": list(tools),
            }
        ).bind(**kwargs)

    def _get_tool_schema(self, tool: BaseTool) -> dict[str, Any]:
        """Extract JSON schema from LangChain tool."""
        if hasattr(tool, "args_schema") and tool.args_schema:
            try:
                return tool.args_schema.model_json_schema()
            except PydanticInvalidForJsonSchema:
                # Some tool arg models include callables that cannot be rendered to JSON Schema.
                return {"type": "object", "properties": {}, "required": []}
            except Exception:
                return {"type": "object", "properties": {}, "required": []}
        return {"type": "object", "properties": {}, "required": []}

    def _wrap_langchain_tool(
        self,
        tool: BaseTool,
        schema: dict[str, Any],
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
                if hasattr(tool, "_arun") and asyncio.iscoroutinefunction(tool._arun):
                    result = await tool._arun(**args)
                else:
                    result = tool._run(**args)

                captured = self._tool_results_var.get(None) if self._tool_results_var else None
                if captured is not None:
                    captured.append(
                        {
                            "name": tool.name,
                            "args": args,
                            "result": result,
                        }
                    )

                return {"content": [{"type": "text", "text": str(result)}]}
            except Exception as e:
                return {
                    "content": [{"type": "text", "text": f"Error: {e}"}],
                    "is_error": True,
                }

        return wrapped_tool

    def resume_from_thread(self, thread_id: str) -> Runnable:
        """Return a bound model that resumes the provider session using a LangGraph thread_id."""
        return self.with_config(config={"configurable": {"session_id": thread_id}})

    def enable_tools(self, tools: list[str | ClaudeTool]) -> Runnable:
        """Return a new model with the given tools added to the allow list."""
        merged = normalize_tools(normalize_tools(list(self.allowed_tools)) + list(tools))
        return self.model_copy(update={"allowed_tools": merged})

    @property
    def last_result(self) -> ResultMessage | None:
        """Get the last ResultMessage with cost/usage info."""
        return self._last_result

    def _run_sync(self, coro: asyncio.Future) -> Any:
        """Run coroutine safely from sync context without relying on global loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        raise RuntimeError(
            "Cannot call synchronous ClaudeCodeChatModel methods while an event loop is running. Use 'ainvoke' instead."
        )