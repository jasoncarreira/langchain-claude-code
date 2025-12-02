# Repository Guidelines

## Project Structure & Module Organization
- Source code lives in `src/langchain_claude_code/` (chat model, tool catalog, examples like `main.py` and `deep.py`).
- Tests are in `tests/` and must accompany any new behavior.
- Repository rules live in `.cursor/rules/` (project conventions and agent notes).

## Build, Test, and Development Commands
- `python -m unittest discover -v tests` — run the full test suite.
- `uv run src/langchain_claude_code/main.py` — example invocation with DuckDuckGo tool.
- `uv run src/langchain_claude_code/deep.py` — DeepAgent demo.
- Prefer `uv` (uses `uv.lock`) for reproducible environments; `pip install -e .` works for local edits.

## Coding Style & Naming Conventions
- Python 3.12+, async-first APIs; keep sync shims minimal.
- Pydantic models/fields for config; prefer `model_copy(update=...)` over mutating state.
- Use `normalize_tools` and `ClaudeTool` enums when handling allowlists; avoid raw strings.
- Comments should be short and purposeful; keep files ASCII.

## Testing Guidelines
- Framework: `unittest`; locate tests under `tests/` with `test_*.py` naming.
- Every new feature or bugfix must ship with a test (policy captured in `.cursor/rules/langchain-claude-code.mdc`).
- For tool/agent behavior, add assertions for response metadata (`tool_results`, `internal_tool_calls`) and streamed chunks when relevant.

## Commit & Pull Request Guidelines
- Write concise commits summarizing the change and its scope; use present tense (e.g., "Add tool result capture to metadata").
- PRs should include: what changed, why, testing performed (`python -m unittest ...`), and any new docs or examples touched.
- Link issues or tickets when available; include screenshots/log snippets for user-visible changes (CLI output is sufficient).

## Agent-Specific Instructions
- When binding LangChain tools, use `bind_tools([...])` or `enable_tools([...])`; tool outputs should surface in `response_metadata.tool_results`.
- To align LangGraph threads with provider sessions, use `resume_from_thread(thread_id)`; keep `allowed_tools` explicit for safety.
- Always enable tests for new agent flows; an untested feature is treated as broken by default.
