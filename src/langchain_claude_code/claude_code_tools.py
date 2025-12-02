from __future__ import annotations

from enum import Enum


class ClaudeTool(str, Enum):
    ASK_USER_QUESTION = "AskUserQuestion"
    BASH = "Bash"
    BASH_OUTPUT = "BashOutput"
    EDIT = "Edit"
    EXIT_PLAN_MODE = "ExitPlanMode"
    GLOB = "Glob"
    GREP = "Grep"
    KILL_SHELL = "KillShell"
    NOTEBOOK_EDIT = "NotebookEdit"
    READ = "Read"
    SKILL = "Skill"
    SLASH_COMMAND = "SlashCommand"
    TASK = "Task"
    TODO_WRITE = "TodoWrite"
    WEB_FETCH = "WebFetch"
    WEB_SEARCH = "WebSearch"
    WRITE = "Write"


DEFAULT_READ_ONLY = [ClaudeTool.GLOB, ClaudeTool.GREP, ClaudeTool.READ]
DEFAULT_WRITE = [ClaudeTool.EDIT, ClaudeTool.WRITE]
DEFAULT_NETWORK = [ClaudeTool.WEB_FETCH, ClaudeTool.WEB_SEARCH]
DEFAULT_SHELL = [ClaudeTool.BASH, ClaudeTool.BASH_OUTPUT, ClaudeTool.KILL_SHELL]


def normalize_tools(tools: list[str | ClaudeTool]) -> list[str]:
    """Convert enum members/strings to plain strings, preserving order and uniqueness."""
    seen: set[str] = set()
    normalized: list[str] = []
    for tool in tools:
        name = tool.value if isinstance(tool, ClaudeTool) else str(tool)
        if name not in seen:
            seen.add(name)
            normalized.append(name)
    return normalized
