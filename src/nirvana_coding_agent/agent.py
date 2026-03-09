from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    FilesystemFileSearchMiddleware,
    ShellToolMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
)
from langchain.agents.middleware.shell_tool import HostExecutionPolicy
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.base import BaseCheckpointSaver

from .config import Settings
from .planning import PLAN_RELATIVE_PATH, PlanFileMiddleware

CODING_TODO_SYSTEM_PROMPT = f"""
## Required planning workflow

You are operating as a coding agent. For any request that involves coding, debugging,
environment setup, shell commands, repository inspection, or file changes, your first
meaningful action must be to call `write_todos`.

Create a short, concrete plan before using any other non-trivial tool. Mark the first
task as `in_progress` immediately. Keep the list up to date as you work.

Every `write_todos` call is mirrored into `{PLAN_RELATIVE_PATH}`. After the first todo
update for a coding task, read `{PLAN_RELATIVE_PATH}` before taking the next
non-planning action. Use that file as the canonical checklist and work through the
items one by one.

Writing the todo list is only the planning step. After creating or updating it, continue
executing the task with the other tools until the user's request is actually complete.

If a todo list exists for the current task, you must call `write_todos` one final time
immediately before giving your final answer. That final update must reflect the actual
current state of the work. If the task is complete, mark all completed items as
`completed` before responding to the user.

After that final todo update, read `{PLAN_RELATIVE_PATH}` one more time, double-check
that every item is complete, delete `{PLAN_RELATIVE_PATH}`, and only then give the
final answer to the user.

You may skip `write_todos` only for purely conversational or extremely trivial requests
that do not require tools or file changes.
""".strip()

@dataclass(frozen=True)
class ShellRuntime:
    mode: str
    enforcement: str
    warning: str | None = None


def build_agent(
    settings: Settings,
    workspace_root: Path,
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
):
    workspace_root = workspace_root.expanduser().resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)

    execution_policy = _build_execution_policy(settings)
    shell_runtime = ShellRuntime(
        mode="host",
        enforcement="Shell commands run on the host machine with the workspace as the working directory.",
    )

    tools = [_build_web_search_tool(settings)]

    middleware = [
        PlanFileMiddleware(workspace_root),
        TodoListMiddleware(
            system_prompt=CODING_TODO_SYSTEM_PROMPT,
        ),
        SummarizationMiddleware(
            model=_build_model(
                deployment_name=settings.azure_openai_summary_deployment,
                model_name=settings.summary_model,
                settings=settings,
                enable_parallel_tool_calls=False,
            ),
            trigger=("fraction", settings.summary_trigger_fraction),
            keep=("messages", settings.summary_keep_messages),
        ),
        FilesystemFileSearchMiddleware(root_path=str(workspace_root)),
        ShellToolMiddleware(
            workspace_root=workspace_root,
            execution_policy=execution_policy,
            shell_command=(settings.shell_program,),
            env={"WORKSPACE_ROOT": str(workspace_root)},
            tool_description=(
                "Run shell commands in the coding workspace. The current working directory "
                f"is the workspace root at {workspace_root}."
            ),
        ),
    ]

    agent = create_agent(
        model=_build_model(
            deployment_name=settings.azure_openai_deployment,
            model_name=settings.model,
            settings=settings,
        ),
        tools=tools,
        system_prompt=_build_system_prompt(
            workspace_root=workspace_root,
            shell_runtime=shell_runtime,
        ),
        middleware=middleware,
        checkpointer=checkpointer,
        name="zain",
    )

    return agent, shell_runtime


def _build_model(
    *,
    deployment_name: str | None,
    model_name: str,
    settings: Settings,
    enable_parallel_tool_calls: bool = False,
) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        model=model_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        azure_deployment=deployment_name,
        temperature=0,
        timeout=settings.request_timeout_seconds,
        max_retries=2,
        output_version=settings.output_version,
        use_responses_api=True,
        profile={"max_input_tokens": settings.context_window_tokens},
        model_kwargs={"parallel_tool_calls": enable_parallel_tool_calls},
    )


def _build_execution_policy(settings: Settings) -> HostExecutionPolicy:
    return HostExecutionPolicy(
        command_timeout=settings.shell_timeout_seconds,
        startup_timeout=settings.shell_startup_timeout_seconds,
        termination_timeout=settings.shell_termination_timeout_seconds,
        max_output_lines=settings.shell_max_output_lines,
        max_output_bytes=settings.shell_max_output_bytes,
    )


def _build_web_search_tool(settings: Settings):
    tavily_search = TavilySearch(
        max_results=settings.tavily_max_results,
        search_depth="advanced",
        topic="general",
        include_answer="basic",
    )

    @tool("web_search")
    def web_search(query: str) -> str:
        """Search the web using Tavily for current documentation, APIs, and technical references."""

        result = tavily_search.invoke({"query": query})
        return _format_tavily_results(result)

    return web_search


def _format_tavily_results(result: dict[str, Any]) -> str:
    sections: list[str] = []

    answer = result.get("answer")
    if isinstance(answer, str) and answer.strip():
        sections.append(f"Answer:\n{answer.strip()}")

    results = result.get("results")
    if isinstance(results, list) and results:
        lines = ["Results:"]
        for index, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "Untitled result").strip()
            url = str(item.get("url") or "").strip()
            content = str(item.get("content") or "").strip()
            snippet = content[:400] + ("..." if len(content) > 400 else "")
            lines.append(f"{index}. {title}")
            if url:
                lines.append(f"   URL: {url}")
            if snippet:
                lines.append(f"   Snippet: {snippet}")
        sections.append("\n".join(lines))

    response_time = result.get("response_time")
    if response_time is not None:
        sections.append(f"Response time: {response_time}s")

    return "\n\n".join(sections) if sections else "No web results found."


def _build_system_prompt(
    *,
    workspace_root: Path,
    shell_runtime: ShellRuntime,
) -> str:
    return "\n".join(
        [
            "You are Zain, a coding command-line agent built with LangChain.",
            f"Workspace root: {workspace_root}",
            f"Shell mode: {shell_runtime.mode}",
            f"Shell enforcement: {shell_runtime.enforcement}",
            "",
            "Workspace scoping rules:",
            "- Treat the provided workspace root as the project root for all questions about files, code, architecture, summaries, or the project/codebase in general.",
            "- Interpret all relative paths from the workspace root unless the user explicitly states a different path inside the workspace.",
            "- Scope all repository inspection, code summaries, and project-level answers to the workspace root only.",
            "- Do not assume files, modules, repositories, or project context outside the workspace root are relevant.",
            "",
            "Operating rules:",
            "- For coding work, call `write_todos` before other non-trivial actions.",
            f"- Every todo update is mirrored into `{PLAN_RELATIVE_PATH}` inside the workspace.",
            f"- After creating the todo list for a coding task, read `{PLAN_RELATIVE_PATH}` and use it as the canonical checklist.",
            "- Complete the checklist one item at a time instead of batching unrelated steps together.",
            "- Updating the todo list does not finish the job; continue executing the request after planning.",
            f"- Before the final answer, call `write_todos` one last time, read `{PLAN_RELATIVE_PATH}` again, verify all items are completed, delete the file, and only then reply.",
            "- Use `glob_search` and `grep_search` for file discovery before broad shell scans.",
            "- Use the `shell` tool for reading, creating, editing, renaming, and deleting files.",
            "- Search tools return workspace-rooted virtual paths like `/src/app.py`; convert these to shell paths like `src/app.py`.",
            "- Do not intentionally access files outside the workspace root.",
            "- Use the Tavily-backed `web_search` tool when you need current documentation or external references.",
            "- Keep responses concise and action-oriented.",
        ]
    )
