from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from langchain.agents.middleware.todo import PlanningState, Todo
from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from .paths import APP_STATE_DIRNAME, PLAN_FILENAME

PLAN_RELATIVE_PATH = f"{APP_STATE_DIRNAME}/{PLAN_FILENAME}"


class PlanFileMiddleware(AgentMiddleware[PlanningState[Any], None, Any]):
    state_schema = PlanningState

    def __init__(self, workspace_root: Path) -> None:
        super().__init__()
        self.workspace_root = workspace_root.expanduser().resolve()
        self.state_dir = self.workspace_root / APP_STATE_DIRNAME
        self.plan_path = self.state_dir / PLAN_FILENAME

    def before_agent(
        self, state: PlanningState[Any], runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        del runtime

        todos = _normalize_todos(state.get("todos"))
        self.state_dir.mkdir(parents=True, exist_ok=True)

        if not todos or _all_todos_completed(todos):
            self._delete_plan_file()
            return None

        self._write_plan_file(todos)
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command[Any]:
        result = handler(request)

        if request.tool_call.get("name") != "write_todos":
            return result

        todos = _extract_todos(request.tool_call.get("args"))
        if not todos:
            self._delete_plan_file()
            return result

        self._write_plan_file(todos)
        reminder = (
            f"The current todo list has been written to `{PLAN_RELATIVE_PATH}`. "
            f"Read `{PLAN_RELATIVE_PATH}` before continuing, execute the plan one item "
            "at a time, keep it in sync with `write_todos`, and before the final answer "
            f"read `{PLAN_RELATIVE_PATH}` again, confirm every item is completed, delete "
            "the file, and only then respond to the user."
        )
        return _append_command_message(result, request.tool_call["id"], reminder)

    @hook_config(can_jump_to=["model"])
    def after_model(
        self, state: PlanningState[Any], runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        del runtime

        last_ai_message = next(
            (message for message in reversed(state["messages"]) if isinstance(message, AIMessage)),
            None,
        )
        if last_ai_message is None or last_ai_message.tool_calls:
            return None

        todos = _normalize_todos(state.get("todos"))
        if not todos:
            self._delete_plan_file()
            return None

        if not _all_todos_completed(todos):
            self._write_plan_file(todos)
            return {
                "messages": [
                    HumanMessage(
                        content=(
                            "Workflow reminder: do not answer the user yet. Read "
                            f"`{PLAN_RELATIVE_PATH}` and continue the coding task by "
                            "completing the remaining items one by one. Update the plan "
                            "with `write_todos` as you make progress."
                        )
                    )
                ],
                "jump_to": "model",
            }

        if self.plan_path.exists():
            return {
                "messages": [
                    HumanMessage(
                        content=(
                            "Workflow reminder: before your final answer, use the shell "
                            f"tool to read `{PLAN_RELATIVE_PATH}` one more time, verify "
                            "that every item is completed, delete the file, and only then "
                            "respond to the user."
                        )
                    )
                ],
                "jump_to": "model",
            }

        return None

    def _write_plan_file(self, todos: list[Todo]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.plan_path.write_text(_render_plan_markdown(self.workspace_root, todos))

    def _delete_plan_file(self) -> None:
        try:
            self.plan_path.unlink()
        except FileNotFoundError:
            pass


def _append_command_message(
    result: ToolMessage | Command[Any], tool_call_id: str, content: str
) -> ToolMessage | Command[Any]:
    if not isinstance(result, Command):
        return result

    update = result.update if isinstance(result.update, dict) else {}
    merged_update = dict(update)
    messages = list(merged_update.get("messages", []))
    messages.append(ToolMessage(content=content, tool_call_id=tool_call_id, name="write_todos"))
    merged_update["messages"] = messages

    return Command(
        graph=result.graph,
        update=merged_update,
        resume=result.resume,
        goto=result.goto,
    )


def _extract_todos(args: object) -> list[Todo]:
    if not isinstance(args, dict):
        return []
    return _normalize_todos(args.get("todos"))


def _normalize_todos(raw_todos: object) -> list[Todo]:
    if not isinstance(raw_todos, list):
        return []

    todos: list[Todo] = []
    for item in raw_todos:
        if not isinstance(item, dict):
            continue

        content = str(item.get("content", "")).strip()
        status = item.get("status")
        if not content or status not in {"pending", "in_progress", "completed"}:
            continue

        todos.append(
            {
                "content": content,
                "status": cast("Todo['status']", status),
            }
        )

    return todos


def _all_todos_completed(todos: list[Todo]) -> bool:
    return all(todo["status"] == "completed" for todo in todos)


def _render_plan_markdown(workspace_root: Path, todos: list[Todo]) -> str:
    status_prefix = {
        "pending": "[ ]",
        "in_progress": "[-]",
        "completed": "[x]",
    }

    lines = [
        "# Current Plan",
        "",
        "This file is managed automatically from the agent todo list.",
        "Read it before executing the next step and keep it in sync with `write_todos`.",
        "",
        f"Workspace root: `{workspace_root}`",
        "",
        "## Todos",
        "",
    ]

    for index, todo in enumerate(todos, start=1):
        marker = status_prefix[todo["status"]]
        lines.append(f"{index}. {marker} {todo['content']}")

    lines.extend(
        [
            "",
            "Before the final answer, read this file again, make sure every item is `[x]`,",
            "delete this file, and only then reply to the user.",
            "",
        ]
    )
    return "\n".join(lines)
