from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .agent import ShellRuntime, build_agent
from .config import Settings
from .memory import ConversationMemory, ConversationRecord

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Zain: interactive coding agent CLI built with LangChain and Azure OpenAI.",
    pretty_exceptions_enable=False,
)
console = Console()


@dataclass
class AgentSession:
    agent: Any
    memory: ConversationMemory
    conversation_name: str | None = None


@app.command()
def main(
    workspace: Path = typer.Argument(
        ...,
        resolve_path=True,
        help="Directory the agent should use as its workspace root.",
    ),
) -> None:
    project_dotenv = Path(__file__).resolve().parents[2] / ".env"
    if project_dotenv.exists():
        load_dotenv(project_dotenv, override=False)

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)

    settings = Settings.from_env()
    if not settings.azure_openai_api_key:
        console.print(
            "[bold red]AZURE_OPENAI_API_KEY is not configured.[/bold red] "
            "Update the local .env file or export it in your shell."
        )
        raise typer.Exit(code=1)
    if not settings.azure_openai_endpoint:
        console.print(
            "[bold red]AZURE_OPENAI_ENDPOINT is not configured.[/bold red] "
            "Update the local .env file or export it in your shell."
        )
        raise typer.Exit(code=1)
    if not settings.azure_openai_api_version:
        console.print(
            "[bold red]AZURE_OPENAI_API_VERSION is not configured.[/bold red] "
            "Update the local .env file or export it in your shell."
        )
        raise typer.Exit(code=1)
    if not settings.azure_openai_deployment:
        console.print(
            "[bold red]AZURE_OPENAI_DEPLOYMENT is not configured.[/bold red] "
            "Update the local .env file or export it in your shell."
        )
        raise typer.Exit(code=1)
    if not settings.tavily_api_key:
        console.print(
            "[bold red]TAVILY_API_KEY is not configured.[/bold red] "
            "Update the local .env file or export it in your shell."
        )
        raise typer.Exit(code=1)

    workspace = workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    memory = ConversationMemory(workspace)

    try:
        agent, shell_runtime = build_agent(
            settings,
            workspace,
            checkpointer=memory.checkpointer,
        )
    except Exception as exc:
        memory.close()
        console.print(f"[bold red]Failed to initialize the agent:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    session = AgentSession(agent=agent, memory=memory)
    _print_banner(workspace, shell_runtime, settings.model)

    try:
        while True:
            try:
                user_input = console.input("[bold cyan]you> [/]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break

            if not user_input:
                continue

            if _handle_command(session, user_input):
                if user_input.strip().lower() == "/exit":
                    break
                continue

            if session.conversation_name is None:
                console.print(
                    "[yellow]Start a conversation first with [/yellow]"
                    "[bold]/start <conversation-name>[/bold][yellow].[/yellow]"
                )
                continue

            _run_turn(session, user_input)
    finally:
        memory.close()


def _run_turn(session: AgentSession, user_input: str) -> None:
    config = _conversation_config(session.conversation_name)

    try:
        with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
            result = session.agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
    except Exception as exc:
        console.print(f"[bold red]Agent execution failed:[/bold red] {exc}")
        return

    if session.conversation_name is not None:
        session.memory.touch_conversation(session.conversation_name)

    message = _last_ai_message(result)
    if message is None:
        console.print("[yellow]The agent did not return a final assistant message.[/yellow]")
        return

    text, citations = _extract_text_and_citations(message)
    if text:
        _print_assistant_response(text)
    else:
        console.print(Panel("<no text response>", title="assistant", border_style="yellow"))

    if citations:
        table = Table(title="Citations", show_header=True, header_style="bold blue")
        table.add_column("Title")
        table.add_column("URL")
        for title, url in citations:
            table.add_row(title, url)
        console.print(table)

    _print_todos(result)


def _handle_command(session: AgentSession, user_input: str) -> bool:
    stripped = user_input.strip()
    normalized = stripped.lower()

    if normalized == "/exit":
        return True

    if normalized == "/help":
        _print_help()
        return True

    if normalized in {"/conversation", "/conversations"}:
        _print_conversations(session.memory.list_conversations())
        return True

    if normalized == "/todos":
        _print_todos(_current_state_values(session))
        return True

    if normalized == "/start":
        console.print("[yellow]Usage: /start <conversation-name>[/yellow]")
        return True

    if normalized.startswith("/start "):
        conversation_name = stripped[7:].strip()
        if not conversation_name:
            console.print("[yellow]Usage: /start <conversation-name>[/yellow]")
            return True
        existed = session.memory.ensure_conversation(conversation_name)
        session.conversation_name = conversation_name
        if existed:
            console.print(
                f"[green]Resumed conversation:[/green] [bold]{conversation_name}[/bold]"
            )
        else:
            console.print(
                f"[green]Started new conversation:[/green] [bold]{conversation_name}[/bold]"
            )
        return True

    return False


def _current_state_values(session: AgentSession) -> dict[str, Any] | None:
    if session.conversation_name is None:
        return None

    try:
        snapshot = session.agent.get_state(_conversation_config(session.conversation_name))
    except Exception:
        return None

    values = getattr(snapshot, "values", None)
    return values if isinstance(values, dict) else None


def _conversation_config(conversation_name: str | None) -> dict[str, Any]:
    if conversation_name is None:
        raise ValueError("Conversation name is required.")
    return {"configurable": {"thread_id": conversation_name}}


def _last_ai_message(result: dict[str, Any]) -> AIMessage | None:
    messages = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _extract_text_and_citations(message: AIMessage) -> tuple[str, list[tuple[str, str]]]:
    parts: list[str] = []
    citations: list[tuple[str, str]] = []
    seen_citations: set[tuple[str, str]] = set()

    def visit(node: Any) -> None:
        if isinstance(node, str):
            if node.strip():
                parts.append(node)
            return

        if isinstance(node, list):
            for item in node:
                visit(item)
            return

        if isinstance(node, dict):
            text = node.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)

            for annotation in node.get("annotations", []):
                if not isinstance(annotation, dict):
                    continue
                if annotation.get("type") != "url_citation":
                    continue
                url = annotation.get("url")
                if not isinstance(url, str):
                    continue
                title = annotation.get("title")
                label = title if isinstance(title, str) and title.strip() else url
                citation = (label, url)
                if citation not in seen_citations:
                    seen_citations.add(citation)
                    citations.append(citation)

            summary = node.get("summary")
            if isinstance(summary, list):
                visit(summary)
            return

        rendered = str(node).strip()
        if rendered:
            parts.append(rendered)

    visit(message.content)
    return "\n\n".join(parts).strip(), citations


def _print_assistant_response(text: str) -> None:
    try:
        console.print(Panel(Markdown(text), title="assistant", border_style="green"))
        return
    except Exception as exc:
        console.print(
            f"[yellow]Markdown rendering failed; showing plain text instead:[/yellow] {exc}"
        )

    try:
        console.print(Panel(text, title="assistant", border_style="green"))
    except Exception:
        console.print(text)


def _print_banner(
    workspace: Path, shell_runtime: ShellRuntime, model_name: str
) -> None:
    lines = [
        f"Workspace: {workspace}",
        f"Model: {model_name}",
        f"Shell mode: {shell_runtime.mode}",
        "Web search: Tavily",
        "Commands: /start <name>, /conversation, /todos, /help, /exit",
    ]
    console.print(Panel("\n".join(lines), title="Zain", border_style="blue"))
    if shell_runtime.warning:
        console.print(f"[yellow]{shell_runtime.warning}[/yellow]")


def _print_help() -> None:
    console.print(
        Panel(
            "\n".join(
                [
                    "/start <name> Start or resume a named conversation",
                    "/conversation  Show saved conversation names",
                    "/help  Show available commands",
                    "/todos Show the current todo list for the active conversation",
                    "/exit  Leave the CLI",
                ]
            ),
            title="Commands",
            border_style="blue",
        )
    )


def _print_todos(result: dict[str, Any] | None) -> None:
    todos = result.get("todos") if result else None
    if not todos:
        return

    table = Table(title="Current Plan", show_header=True, header_style="bold magenta")
    table.add_column("Status", width=12)
    table.add_column("Task")

    style_map = {
        "pending": "yellow",
        "in_progress": "cyan",
        "completed": "green",
    }

    for todo in todos:
        status = todo.get("status", "pending")
        content = todo.get("content", "")
        table.add_row(f"[{style_map.get(status, 'white')}]{status}[/]", content)

    console.print(table)


def _print_conversations(conversations: list[ConversationRecord]) -> None:
    if not conversations:
        console.print("[yellow]No saved conversations.[/yellow]")
        return

    table = Table(title="Saved Conversations", show_header=True, header_style="bold blue")
    table.add_column("Name")
    table.add_column("Updated")

    for conversation in conversations:
        table.add_row(conversation.name, conversation.updated_at)

    console.print(table)
