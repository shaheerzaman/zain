from __future__ import annotations

import posixpath
from pathlib import Path, PurePosixPath
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

FILESYSTEM_TOOL_PATH_ARGS: dict[str, tuple[str, ...]] = {
    "ls": ("path",),
    "read_file": ("file_path",),
    "write_file": ("file_path",),
    "edit_file": ("file_path",),
    "glob": ("path",),
    "grep": ("path",),
}

HOST_ABSOLUTE_PATH_PREFIXES = (
    "/Applications",
    "/bin",
    "/cores",
    "/dev",
    "/etc",
    "/home",
    "/Library",
    "/mnt",
    "/opt",
    "/private",
    "/proc",
    "/System",
    "/tmp",
    "/Users",
    "/usr",
    "/var",
    "/Volumes",
)


class FilesystemPathMiddleware(AgentMiddleware[Any, None, Any]):
    """Normalize filesystem tool paths to workspace-rooted virtual paths."""

    def __init__(self, workspace_root: Path) -> None:
        super().__init__()
        self.workspace_root = workspace_root.expanduser().resolve()

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command[Any]:
        normalized_request = self._normalize_request(request)
        if isinstance(normalized_request, ToolMessage):
            return normalized_request
        return handler(normalized_request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command[Any]:
        normalized_request = self._normalize_request(request)
        if isinstance(normalized_request, ToolMessage):
            return normalized_request
        return await handler(normalized_request)

    def _normalize_request(
        self, request: ToolCallRequest
    ) -> ToolCallRequest | ToolMessage:
        tool_name = request.tool_call["name"]
        path_args = FILESYSTEM_TOOL_PATH_ARGS.get(tool_name)
        if not path_args:
            return request

        raw_args = request.tool_call.get("args")
        if not isinstance(raw_args, dict):
            return request

        updated_args = dict(raw_args)
        changed = False

        for arg_name in path_args:
            raw_value = updated_args.get(arg_name)
            if raw_value is None or not isinstance(raw_value, str):
                continue

            try:
                normalized_value = self._normalize_path(raw_value)
            except ValueError as exc:
                return ToolMessage(
                    content=str(exc),
                    tool_call_id=request.tool_call["id"],
                    name=tool_name,
                )

            if normalized_value != raw_value:
                updated_args[arg_name] = normalized_value
                changed = True

        if not changed:
            return request

        return request.override(
            tool_call={
                **request.tool_call,
                "args": updated_args,
            }
        )

    def _normalize_path(self, raw_path: str) -> str:
        candidate = raw_path.strip()
        if not candidate:
            return raw_path

        host_path = Path(candidate).expanduser()
        if host_path.is_absolute():
            resolved_host_path = host_path.resolve(strict=False)
            try:
                relative_to_workspace = resolved_host_path.relative_to(self.workspace_root)
            except ValueError:
                if _looks_like_host_absolute_path(candidate):
                    raise ValueError(
                        "Error: filesystem tool paths must stay inside the workspace root. "
                        "Use virtual workspace paths like `/.zain/PLAN.md` or "
                        "`/src/app/main.py`, not host paths like "
                        f"`{candidate}`."
                    ) from None
                return _normalize_virtual_path(candidate)

            return _relative_path_to_virtual(relative_to_workspace)

        return _normalize_virtual_path(candidate)


def _relative_path_to_virtual(path: Path) -> str:
    relative = path.as_posix()
    if relative in {"", "."}:
        return "/"
    return _normalize_virtual_path(relative)


def _normalize_virtual_path(raw_path: str) -> str:
    parts = PurePosixPath(raw_path).parts
    if ".." in parts:
        raise ValueError(
            "Error: filesystem tool paths must not use `..`. "
            "Resolve them from the workspace root and use virtual paths like "
            "`/src/app/main.py`."
        )

    normalized = posixpath.normpath(f"/{raw_path.lstrip('/')}")
    return "/" if normalized == "." else normalized


def _looks_like_host_absolute_path(raw_path: str) -> bool:
    return any(
        raw_path == prefix or raw_path.startswith(f"{prefix}/")
        for prefix in HOST_ABSOLUTE_PATH_PREFIXES
    )
