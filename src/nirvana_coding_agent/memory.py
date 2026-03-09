from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
import shutil

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.sqlite import SqliteSaver

from .paths import APP_STATE_DIRNAME, LEGACY_APP_STATE_DIRNAME, MEMORY_DB_FILENAME

PREGEL_TASKS_CHANNEL = "__pregel_tasks"


@dataclass(frozen=True)
class ConversationRecord:
    name: str
    created_at: str
    updated_at: str


class ConversationSqliteSaver(SqliteSaver):
    def put_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, object]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        filtered_writes = [
            (channel, value) for channel, value in writes if channel != PREGEL_TASKS_CHANNEL
        ]
        if not filtered_writes:
            return
        super().put_writes(config, filtered_writes, task_id, task_path)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        sanitized = dict(checkpoint)

        channel_values = dict(sanitized.get("channel_values", {}))
        channel_values.pop(PREGEL_TASKS_CHANNEL, None)
        sanitized["channel_values"] = channel_values

        channel_versions = dict(sanitized.get("channel_versions", {}))
        channel_versions.pop(PREGEL_TASKS_CHANNEL, None)
        sanitized["channel_versions"] = channel_versions

        versions_seen = sanitized.get("versions_seen", {})
        sanitized["versions_seen"] = {
            node: {key: value for key, value in versions.items() if key != PREGEL_TASKS_CHANNEL}
            for node, versions in versions_seen.items()
        }

        updated_channels = sanitized.get("updated_channels")
        if isinstance(updated_channels, list):
            sanitized["updated_channels"] = [
                channel for channel in updated_channels if channel != PREGEL_TASKS_CHANNEL
            ]

        # These are transient scheduler internals, not durable conversation memory.
        sanitized["pending_sends"] = []

        return super().put(config, sanitized, metadata, new_versions)


class ConversationMemory:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.expanduser().resolve()
        self.memory_dir = self.workspace_root / APP_STATE_DIRNAME
        self.legacy_memory_dir = self.workspace_root / LEGACY_APP_STATE_DIRNAME
        self._migrate_legacy_memory_dir()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.memory_dir / MEMORY_DB_FILENAME
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.checkpointer = ConversationSqliteSaver(self.connection)
        self.checkpointer.setup()
        self._setup_conversations_table()

    def close(self) -> None:
        self.connection.close()

    def list_conversations(self) -> list[ConversationRecord]:
        rows = self.connection.execute(
            """
            SELECT name, created_at, updated_at
            FROM conversations
            ORDER BY lower(name) ASC
            """
        ).fetchall()
        return [
            ConversationRecord(name=row[0], created_at=row[1], updated_at=row[2]) for row in rows
        ]

    def conversation_exists(self, name: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM conversations WHERE name = ? LIMIT 1",
            (name,),
        ).fetchone()
        return row is not None

    def ensure_conversation(self, name: str) -> bool:
        existed = self.conversation_exists(name)
        if existed:
            self.touch_conversation(name)
            return True

        self.connection.execute(
            """
            INSERT INTO conversations (name)
            VALUES (?)
            """,
            (name,),
        )
        self.connection.commit()
        return False

    def touch_conversation(self, name: str) -> None:
        self.connection.execute(
            """
            UPDATE conversations
            SET updated_at = CURRENT_TIMESTAMP
            WHERE name = ?
            """,
            (name,),
        )
        self.connection.commit()

    def _setup_conversations_table(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                name TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.commit()

    def _migrate_legacy_memory_dir(self) -> None:
        if self.memory_dir.exists() or not self.legacy_memory_dir.exists():
            return

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        for source in self.legacy_memory_dir.glob(f"{MEMORY_DB_FILENAME}*"):
            if source.is_file():
                shutil.copy2(source, self.memory_dir / source.name)
