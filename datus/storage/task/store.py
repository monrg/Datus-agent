# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Task storage implementation using a relational backend (default: SQLite).
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from datus.storage.backends.relational import ColumnSpec, TableSchema
from datus.storage.db_manager import DBManager
from datus.storage.lancedb_conditions import and_, eq, lt, ne
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


TASK_SCHEMA = TableSchema(
    name="tasks",
    columns=(
        ColumnSpec("task_id", "TEXT", nullable=False, primary_key=True),
        ColumnSpec("task_query", "TEXT", nullable=False),
        ColumnSpec("sql_query", "TEXT", nullable=True, default=""),
        ColumnSpec("sql_result", "TEXT", nullable=True, default=""),
        ColumnSpec("status", "TEXT", nullable=True, default=""),
        ColumnSpec("user_feedback", "TEXT", nullable=True, default=""),
        ColumnSpec("created_at", "TEXT", nullable=False),
        ColumnSpec("updated_at", "TEXT", nullable=False),
    ),
)


class TaskStore:
    """Relational storage for task and feedback data (default: SQLite)."""

    def __init__(
        self,
        db_path: str,
        backend_type: str = "sqlalchemy",
        connection_string: Optional[str] = None,
        ddl_mode: str = "auto",
        **backend_config: Any,
    ):
        """Initialize the task store.

        Args:
            db_path: Path to the directory where the SQLite database will be stored
            backend_type: Backend type name (default: sqlalchemy)
            connection_string: Optional SQLAlchemy connection string override
            ddl_mode: DDL handling mode ('auto', 'disabled', 'required')
            **backend_config: Backend-specific config
        """
        self.db_path = db_path
        self.db_manager = DBManager.get_instance(
            db_path,
            db_name="task.db",
            backend_type=backend_type,
            connection_string=connection_string,
            ddl_mode=ddl_mode,
            **backend_config,
        )
        self.table = self.db_manager.ensure_table(TASK_SCHEMA)
        self.db_location = f"{self.db_manager.db_path}/task.db"
        self._ensure_table()

    def _ensure_table(self):
        """Ensure the tasks table exists in the database."""
        logger.debug(f"Tasks table ensured in {self.db_location}")

    def record_feedback(self, task_id: str, status: str) -> Dict[str, Any]:
        """Record user feedback for a task.

        Args:
            task_id: The task ID to record feedback for
            status: The feedback status ("success" or "failed")

        Returns:
            Dictionary containing the updated task data

        Raises:
            DatusException: If the feedback recording fails
        """
        try:
            updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            updated = self.table.update(
                where=eq("task_id", task_id),
                values={"user_feedback": status, "updated_at": updated_at},
            )

            if updated == 0:
                raise DatusException(ErrorCode.TOOL_STORE_FAILED, message=f"Task {task_id} not found")

            row = self.table.select_one(where=eq("task_id", task_id))
            if row:
                logger.info(f"Recorded feedback for task {task_id}: {status}")
                return {
                    "task_id": row.get("task_id"),
                    "task_query": row.get("task_query"),
                    "sql_query": row.get("sql_query"),
                    "sql_result": row.get("sql_result"),
                    "status": row.get("status"),
                    "user_feedback": row.get("user_feedback"),
                    "created_at": row.get("created_at"),
                    "recorded_at": row.get("updated_at"),  # Use updated_at as recorded_at for compatibility
                }
            raise DatusException(ErrorCode.TOOL_STORE_FAILED, message=f"Failed to retrieve updated task {task_id}")

        except Exception as e:
            raise DatusException(
                ErrorCode.TOOL_STORE_FAILED, message=f"Failed to record feedback for task {task_id}: {str(e)}"
            ) from e

    def get_feedback(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get feedback for a specific task.

        Args:
            task_id: The task ID to get feedback for

        Returns:
            Dictionary containing the task data with feedback, or None if not found
        """
        try:
            row = self.table.select_one(where=and_(eq("task_id", task_id), ne("user_feedback", "")))
            if row:
                return {
                    "task_id": row.get("task_id"),
                    "task_query": row.get("task_query"),
                    "sql_query": row.get("sql_query"),
                    "sql_result": row.get("sql_result"),
                    "status": row.get("status"),
                    "user_feedback": row.get("user_feedback"),
                    "created_at": row.get("created_at"),
                    "recorded_at": row.get("updated_at"),
                }
            return None

        except Exception as e:
            logger.error(f"Failed to get feedback for task {task_id}: {str(e)}")
            return None

    def get_all_feedback(self) -> list[Dict[str, Any]]:
        """Get all recorded feedback.

        Returns:
            List of dictionaries containing all tasks with feedback
        """
        try:
            rows = self.table.select(
                where=ne("user_feedback", ""),
                order_by=[("updated_at", "desc")],
            )
            return [
                {
                    "task_id": row.get("task_id"),
                    "task_query": row.get("task_query"),
                    "sql_query": row.get("sql_query"),
                    "sql_result": row.get("sql_result"),
                    "status": row.get("status"),
                    "user_feedback": row.get("user_feedback"),
                    "created_at": row.get("created_at"),
                    "recorded_at": row.get("updated_at"),
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to get all feedback: {str(e)}")
            return []

    def delete_feedback(self, task_id: str) -> bool:
        """Clear feedback for a specific task.

        Args:
            task_id: The task ID to clear feedback for

        Returns:
            True if feedback was cleared, False if not found
        """
        try:
            updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            updated = self.table.update(
                where=and_(eq("task_id", task_id), ne("user_feedback", "")),
                values={"user_feedback": "", "updated_at": updated_at},
            )
            if updated > 0:
                logger.info(f"Cleared feedback for task {task_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to clear feedback for task {task_id}: {str(e)}")
            return False

    # Task management methods
    def create_task(self, task_id: str, task_query: str) -> Dict[str, Any]:
        """Create a new task record.

        Args:
            task_id: The task ID
            task_query: The original user task/query

        Returns:
            Dictionary containing the created task data

        Raises:
            DatusException: If the task creation fails
        """
        try:
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            row = {
                "task_id": task_id,
                "task_query": task_query,
                "sql_query": "",
                "sql_result": "",
                "status": "running",
                "user_feedback": "",
                "created_at": now,
                "updated_at": now,
            }
            self.table.upsert(row, conflict_columns=("task_id",))
            logger.debug(f"Created task record for {task_id}")
            return row

        except Exception as e:
            raise DatusException(
                ErrorCode.TOOL_STORE_FAILED, message=f"Failed to create task {task_id}: {str(e)}"
            ) from e

    def update_task(self, task_id: str, sql_query: str = None, sql_result: str = None, status: str = None) -> bool:
        """Update task information.

        Args:
            task_id: The task ID to update
            sql_query: The generated SQL query (optional)
            sql_result: The SQL execution result (optional)
            status: The task status (optional)

        Returns:
            True if task was updated, False if not found
        """
        try:
            updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            values: Dict[str, Any] = {}
            if sql_query is not None:
                values["sql_query"] = sql_query
            if sql_result is not None:
                values["sql_result"] = sql_result
            if status is not None:
                values["status"] = status
            if not values:
                return False
            values["updated_at"] = updated_at
            updated = self.table.update(where=eq("task_id", task_id), values=values)
            if updated > 0:
                logger.debug(f"Updated task {task_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to update task {task_id}: {str(e)}")
            return False

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task information.

        Args:
            task_id: The task ID to get

        Returns:
            Dictionary containing the task data, or None if not found
        """
        try:
            row = self.table.select_one(where=eq("task_id", task_id))
            if row:
                return {
                    "task_id": row.get("task_id"),
                    "task_query": row.get("task_query"),
                    "sql_query": row.get("sql_query"),
                    "sql_result": row.get("sql_result"),
                    "status": row.get("status"),
                    "user_feedback": row.get("user_feedback"),
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at"),
                }
            return None

        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {str(e)}")
            return None

    def delete_task(self, task_id: str) -> bool:
        """Delete a task record.

        Args:
            task_id: The task ID to delete

        Returns:
            True if task was deleted, False if not found
        """
        try:
            deleted = self.table.delete(where=eq("task_id", task_id))
            if deleted > 0:
                logger.debug(f"Deleted task {task_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {str(e)}")
            return False

    def cleanup_old_tasks(self, hours: int = 24) -> int:
        """Clean up old task records.

        Args:
            hours: Delete tasks older than this many hours

        Returns:
            Number of tasks deleted
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            cutoff_str = cutoff_time.isoformat().replace("+00:00", "Z")
            deleted_count = self.table.delete(where=lt("created_at", cutoff_str))
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old tasks")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old tasks: {str(e)}")
            return 0
