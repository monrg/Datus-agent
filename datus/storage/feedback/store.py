# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Feedback storage implementation using a relational backend (default: SQLite).
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from datus.storage.backends.relational import ColumnSpec, TableSchema
from datus.storage.db_manager import DBManager
from datus.storage.lancedb_conditions import eq
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


FEEDBACK_SCHEMA = TableSchema(
    name="feedback",
    columns=(
        ColumnSpec("task_id", "TEXT", nullable=False, primary_key=True),
        ColumnSpec("status", "TEXT", nullable=False),
        ColumnSpec("created_at", "TEXT", nullable=False),
    ),
)


class FeedbackStore:
    """Relational storage for user feedback data (default: SQLite)."""

    def __init__(
        self,
        db_path: str,
        backend_type: str = "sqlalchemy",
        connection_string: Optional[str] = None,
        ddl_mode: str = "auto",
        **backend_config: Any,
    ):
        """Initialize the feedback store.

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
            db_name="feedback.db",
            backend_type=backend_type,
            connection_string=connection_string,
            ddl_mode=ddl_mode,
            **backend_config,
        )
        self.table = self.db_manager.ensure_table(FEEDBACK_SCHEMA)
        self.db_location = f"{self.db_manager.db_path}/feedback.db"
        self._ensure_table()

    def _ensure_table(self):
        """Ensure the feedback table exists in the database."""
        logger.debug(f"Feedback table ensured in {self.db_location}")

    def record_feedback(self, task_id: str, status: str) -> Dict[str, Any]:
        """Record user feedback for a task.

        Args:
            task_id: The task ID to record feedback for
            status: The feedback status ("success" or "failed")

        Returns:
            Dictionary containing the recorded feedback data

        Raises:
            DatusException: If the feedback recording fails
        """
        try:
            recorded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            row = {"task_id": task_id, "status": status, "created_at": recorded_at}
            self.table.upsert(row, conflict_columns=("task_id",))
            logger.info(f"Recorded feedback for task {task_id}: {status}")
            return {"task_id": task_id, "status": status, "recorded_at": recorded_at}

        except Exception as e:
            raise DatusException(
                ErrorCode.TOOL_STORE_FAILED, message=f"Failed to record feedback for task {task_id}: {str(e)}"
            ) from e

    def get_feedback(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get feedback for a specific task.

        Args:
            task_id: The task ID to get feedback for

        Returns:
            Dictionary containing the feedback data, or None if not found
        """
        try:
            row = self.table.select_one(where=eq("task_id", task_id))
            if row:
                return {
                    "task_id": row.get("task_id"),
                    "status": row.get("status"),
                    "recorded_at": row.get("created_at"),
                }
            return None

        except Exception as e:
            logger.error(f"Failed to get feedback for task {task_id}: {str(e)}")
            return None

    def get_all_feedback(self) -> list[Dict[str, Any]]:
        """Get all recorded feedback.

        Returns:
            List of dictionaries containing all feedback data
        """
        try:
            rows = self.table.select(order_by=[("created_at", "desc")])
            return [
                {
                    "task_id": row.get("task_id"),
                    "status": row.get("status"),
                    "recorded_at": row.get("created_at"),
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to get all feedback: {str(e)}")
            return []

    def delete_feedback(self, task_id: str) -> bool:
        """Delete feedback for a specific task.

        Args:
            task_id: The task ID to delete feedback for

        Returns:
            True if feedback was deleted, False if not found
        """
        try:
            deleted = self.table.delete(where=eq("task_id", task_id))
            if deleted > 0:
                logger.info(f"Deleted feedback for task {task_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete feedback for task {task_id}: {str(e)}")
            return False
