# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Database manager facade for relational storage.

This module provides a unified interface for database operations,
abstracting away the specific backend implementation (SQLite, PostgreSQL, etc.).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Type

from datus.storage.backends.relational.interfaces import (
    RelationalBackend,
    RelationalCapabilities,
    RelationalTable,
    TableSchema,
    TransactionContext,
)
from datus.storage.backends.relational.sqlite_backend import SQLiteBackend
from datus.storage.backends.relational.sqlalchemy_backend import SQLAlchemyBackend
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# Backend registry
_BACKEND_REGISTRY: Dict[str, Type[RelationalBackend]] = {
    "sqlite": SQLiteBackend,
    "sqlalchemy": SQLAlchemyBackend,
}


def register_backend(name: str, backend_class: Type[RelationalBackend]) -> None:
    """Register a new backend type.

    Args:
        name: Backend name (e.g., 'postgresql')
        backend_class: Backend class implementing RelationalBackend protocol
    """
    _BACKEND_REGISTRY[name] = backend_class
    logger.debug(f"Registered backend: {name}")


class DBManager:
    """Thread-safe database manager facade.

    Provides a unified interface for database operations with support for
    multiple backend types. Uses instance-based management with a registry
    for sharing instances across the application.

    Example:
        >>> db = DBManager.get_instance("/path/to/data", db_name="myapp.db")
        >>> table = db.ensure_table(my_schema)
        >>> table.insert({"name": "Alice", "age": 30})
    """

    # Class-level registry for shared instances
    _instances: Dict[str, "DBManager"] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        db_path: str,
        db_name: str = "database.db",
        backend_type: str = "sqlite",
        **backend_config: Any,
    ):
        """Initialize DBManager.

        Args:
            db_path: Directory path for database storage
            db_name: Database filename
            backend_type: Backend type ('sqlite', 'postgresql', etc.)
            **backend_config: Additional backend-specific configuration
        """
        self._db_path = db_path
        self._db_name = db_name
        self._backend_type = backend_type

        # Get backend class
        backend_class = _BACKEND_REGISTRY.get(backend_type)
        if backend_class is None:
            available = ", ".join(_BACKEND_REGISTRY.keys())
            raise ValueError(
                f"Unknown backend type: {backend_type}. Available: {available}"
            )

        # Initialize backend
        self._backend: RelationalBackend = backend_class(
            db_path=db_path, db_name=db_name, **backend_config
        )

        logger.debug(
            f"DBManager initialized: path={db_path}, db={db_name}, backend={backend_type}"
        )

    @classmethod
    def get_instance(
        cls,
        db_path: str,
        db_name: str = "database.db",
        backend_type: str = "sqlite",
        **backend_config: Any,
    ) -> "DBManager":
        """Get or create a DBManager instance.

        This method provides instance sharing for the same database path/name
        combination, ensuring efficient resource usage.

        Args:
            db_path: Directory path for database storage
            db_name: Database filename
            backend_type: Backend type
            **backend_config: Backend-specific configuration

        Returns:
            DBManager instance
        """
        # Create a unique key for this database
        key = f"{db_path}:{db_name}:{backend_type}"

        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(
                    db_path=db_path,
                    db_name=db_name,
                    backend_type=backend_type,
                    **backend_config,
                )
            return cls._instances[key]

    @classmethod
    def clear_instances(cls) -> None:
        """Clear all cached instances.

        Useful for testing or when reconfiguring the database layer.
        """
        with cls._lock:
            for instance in cls._instances.values():
                instance.close()
            cls._instances.clear()
            logger.debug("All DBManager instances cleared")

    @property
    def db_path(self) -> str:
        """Database directory path."""
        return self._db_path

    @property
    def db_name(self) -> str:
        """Database filename."""
        return self._db_name

    @property
    def backend_type(self) -> str:
        """Backend type name."""
        return self._backend_type

    @property
    def caps(self) -> RelationalCapabilities:
        """Backend capabilities."""
        return self._backend.caps

    def ensure_table(self, schema: TableSchema) -> RelationalTable:
        """Create table if not exists and return a table handle.

        Args:
            schema: Table schema specification

        Returns:
            RelationalTable handle for CRUD operations
        """
        return self._backend.ensure_table(schema)

    def get_table(self, name: str) -> Optional[RelationalTable]:
        """Get a table handle by name.

        Args:
            name: Table name

        Returns:
            RelationalTable handle or None if not found
        """
        return self._backend.get_table(name)

    def table_exists(self, name: str) -> bool:
        """Check if a table exists.

        Args:
            name: Table name

        Returns:
            True if table exists
        """
        return self._backend.table_exists(name)

    def drop_table(self, name: str) -> None:
        """Drop a table.

        Args:
            name: Table name to drop
        """
        self._backend.drop_table(name)

    def transaction(self) -> TransactionContext:
        """Create a transaction context.

        Returns:
            TransactionContext for managing the transaction

        Example:
            >>> with db.transaction() as txn:
            ...     table.insert({"name": "Alice"})
            ...     table.insert({"name": "Bob"})
            ...     # Auto-commits on success, rollbacks on exception
        """
        return self._backend.transaction()

    def close(self) -> None:
        """Close the database manager and release resources."""
        self._backend.close()
        logger.debug(f"DBManager closed: {self._db_path}/{self._db_name}")
