# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Interfaces and data structures for relational database backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from datus.storage.backends.vector.interfaces import FilterExpr


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a database column.

    Attributes:
        name: Column name
        data_type: SQL data type (TEXT, INTEGER, REAL, BLOB)
        nullable: Whether the column allows NULL values
        primary_key: Whether this column is the primary key
        autoincrement: Whether to auto-increment (only valid with INTEGER primary key)
        unique: Whether values must be unique
        default: Default value for the column
    """

    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    autoincrement: bool = False
    unique: bool = False
    default: Optional[Any] = None


@dataclass(frozen=True)
class IndexSpec:
    """Specification for a database index.

    Attributes:
        name: Index name
        columns: Tuple of column names to index
        unique: Whether the index enforces uniqueness
    """

    name: str
    columns: Tuple[str, ...]
    unique: bool = False


@dataclass(frozen=True)
class TableSchema:
    """Schema definition for a relational table.

    Attributes:
        name: Table name
        columns: Tuple of column specifications
        indexes: Tuple of index specifications
        unique_constraints: Tuple of unique constraint column groups
    """

    name: str
    columns: Tuple[ColumnSpec, ...]
    indexes: Tuple[IndexSpec, ...] = ()
    unique_constraints: Tuple[Tuple[str, ...], ...] = ()

    def get_column(self, name: str) -> Optional[ColumnSpec]:
        """Get column spec by name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def get_primary_key(self) -> Optional[ColumnSpec]:
        """Get the primary key column if any."""
        for col in self.columns:
            if col.primary_key:
                return col
        return None


@dataclass(frozen=True)
class RelationalCapabilities:
    """Capabilities of a relational database backend.

    Attributes:
        upsert: Whether backend supports UPSERT/INSERT OR REPLACE
        returning: Whether INSERT/UPDATE supports RETURNING clause
        json_type: Whether backend has native JSON type support
        wal_mode: Whether backend supports WAL mode
        ddl: Whether backend supports DDL operations (CREATE TABLE/INDEX)
    """

    upsert: bool = True
    returning: bool = False
    json_type: bool = False
    wal_mode: bool = True
    ddl: bool = True


class TransactionContext(Protocol):
    """Protocol for database transaction context manager."""

    def __enter__(self) -> "TransactionContext":
        """Enter the transaction context."""
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the transaction context, committing or rolling back."""
        ...

    def commit(self) -> None:
        """Explicitly commit the transaction."""
        ...

    def rollback(self) -> None:
        """Explicitly rollback the transaction."""
        ...


class RelationalTable(Protocol):
    """Protocol for relational table operations.

    All methods that accept `where` parameter use FilterExpr from
    datus.storage.backends.vector.interfaces for consistency with vector backends.
    """

    @property
    def name(self) -> str:
        """Table name."""
        ...

    def insert(self, row: Mapping[str, Any]) -> int:
        """Insert a single row.

        Args:
            row: Column-value mapping to insert

        Returns:
            The last inserted row ID (typically auto-increment primary key)
        """
        ...

    def insert_many(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """Insert multiple rows.

        Args:
            rows: Sequence of column-value mappings to insert

        Returns:
            Number of rows inserted
        """
        ...

    def upsert(
        self,
        row: Mapping[str, Any],
        conflict_columns: Sequence[str],
    ) -> int:
        """Insert or update on conflict.

        Args:
            row: Column-value mapping to upsert
            conflict_columns: Columns that define uniqueness for conflict detection

        Returns:
            The last row ID affected
        """
        ...

    def update(
        self,
        where: Optional[FilterExpr],
        values: Mapping[str, Any],
    ) -> int:
        """Update rows matching the filter.

        Args:
            where: Filter expression (None updates all rows)
            values: Column-value mapping of values to set

        Returns:
            Number of rows updated
        """
        ...

    def delete(self, where: Optional[FilterExpr]) -> int:
        """Delete rows matching the filter.

        Args:
            where: Filter expression (None deletes all rows)

        Returns:
            Number of rows deleted
        """
        ...

    def select(
        self,
        columns: Optional[Sequence[str]] = None,
        where: Optional[FilterExpr] = None,
        order_by: Optional[Sequence[Tuple[str, str]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Select rows matching the filter.

        Args:
            columns: Columns to select (None for all columns)
            where: Filter expression (None selects all rows)
            order_by: Sequence of (column, direction) tuples for ordering
            limit: Maximum number of rows to return
            offset: Number of rows to skip

        Returns:
            List of row dictionaries
        """
        ...

    def select_one(
        self,
        columns: Optional[Sequence[str]] = None,
        where: Optional[FilterExpr] = None,
    ) -> Optional[Dict[str, Any]]:
        """Select a single row matching the filter.

        Args:
            columns: Columns to select (None for all columns)
            where: Filter expression

        Returns:
            Row dictionary or None if not found
        """
        ...

    def count(self, where: Optional[FilterExpr] = None) -> int:
        """Count rows matching the filter.

        Args:
            where: Filter expression (None counts all rows)

        Returns:
            Number of matching rows
        """
        ...

    def exists(self, where: FilterExpr) -> bool:
        """Check if any rows match the filter.

        Args:
            where: Filter expression

        Returns:
            True if at least one row matches
        """
        ...


class RelationalBackend(Protocol):
    """Protocol for relational database backends.

    Backends are responsible for connection management, table creation,
    and providing table handles for CRUD operations.
    """

    @property
    def name(self) -> str:
        """Backend name (e.g., 'sqlite', 'postgresql')."""
        ...

    @property
    def caps(self) -> RelationalCapabilities:
        """Backend capabilities."""
        ...

    def ensure_table(self, schema: TableSchema) -> RelationalTable:
        """Create table if not exists and return a table handle.

        Args:
            schema: Table schema specification

        Returns:
            RelationalTable handle for the table
        """
        ...

    def drop_table(self, name: str) -> None:
        """Drop a table.

        Args:
            name: Table name to drop
        """
        ...

    def table_exists(self, name: str) -> bool:
        """Check if a table exists.

        Args:
            name: Table name to check

        Returns:
            True if table exists
        """
        ...

    def get_table(self, name: str) -> Optional[RelationalTable]:
        """Get a table handle by name.

        Args:
            name: Table name

        Returns:
            RelationalTable handle or None if table doesn't exist
        """
        ...

    def transaction(self) -> TransactionContext:
        """Create a transaction context.

        Returns:
            TransactionContext for managing the transaction
        """
        ...

    def close(self) -> None:
        """Close the backend and release resources."""
        ...
