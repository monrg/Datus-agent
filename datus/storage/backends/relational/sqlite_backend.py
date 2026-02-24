# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""SQLite backend implementation for relational storage."""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from datus.storage.backends.relational.filter_compiler import SQLFilterCompiler
from datus.storage.backends.relational.interfaces import ColumnSpec, RelationalCapabilities, TableSchema
from datus.storage.backends.vector.interfaces import FilterExpr
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SQLiteTransaction:
    """Transaction context manager for SQLite.

    Provides explicit commit/rollback control with automatic cleanup.
    """

    def __init__(self, connection: sqlite3.Connection):
        self._conn = connection
        self._committed = False
        self._rolled_back = False

    def __enter__(self) -> "SQLiteTransaction":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self.rollback()
        elif not self._committed and not self._rolled_back:
            self.commit()

    def commit(self) -> None:
        """Commit the transaction."""
        if not self._committed and not self._rolled_back:
            self._conn.commit()
            self._committed = True

    def rollback(self) -> None:
        """Rollback the transaction."""
        if not self._committed and not self._rolled_back:
            self._conn.rollback()
            self._rolled_back = True


class SQLiteTable:
    """SQLite table operations implementation.

    Provides all CRUD operations with parameterized queries for safety.
    """

    def __init__(
        self,
        backend: "SQLiteBackend",
        schema: TableSchema,
        compiler: SQLFilterCompiler,
    ):
        self._backend = backend
        self._schema = schema
        self._compiler = compiler
        self._name = schema.name

    @property
    def name(self) -> str:
        """Table name."""
        return self._name

    def insert(self, row: Mapping[str, Any]) -> int:
        """Insert a single row.

        Args:
            row: Column-value mapping to insert

        Returns:
            The last inserted row ID
        """
        columns = list(row.keys())
        values = [row[c] for c in columns]
        placeholders = ", ".join(["?"] * len(columns))
        column_list = ", ".join(columns)

        sql = f"INSERT INTO {self._name} ({column_list}) VALUES ({placeholders})"

        with self._backend._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, values)
            conn.commit()
            return cursor.lastrowid

    def insert_many(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """Insert multiple rows.

        Args:
            rows: Sequence of column-value mappings

        Returns:
            Number of rows inserted
        """
        if not rows:
            return 0

        columns = list(rows[0].keys())
        placeholders = ", ".join(["?"] * len(columns))
        column_list = ", ".join(columns)

        sql = f"INSERT INTO {self._name} ({column_list}) VALUES ({placeholders})"

        with self._backend._get_connection() as conn:
            cursor = conn.cursor()
            for row in rows:
                values = [row.get(c) for c in columns]
                cursor.execute(sql, values)
            conn.commit()
            return len(rows)

    def upsert(
        self,
        row: Mapping[str, Any],
        conflict_columns: Sequence[str],
    ) -> int:
        """Insert or update on conflict.

        Uses INSERT OR REPLACE for SQLite.

        Args:
            row: Column-value mapping
            conflict_columns: Columns that define uniqueness (used for conflict detection)

        Returns:
            The last row ID affected
        """
        columns = list(row.keys())
        values = [row[c] for c in columns]
        placeholders = ", ".join(["?"] * len(columns))
        column_list = ", ".join(columns)

        sql = f"INSERT OR REPLACE INTO {self._name} ({column_list}) VALUES ({placeholders})"

        with self._backend._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, values)
            conn.commit()
            return cursor.lastrowid

    def update(
        self,
        where: Optional[FilterExpr],
        values: Mapping[str, Any],
    ) -> int:
        """Update rows matching the filter.

        Args:
            where: Filter expression
            values: Column-value mapping of updates

        Returns:
            Number of rows updated
        """
        if not values:
            return 0

        set_parts = []
        params = []
        for col, val in values.items():
            set_parts.append(f"{col} = ?")
            params.append(val)

        set_clause = ", ".join(set_parts)
        sql = f"UPDATE {self._name} SET {set_clause}"

        where_clause, where_params = self._compiler.compile(where)
        if where_clause:
            sql += f" WHERE {where_clause}"
            params.extend(where_params)

        with self._backend._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            return cursor.rowcount

    def delete(self, where: Optional[FilterExpr]) -> int:
        """Delete rows matching the filter.

        Args:
            where: Filter expression

        Returns:
            Number of rows deleted
        """
        sql = f"DELETE FROM {self._name}"

        where_clause, params = self._compiler.compile(where)
        if where_clause:
            sql += f" WHERE {where_clause}"

        with self._backend._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            return cursor.rowcount

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
            columns: Columns to select (None for *)
            where: Filter expression
            order_by: Sequence of (column, direction) tuples
            limit: Maximum rows to return
            offset: Rows to skip

        Returns:
            List of row dictionaries
        """
        col_list = ", ".join(columns) if columns else "*"
        sql = f"SELECT {col_list} FROM {self._name}"

        params: List[Any] = []
        where_clause, where_params = self._compiler.compile(where)
        if where_clause:
            sql += f" WHERE {where_clause}"
            params.extend(where_params)

        if order_by:
            order_parts = [f"{col} {direction.upper()}" for col, direction in order_by]
            sql += f" ORDER BY {', '.join(order_parts)}"

        if limit is not None:
            sql += f" LIMIT {limit}"

        if offset is not None:
            sql += f" OFFSET {offset}"

        with self._backend._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def select_one(
        self,
        columns: Optional[Sequence[str]] = None,
        where: Optional[FilterExpr] = None,
    ) -> Optional[Dict[str, Any]]:
        """Select a single row.

        Args:
            columns: Columns to select
            where: Filter expression

        Returns:
            Row dictionary or None
        """
        results = self.select(columns=columns, where=where, limit=1)
        return results[0] if results else None

    def count(self, where: Optional[FilterExpr] = None) -> int:
        """Count rows matching the filter.

        Args:
            where: Filter expression

        Returns:
            Row count
        """
        sql = f"SELECT COUNT(*) FROM {self._name}"

        params: List[Any] = []
        where_clause, where_params = self._compiler.compile(where)
        if where_clause:
            sql += f" WHERE {where_clause}"
            params.extend(where_params)

        with self._backend._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            result = cursor.fetchone()
            return result[0] if result else 0

    def exists(self, where: FilterExpr) -> bool:
        """Check if any rows match the filter.

        Args:
            where: Filter expression

        Returns:
            True if at least one row matches
        """
        return self.count(where) > 0


class SQLiteBackend:
    """SQLite backend implementation.

    Provides connection management, table creation, and table handles.
    Uses WAL mode for better concurrent access.
    """

    def __init__(self, db_path: str, db_name: str = "database.db"):
        """Initialize SQLite backend.

        Args:
            db_path: Directory path for database storage
            db_name: Database filename
        """
        self._db_path = db_path
        self._db_name = db_name
        self._db_file = os.path.join(db_path, db_name)
        self._compiler = SQLFilterCompiler()
        self._tables: Dict[str, SQLiteTable] = {}
        self._lock = threading.Lock()

        # Ensure directory exists
        os.makedirs(db_path, exist_ok=True)

        # Initialize WAL mode
        self._init_wal_mode()

        logger.debug(f"SQLiteBackend initialized at {self._db_file}")

    @property
    def name(self) -> str:
        """Backend name."""
        return "sqlite"

    @property
    def caps(self) -> RelationalCapabilities:
        """Backend capabilities."""
        return RelationalCapabilities(
            upsert=True,
            returning=False,  # SQLite 3.35+ supports RETURNING but not all versions
            json_type=False,
            wal_mode=True,
        )

    def _init_wal_mode(self) -> None:
        """Initialize WAL mode for better concurrency."""
        try:
            with self._get_connection() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                logger.debug("SQLite WAL mode enabled")
        except Exception as e:
            logger.warning(f"Failed to enable WAL mode: {e}")

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection.

        Yields:
            SQLite connection
        """
        conn = None
        try:
            conn = sqlite3.connect(self._db_file, check_same_thread=False)
            yield conn
        finally:
            if conn:
                conn.close()

    def ensure_table(self, schema: TableSchema) -> SQLiteTable:
        """Create table if not exists and return a table handle.

        Args:
            schema: Table schema specification

        Returns:
            SQLiteTable handle
        """
        with self._lock:
            if schema.name in self._tables:
                return self._tables[schema.name]

            self._create_table(schema)
            table = SQLiteTable(self, schema, self._compiler)
            self._tables[schema.name] = table
            return table

    def _create_table(self, schema: TableSchema) -> None:
        """Create table and indexes from schema.

        Args:
            schema: Table schema specification
        """
        # Build column definitions
        col_defs = []
        for col in schema.columns:
            col_def = self._build_column_def(col)
            col_defs.append(col_def)

        # Add unique constraints
        for unique_cols in schema.unique_constraints:
            constraint = f"UNIQUE({', '.join(unique_cols)})"
            col_defs.append(constraint)

        columns_sql = ", ".join(col_defs)
        create_sql = f"CREATE TABLE IF NOT EXISTS {schema.name} ({columns_sql})"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_sql)

            # Create indexes
            for idx in schema.indexes:
                unique = "UNIQUE " if idx.unique else ""
                cols = ", ".join(idx.columns)
                idx_sql = f"CREATE {unique}INDEX IF NOT EXISTS {idx.name} ON {schema.name}({cols})"
                cursor.execute(idx_sql)

            conn.commit()
            logger.debug(f"Table {schema.name} created/verified")

    def _build_column_def(self, col: ColumnSpec) -> str:
        """Build SQL column definition.

        Args:
            col: Column specification

        Returns:
            SQL column definition string
        """
        parts = [col.name, col.data_type]

        if col.primary_key:
            parts.append("PRIMARY KEY")
            if col.autoincrement:
                parts.append("AUTOINCREMENT")

        if not col.nullable and not col.primary_key:
            parts.append("NOT NULL")

        if col.unique and not col.primary_key:
            parts.append("UNIQUE")

        if col.default is not None:
            if isinstance(col.default, str):
                parts.append(f"DEFAULT '{col.default}'")
            else:
                parts.append(f"DEFAULT {col.default}")

        return " ".join(parts)

    def drop_table(self, name: str) -> None:
        """Drop a table.

        Args:
            name: Table name to drop
        """
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(f"DROP TABLE IF EXISTS {name}")
                conn.commit()

            if name in self._tables:
                del self._tables[name]

            logger.debug(f"Table {name} dropped")

    def table_exists(self, name: str) -> bool:
        """Check if a table exists.

        Args:
            name: Table name

        Returns:
            True if table exists
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (name,),
            )
            return cursor.fetchone() is not None

    def get_table(self, name: str) -> Optional[SQLiteTable]:
        """Get a table handle by name.

        Args:
            name: Table name

        Returns:
            SQLiteTable or None if not found
        """
        with self._lock:
            if name in self._tables:
                return self._tables[name]

            if not self.table_exists(name):
                return None

            # Table exists but we don't have a handle - this shouldn't happen
            # in normal usage since ensure_table creates both
            logger.warning(f"Table {name} exists but no schema available")
            return None

    def transaction(self) -> SQLiteTransaction:
        """Create a transaction context.

        Returns:
            SQLiteTransaction context manager
        """
        conn = sqlite3.connect(self._db_file, check_same_thread=False)
        return SQLiteTransaction(conn)

    def close(self) -> None:
        """Close the backend and release resources."""
        with self._lock:
            self._tables.clear()
        logger.debug(f"SQLiteBackend closed: {self._db_file}")
