# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""SQLAlchemy backend implementation for relational storage."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sqlalchemy import Column, MetaData, Table
from sqlalchemy import and_ as sa_and
from sqlalchemy import delete as sa_delete
from sqlalchemy import false as sa_false
from sqlalchemy import func, inspect, literal
from sqlalchemy import not_ as sa_not
from sqlalchemy import or_ as sa_or
from sqlalchemy import select, text
from sqlalchemy import true as sa_true
from sqlalchemy import update as sa_update
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import Select
from sqlalchemy.sql.elements import ClauseElement
from sqlalchemy.sql.schema import Index, UniqueConstraint
from sqlalchemy.types import BigInteger, Boolean, Date, DateTime, Float, Integer, LargeBinary, SmallInteger, String, Text

from datus.storage.backends.relational.interfaces import (
    ColumnSpec,
    IndexSpec,
    RelationalBackend,
    RelationalCapabilities,
    RelationalTable,
    TableSchema,
    TransactionContext,
)
from datus.storage.backends.vector.interfaces import FilterExpr
from datus.storage.lancedb_conditions import And, Condition, Not, Op, Or
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

from .sqlalchemy_connector import SQLAlchemyConnector

logger = get_logger(__name__)


@dataclass(frozen=True)
class UnsafeRawSQL:
    """Explicit wrapper for executing raw SQL filters."""

    sql: str


class SQLAlchemyTransaction:
    """Transaction context manager for SQLAlchemy."""

    def __init__(self, engine: Engine, connector: SQLAlchemyConnector):
        self._engine = engine
        self._connector = connector
        self._conn = None
        self._txn = None
        self._committed = False
        self._rolled_back = False

    def __enter__(self) -> "SQLAlchemyTransaction":
        try:
            self._conn = self._engine.connect()
            self._txn = self._conn.begin()
        except Exception as exc:
            raise self._connector.handle_exception(exc, operation="transaction begin") from exc
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if exc_type is not None:
                try:
                    self.rollback()
                except Exception as rollback_exc:
                    logger.warning(f"Failed to rollback transaction: {rollback_exc}")
            elif not self._committed and not self._rolled_back:
                try:
                    self.commit()
                except Exception as commit_exc:
                    logger.warning(f"Failed to commit transaction: {commit_exc}")
        finally:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    logger.warning("Failed to close SQLAlchemy connection")
                finally:
                    self._conn = None
                    self._txn = None

    def commit(self) -> None:
        if self._txn is not None and not self._committed and not self._rolled_back:
            try:
                self._txn.commit()
                self._committed = True
            except Exception as exc:
                raise self._connector.handle_exception(exc, operation="transaction commit") from exc

    def rollback(self) -> None:
        if self._txn is not None and not self._committed and not self._rolled_back:
            try:
                self._txn.rollback()
                self._rolled_back = True
            except Exception as exc:
                raise self._connector.handle_exception(exc, operation="transaction rollback") from exc


class SQLAlchemyTable:
    """SQLAlchemy table operations implementation."""

    def __init__(self, backend: "SQLAlchemyBackend", table: Table):
        self._backend = backend
        self._table = table

    @property
    def name(self) -> str:
        return self._table.name

    def insert(self, row: Mapping[str, Any]) -> int:
        stmt = self._table.insert().values(**row)
        return self._backend._execute_insert(stmt)

    def insert_many(self, rows: Sequence[Mapping[str, Any]]) -> int:
        if not rows:
            return 0
        stmt = self._table.insert()
        return self._backend._execute_many(stmt, rows)

    def upsert(self, row: Mapping[str, Any], conflict_columns: Sequence[str]) -> int:
        if not conflict_columns:
            return self.insert(row)

        stmt = self._backend._build_upsert(self._table, row, conflict_columns)
        if stmt is None:
            return self._backend._fallback_upsert(self, row, conflict_columns)
        return self._backend._execute_insert(stmt)

    def update(self, where: Optional[FilterExpr], values: Mapping[str, Any]) -> int:
        if not values:
            return 0
        stmt = sa_update(self._table).values(**values)
        where_clause = self._backend._compile_filter(where, self._table)
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        return self._backend._execute_write(stmt, operation="update")

    def delete(self, where: Optional[FilterExpr]) -> int:
        stmt = sa_delete(self._table)
        where_clause = self._backend._compile_filter(where, self._table)
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        return self._backend._execute_write(stmt, operation="delete")

    def select(
        self,
        columns: Optional[Sequence[str]] = None,
        where: Optional[FilterExpr] = None,
        order_by: Optional[Sequence[Tuple[str, str]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        stmt = self._build_select(columns, where, order_by, limit, offset)
        return self._backend._execute_select(stmt)

    def select_one(
        self,
        columns: Optional[Sequence[str]] = None,
        where: Optional[FilterExpr] = None,
    ) -> Optional[Dict[str, Any]]:
        stmt = self._build_select(columns, where, limit=1)
        rows = self._backend._execute_select(stmt)
        return rows[0] if rows else None

    def count(self, where: Optional[FilterExpr] = None) -> int:
        stmt = select(func.count()).select_from(self._table)
        where_clause = self._backend._compile_filter(where, self._table)
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        return self._backend._execute_scalar(stmt)

    def exists(self, where: FilterExpr) -> bool:
        where_clause = self._backend._compile_filter(where, self._table)
        if where_clause is None:
            return False
        stmt = select(literal(True)).select_from(self._table).where(where_clause).limit(1)
        return self._backend._execute_exists(stmt)

    def _build_select(
        self,
        columns: Optional[Sequence[str]],
        where: Optional[FilterExpr],
        order_by: Optional[Sequence[Tuple[str, str]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Select:
        if columns:
            col_objs = [self._table.c[col] for col in columns]
            stmt: Select = select(*col_objs)
        else:
            stmt = select(self._table)

        where_clause = self._backend._compile_filter(where, self._table)
        if where_clause is not None:
            stmt = stmt.where(where_clause)

        if order_by:
            ordering = []
            for col, direction in order_by:
                col_obj = self._table.c[col]
                if direction.lower() == "desc":
                    ordering.append(col_obj.desc())
                else:
                    ordering.append(col_obj.asc())
            stmt = stmt.order_by(*ordering)

        if limit is not None:
            stmt = stmt.limit(limit)
        if offset is not None:
            stmt = stmt.offset(offset)

        return stmt


class SQLAlchemyBackend(RelationalBackend):
    """SQLAlchemy backend for relational storage."""

    def __init__(
        self,
        db_path: str,
        db_name: str,
        connection_string: Optional[str] = None,
        ddl_mode: str = "auto",
        engine_options: Optional[Dict[str, Any]] = None,
        **_unused: Any,
    ):
        self._db_path = db_path
        self._db_name = db_name
        self._ddl_mode = ddl_mode
        self._ddl_capable = True
        self._lock = threading.Lock()

        if connection_string is None:
            os.makedirs(db_path, exist_ok=True)
            db_file = os.path.join(db_path, db_name)
            connection_string = f"sqlite:///{db_file}"

        self._connection_string = connection_string
        self._connector = SQLAlchemyConnector(connection_string, engine_options=engine_options)
        self._metadata = MetaData()
        self._tables: Dict[str, SQLAlchemyTable] = {}

        logger.debug(f"SQLAlchemyBackend initialized (dialect={self._connector.dialect_name})")

    @property
    def name(self) -> str:
        return "sqlalchemy"

    @property
    def caps(self) -> RelationalCapabilities:
        return RelationalCapabilities(
            upsert=True,
            returning=self._connector.dialect_name in {"postgresql"},
            json_type=self._connector.dialect_name in {"postgresql", "mysql"},
            wal_mode=self._connector.dialect_name == "sqlite",
            ddl=self._ddl_capable and self._ddl_mode != "disabled",
        )

    def ensure_table(self, schema: TableSchema) -> RelationalTable:
        with self._lock:
            existing = self._tables.get(schema.name)
            if existing is not None:
                return existing

            table_exists = self.table_exists(schema.name)
            table = self._build_table(schema)

            if not table_exists:
                self._create_table_with_capability(table, schema)
            else:
                self._validate_schema(schema)

            table_handle = SQLAlchemyTable(self, table)
            self._tables[schema.name] = table_handle
            return table_handle

    def drop_table(self, name: str) -> None:
        table = self._metadata.tables.get(name)
        if table is None:
            table = Table(name, self._metadata)
        try:
            table.drop(self._connector.engine, checkfirst=True)
        except Exception as exc:
            raise self._connector.handle_exception(exc, operation="drop table") from exc

        with self._lock:
            self._tables.pop(name, None)
            metadata_table = self._metadata.tables.get(name)
            if metadata_table is not None:
                self._metadata.remove(metadata_table)

    def table_exists(self, name: str) -> bool:
        inspector = inspect(self._connector.engine)
        try:
            return inspector.has_table(name)
        except SQLAlchemyError as exc:
            raise self._connector.handle_exception(exc, operation="inspect table") from exc

    def get_table(self, name: str) -> Optional[RelationalTable]:
        existing = self._tables.get(name)
        if existing is not None:
            return existing

        if not self.table_exists(name):
            return None

        table = Table(name, self._metadata, autoload_with=self._connector.engine)
        table_handle = SQLAlchemyTable(self, table)
        self._tables[name] = table_handle
        return table_handle

    def transaction(self) -> TransactionContext:
        return SQLAlchemyTransaction(self._connector.engine, self._connector)

    def close(self) -> None:
        self._connector.dispose()

    def _execute_select(self, stmt: Select) -> List[Dict[str, Any]]:
        try:
            with self._connector.begin() as conn:
                result = conn.execute(stmt)
                return [dict(row) for row in result.mappings().all()]
        except Exception as exc:
            raise self._connector.handle_exception(exc, sql=str(stmt), operation="select") from exc

    def _execute_scalar(self, stmt: Select) -> int:
        try:
            with self._connector.begin() as conn:
                result = conn.execute(stmt).scalar()
                return int(result or 0)
        except Exception as exc:
            raise self._connector.handle_exception(exc, sql=str(stmt), operation="scalar") from exc

    def _execute_exists(self, stmt: Select) -> bool:
        try:
            with self._connector.begin() as conn:
                result = conn.execute(stmt).first()
                return result is not None
        except Exception as exc:
            raise self._connector.handle_exception(exc, sql=str(stmt), operation="exists") from exc

    def _execute_write(self, stmt: ClauseElement, operation: str) -> int:
        try:
            with self._connector.begin() as conn:
                result = conn.execute(stmt)
                return int(result.rowcount or 0)
        except Exception as exc:
            raise self._connector.handle_exception(exc, sql=str(stmt), operation=operation) from exc

    def _execute_insert(self, stmt: ClauseElement) -> int:
        try:
            with self._connector.begin() as conn:
                result = conn.execute(stmt)
                inserted = getattr(result, "inserted_primary_key", None)
                if inserted:
                    insert_id = self._coerce_insert_id(inserted[0])
                    if insert_id:
                        return insert_id
                    # Handle composite primary keys (e.g. namespace + id)
                    for value in inserted[1:]:
                        insert_id = self._coerce_insert_id(value)
                        if insert_id:
                            return insert_id
                return self._coerce_insert_id(getattr(result, "lastrowid", None))
        except Exception as exc:
            raise self._connector.handle_exception(exc, sql=str(stmt), operation="insert") from exc

    def _execute_many(self, stmt: ClauseElement, rows: Sequence[Mapping[str, Any]]) -> int:
        try:
            with self._connector.begin() as conn:
                result = conn.execute(stmt, list(rows))
                rowcount = result.rowcount
                if rowcount is None:
                    return len(rows)
                return int(rowcount)
        except Exception as exc:
            raise self._connector.handle_exception(exc, sql=str(stmt), operation="bulk insert") from exc

    def _build_upsert(
        self,
        table: Table,
        row: Mapping[str, Any],
        conflict_columns: Sequence[str],
    ) -> Optional[ClauseElement]:
        dialect = self._connector.dialect_name
        conflict_columns = list(conflict_columns)

        if dialect == "sqlite":
            insert_stmt = sqlite_insert(table).values(**row)
            update_values = {k: insert_stmt.excluded[k] for k in row if k not in conflict_columns}
            if update_values:
                return insert_stmt.on_conflict_do_update(index_elements=conflict_columns, set_=update_values)
            return insert_stmt.on_conflict_do_nothing(index_elements=conflict_columns)

        if dialect == "postgresql":
            insert_stmt = pg_insert(table).values(**row)
            update_values = {k: insert_stmt.excluded[k] for k in row if k not in conflict_columns}
            if update_values:
                return insert_stmt.on_conflict_do_update(index_elements=conflict_columns, set_=update_values)
            return insert_stmt.on_conflict_do_nothing(index_elements=conflict_columns)

        if dialect in {"mysql", "mariadb"}:
            insert_stmt = mysql_insert(table).values(**row)
            update_values = {k: insert_stmt.inserted[k] for k in row if k not in conflict_columns}
            if update_values:
                return insert_stmt.on_duplicate_key_update(**update_values)
            return insert_stmt

        return None

    def _fallback_upsert(
        self,
        table_handle: SQLAlchemyTable,
        row: Mapping[str, Any],
        conflict_columns: Sequence[str],
    ) -> int:
        try:
            return table_handle.insert(row)
        except DatusException as exc:
            if exc.code != ErrorCode.DB_CONSTRAINT_VIOLATION:
                raise
            update_values = {k: v for k, v in row.items() if k not in conflict_columns}
            if not update_values:
                return 0
            where = self._build_conflict_where(conflict_columns, row)
            table_handle.update(where, update_values)
            return 0

    def _build_conflict_where(self, conflict_columns: Sequence[str], row: Mapping[str, Any]) -> FilterExpr:
        conditions = []
        for col in conflict_columns:
            if col not in row:
                raise ValueError(f"Missing conflict column '{col}' in upsert row")
            conditions.append(Condition(col, Op.EQ, row[col]))
        if not conditions:
            raise ValueError("Conflict columns required for upsert")
        if len(conditions) == 1:
            return conditions[0]
        return And(conditions)

    def _create_table_with_capability(self, table: Table, schema: TableSchema) -> None:
        if self._ddl_mode == "disabled":
            self._ddl_capable = False
            raise DatusException(ErrorCode.DB_TABLE_NOT_EXISTS, message_args={"table_name": schema.name})

        try:
            table.create(self._connector.engine, checkfirst=True)
            self._ddl_capable = True
        except Exception as exc:
            self._ddl_capable = False
            handled = self._connector.handle_exception(exc, operation="create table")
            if self._ddl_mode == "required":
                raise handled
            if not self.table_exists(schema.name):
                raise handled
            logger.warning(f"DDL failed for {schema.name}, continuing with existing table: {handled}")

        self._validate_schema(schema)

    def _validate_schema(self, schema: TableSchema) -> None:
        inspector = inspect(self._connector.engine)
        try:
            columns = inspector.get_columns(schema.name)
        except SQLAlchemyError as exc:
            raise self._connector.handle_exception(exc, operation="inspect columns") from exc
        existing = {col["name"] for col in columns}
        missing = [col.name for col in schema.columns if col.name not in existing]
        if missing:
            raise DatusException(
                ErrorCode.DB_EXECUTION_ERROR,
                message_args={"error_message": f"Table {schema.name} missing columns: {', '.join(missing)}"},
            )

    def _build_table(self, schema: TableSchema) -> Table:
        existing = self._metadata.tables.get(schema.name)
        if existing is not None:
            return existing

        columns = [self._build_column(col) for col in schema.columns]
        constraints: List[Any] = []
        for unique_cols in schema.unique_constraints:
            constraints.append(UniqueConstraint(*unique_cols))

        table = Table(schema.name, self._metadata, *columns, *constraints)

        for index in schema.indexes:
            self._build_index(index, table)

        return table

    def _build_column(self, spec: ColumnSpec) -> Column:
        col_type = self._map_type(spec.data_type)
        return Column(
            spec.name,
            col_type,
            primary_key=spec.primary_key,
            autoincrement=spec.autoincrement,
            nullable=spec.nullable,
            unique=spec.unique,
            default=spec.default,
        )

    def _build_index(self, spec: IndexSpec, table: Table) -> Index:
        cols = [table.c[col] for col in spec.columns]
        return Index(spec.name, *cols, unique=spec.unique)

    def _map_type(self, data_type: str):
        dtype = data_type.strip().upper()
        if dtype.startswith("VARCHAR") or dtype.startswith("CHAR"):
            length = self._parse_length(dtype)
            return String(length) if length else String()
        if dtype in {"TEXT", "STRING"}:
            return Text()
        if dtype in {"INTEGER", "INT"}:
            return Integer()
        if dtype == "BIGINT":
            return BigInteger()
        if dtype == "SMALLINT":
            return SmallInteger()
        if dtype in {"REAL", "FLOAT", "DOUBLE"}:
            return Float()
        if dtype in {"BLOB", "BINARY"}:
            return LargeBinary()
        if dtype in {"BOOL", "BOOLEAN"}:
            return Boolean()
        if dtype == "DATE":
            return Date()
        if dtype in {"DATETIME", "TIMESTAMP"}:
            return DateTime()
        return Text()

    @staticmethod
    def _parse_length(dtype: str) -> Optional[int]:
        if "(" not in dtype or ")" not in dtype:
            return None
        start = dtype.find("(") + 1
        end = dtype.find(")", start)
        try:
            return int(dtype[start:end])
        except ValueError:
            return None

    @staticmethod
    def _coerce_insert_id(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _compile_filter(self, expr: Optional[FilterExpr], table: Table) -> Optional[ClauseElement]:
        if expr is None:
            return None
        if isinstance(expr, ClauseElement):
            return expr
        if isinstance(expr, UnsafeRawSQL):
            stripped = expr.sql.strip()
            return text(stripped) if stripped else None
        if isinstance(expr, str):
            stripped = expr.strip()
            if not stripped:
                return None
            raise ValueError("Raw string filters are not allowed. Use condition nodes or UnsafeRawSQL.")
        return self._compile_node(expr, table)

    def _compile_node(self, node: Any, table: Table) -> ClauseElement:
        if isinstance(node, Condition) or hasattr(node, "field"):
            return self._compile_condition(node, table)
        if isinstance(node, And) or getattr(node, "__class__", None).__name__ == "And":
            parts = [self._compile_node(n, table) for n in node.nodes if n is not None]
            return sa_true() if not parts else sa_and(*parts)
        if isinstance(node, Or) or getattr(node, "__class__", None).__name__ == "Or":
            parts = [self._compile_node(n, table) for n in node.nodes if n is not None]
            return sa_false() if not parts else sa_or(*parts)
        if isinstance(node, Not) or getattr(node, "__class__", None).__name__ == "Not":
            return sa_not(self._compile_node(node.node, table))
        raise TypeError(f"Unsupported filter node: {type(node)}")

    def _compile_condition(self, cond: Any, table: Table) -> ClauseElement:
        field = cond.field
        if field not in table.c:
            raise ValueError(f"Unknown column '{field}' in filter")
        column = table.c[field]

        op = cond.op
        op_value = op.value if hasattr(op, "value") else str(op)
        value = cond.value

        if value is None:
            if op_value == "=" or op == Op.EQ:
                return column.is_(None)
            if op_value == "!=" or op == Op.NE:
                return column.is_not(None)
            raise ValueError(f"Operator {op} is invalid with NULL (field: {field})")

        if op_value == "IN" or op == Op.IN:
            if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
                raise TypeError("IN expects a non-string iterable value")
            values = list(value)
            if not values:
                return sa_false()
            non_null = [v for v in values if v is not None]
            include_null = any(v is None for v in values)
            parts: List[ClauseElement] = []
            if non_null:
                parts.append(column.in_(non_null))
            if include_null:
                parts.append(column.is_(None))
            if len(parts) == 1:
                return parts[0]
            return sa_or(*parts)

        if op_value == "LIKE" or op == Op.LIKE:
            return column.like(value)

        if op_value in {"=", "!=", ">", ">=", "<", "<="} or op in {
            Op.EQ,
            Op.NE,
            Op.GT,
            Op.GTE,
            Op.LT,
            Op.LTE,
        }:
            if op_value == "=":
                return column == value
            if op_value == "!=":
                return column != value
            if op_value == ">":
                return column > value
            if op_value == ">=":
                return column >= value
            if op_value == "<":
                return column < value
            if op_value == "<=":
                return column <= value

        raise ValueError(f"Unsupported operator: {op}")
