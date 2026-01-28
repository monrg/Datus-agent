# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Relational database backend package."""

from .filter_compiler import SQLFilterCompiler, compile_filter
from .interfaces import (
    ColumnSpec,
    IndexSpec,
    RelationalBackend,
    RelationalCapabilities,
    RelationalTable,
    TableSchema,
    TransactionContext,
)
from .sqlite_backend import SQLiteBackend, SQLiteTable, SQLiteTransaction
from .sqlalchemy_backend import SQLAlchemyBackend, SQLAlchemyTable, SQLAlchemyTransaction
from .sqlalchemy_connector import SQLAlchemyConnector

__all__ = [
    # Interfaces
    "ColumnSpec",
    "IndexSpec",
    "RelationalBackend",
    "RelationalCapabilities",
    "RelationalTable",
    "TableSchema",
    "TransactionContext",
    # SQLite implementation
    "SQLiteBackend",
    "SQLiteTable",
    "SQLiteTransaction",
    # SQLAlchemy implementation
    "SQLAlchemyBackend",
    "SQLAlchemyTable",
    "SQLAlchemyTransaction",
    "SQLAlchemyConnector",
    # Filter compiler
    "SQLFilterCompiler",
    "compile_filter",
]
