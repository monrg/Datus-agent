# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""SQLAlchemy connection helper for relational storage."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import (
    DatabaseError,
    DataError,
    IntegrityError,
    InterfaceError,
    InternalError,
    NoSuchTableError,
    NotSupportedError,
    OperationalError,
    ProgrammingError,
    SQLAlchemyError,
    TimeoutError,
)
from sqlalchemy.pool import NullPool

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SQLAlchemyConnector:
    """SQLAlchemy connector with standardized error handling."""

    def __init__(self, connection_string: str, engine_options: Optional[Dict[str, Any]] = None):
        self.connection_string = connection_string
        self._engine_options = engine_options or {}
        self._engine: Optional[Engine] = None
        self._dialect_name = ""
        self._create_engine()

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._create_engine()
        return self._engine  # type: ignore[return-value]

    @property
    def dialect_name(self) -> str:
        return self._dialect_name

    def dispose(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    @contextmanager
    def begin(self) -> Iterator[Any]:
        try:
            with self.engine.begin() as conn:
                yield conn
        except Exception as exc:
            raise self.handle_exception(exc) from exc

    def handle_exception(self, exc: Exception, sql: str = "", operation: str = "database operation") -> DatusException:
        """Map SQLAlchemy exceptions to Datus exceptions."""
        if isinstance(exc, DatusException):
            return exc

        error_message = self._extract_error_message(exc)
        error_msg_lower = error_message.lower()
        message_args = {"error_message": error_message, "sql": sql, "operation": operation}

        if isinstance(exc, NoSuchTableError):
            return DatusException(ErrorCode.DB_TABLE_NOT_EXISTS, message_args={"table_name": str(exc)})

        if isinstance(exc, IntegrityError):
            return DatusException(ErrorCode.DB_CONSTRAINT_VIOLATION, message_args=message_args)

        if isinstance(exc, TimeoutError):
            return DatusException(ErrorCode.DB_CONNECTION_TIMEOUT, message_args=message_args)

        if isinstance(exc, (OperationalError, InterfaceError)):
            if "permission" in error_msg_lower or "denied" in error_msg_lower:
                return DatusException(
                    ErrorCode.DB_PERMISSION_DENIED,
                    message_args={"operation": operation, "error_message": error_message},
                )
            if "authentication" in error_msg_lower or "access denied" in error_msg_lower:
                return DatusException(ErrorCode.DB_AUTHENTICATION_FAILED, message_args=message_args)
            if "timeout" in error_msg_lower or "timed out" in error_msg_lower:
                return DatusException(ErrorCode.DB_CONNECTION_TIMEOUT, message_args=message_args)
            if "connection" in error_msg_lower:
                return DatusException(ErrorCode.DB_CONNECTION_FAILED, message_args=message_args)
            return DatusException(ErrorCode.DB_EXECUTION_ERROR, message_args=message_args)

        if isinstance(exc, (ProgrammingError, DataError, InternalError, DatabaseError, NotSupportedError)):
            if "syntax" in error_msg_lower or "parse" in error_msg_lower:
                return DatusException(ErrorCode.DB_EXECUTION_SYNTAX_ERROR, message_args=message_args)
            return DatusException(ErrorCode.DB_EXECUTION_ERROR, message_args=message_args)

        if isinstance(exc, SQLAlchemyError):
            return DatusException(ErrorCode.DB_EXECUTION_ERROR, message_args=message_args)

        return DatusException(ErrorCode.DB_FAILED, message_args=message_args)

    def _create_engine(self) -> None:
        url = make_url(self.connection_string)
        self._dialect_name = url.get_backend_name()
        options = dict(self._engine_options)
        options.setdefault("future", True)

        if self._dialect_name == "sqlite":
            options.setdefault("poolclass", NullPool)
            connect_args = dict(options.get("connect_args", {}))
            connect_args.setdefault("check_same_thread", False)
            options["connect_args"] = connect_args
        else:
            options.setdefault("pool_pre_ping", True)

        self._engine = create_engine(self.connection_string, **options)
        logger.debug(f"SQLAlchemy engine created for {self._dialect_name}")

    @staticmethod
    def _extract_error_message(exc: Exception) -> str:
        if hasattr(exc, "detail") and exc.detail:
            detail = exc.detail
            if isinstance(detail, list):
                return "\n".join(detail)
            return str(detail)
        if hasattr(exc, "orig") and exc.orig is not None:
            return str(exc.orig)
        return str(exc)
