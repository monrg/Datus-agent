# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class RelationalBackend(Protocol):
    name: str
    integrity_error: type[Exception]
    location: Optional[str]

    def connect(self) -> Any:
        """Return a DB-API compatible connection."""


@dataclass
class SqliteBackend:
    base_path: str
    filename: str
    connect_kwargs: Optional[dict[str, Any]] = None

    name: str = "sqlite"
    integrity_error: type[Exception] = sqlite3.IntegrityError
    location: Optional[str] = None

    def __post_init__(self) -> None:
        os.makedirs(self.base_path, exist_ok=True)
        self.location = os.path.join(self.base_path, self.filename)

    def connect(self) -> sqlite3.Connection:
        kwargs = self.connect_kwargs or {}
        return sqlite3.connect(self.location, **kwargs)


class DBManager:
    def __init__(self, backend: RelationalBackend):
        self.backend = backend

    @property
    def location(self) -> Optional[str]:
        return self.backend.location

    @property
    def integrity_error(self) -> type[Exception]:
        return self.backend.integrity_error

    @contextmanager
    def connection(self):
        conn = None
        try:
            conn = self.backend.connect()
            yield conn
        except Exception as exc:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    logger.warning("Failed to rollback transaction after error.")
            raise DatusException(
                ErrorCode.DB_CONNECTION_FAILED,
                message=f"Database connection error ({self.backend.name}): {str(exc)}",
            ) from exc
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    logger.warning("Failed to close DB connection.")


def get_default_db_manager(db_path: str, filename: str) -> DBManager:
    return DBManager(SqliteBackend(base_path=db_path, filename=filename))
