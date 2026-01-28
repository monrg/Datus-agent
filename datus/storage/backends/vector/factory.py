# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from typing import Optional

from datus.storage.backends.vector.interfaces import VectorBackend
from datus.storage.backends.vector.lance import LanceBackend
from datus.storage.backends.vector.registry import get_vector_backend


def _build_pg_connection_string(db_config) -> str:
    from urllib.parse import quote_plus

    username = quote_plus(db_config.username or "")
    password = quote_plus(db_config.password or "")
    host = db_config.host or "127.0.0.1"
    port = db_config.port or 5432
    database = db_config.database or "postgres"
    sslmode = "prefer"
    if db_config.extra and isinstance(db_config.extra, dict):
        sslmode = db_config.extra.get("sslmode", sslmode)
    return f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}?sslmode={sslmode}"


def get_default_backend(db_path: str, agent_config: Optional[object] = None) -> VectorBackend:
    if agent_config is not None:
        namespace = getattr(agent_config, "storage_backend_namespace", None)
        if callable(namespace):
            namespace_name = agent_config.storage_backend_namespace("vector")
            if namespace_name:
                db_config = agent_config.resolve_storage_db_config("vector")
                if db_config and db_config.type in ("postgresql", "postgres"):
                    backend_cls = get_vector_backend("pgvector")
                    if backend_cls is None:
                        return LanceBackend(db_path)
                    connection_string = _build_pg_connection_string(db_config)
                    schema = db_config.schema or "public"
                    return backend_cls(
                        db_path=db_path,
                        connection_string=connection_string,
                        schema=schema,
                        namespace=agent_config.current_namespace,
                    )
    return LanceBackend(db_path)
