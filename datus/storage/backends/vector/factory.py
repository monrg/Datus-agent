# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from typing import Optional

from datus.storage.backends.vector.interfaces import VectorBackend
from datus.storage.backends.vector.registry import get_vector_backend


def _build_pg_connection_string(db_config) -> str:
    from urllib.parse import quote_plus

    if getattr(db_config, "uri", ""):
        return db_config.uri

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
    def build_lance_backend() -> VectorBackend:
        from datus.storage.backends.vector.lance import LanceBackend

        return LanceBackend(db_path)

    if agent_config is not None:
        namespace_fn = getattr(agent_config, "storage_backend_namespace", None)
        resolve_db_config_fn = getattr(agent_config, "resolve_storage_db_config", None)
        if callable(namespace_fn) and callable(resolve_db_config_fn):
            try:
                namespace_name = namespace_fn("vector")
            except Exception:
                namespace_name = None
            if namespace_name:
                try:
                    db_config = resolve_db_config_fn("vector")
                except Exception:
                    db_config = None
                if db_config and db_config.type in ("postgresql", "postgres"):
                    backend_cls = get_vector_backend("pgvector")
                    if backend_cls is None:
                        return build_lance_backend()
                    connection_string = _build_pg_connection_string(db_config)
                    schema = db_config.schema or "public"
                    namespace = getattr(agent_config, "current_namespace", "")
                    return backend_cls(
                        db_path=db_path,
                        connection_string=connection_string,
                        schema=schema,
                        namespace=namespace,
                    )
    return build_lance_backend()
