# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Registry for vector storage backends."""

from __future__ import annotations

from typing import Dict, Type

from datus.storage.backends.plugin_loader import try_load_storage_plugin
from datus.storage.backends.vector.interfaces import VectorBackend
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_VECTOR_BACKENDS: Dict[str, Type[VectorBackend]] = {}


def register_vector_backend(name: str, backend_class: Type[VectorBackend]) -> None:
    _VECTOR_BACKENDS[name.lower()] = backend_class
    logger.debug(f"Registered vector backend: {name}")


def get_vector_backend(name: str) -> Type[VectorBackend] | None:
    key = name.lower()
    if key not in _VECTOR_BACKENDS:
        try_load_storage_plugin(key)
    return _VECTOR_BACKENDS.get(key)


def list_vector_backends() -> Dict[str, Type[VectorBackend]]:
    return _VECTOR_BACKENDS.copy()
