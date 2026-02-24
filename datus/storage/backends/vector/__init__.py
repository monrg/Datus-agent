# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from .interfaces import (
    And,
    BackendCapabilities,
    Condition,
    FilterCompiler,
    FilterExpr,
    Not,
    Op,
    Or,
    TableSpec,
    VectorBackend,
    VectorTable,
)
from .lance import LanceBackend
from .registry import register_vector_backend

register_vector_backend("lancedb", LanceBackend)

__all__ = [
    # Vector backend types
    "And",
    "BackendCapabilities",
    "Condition",
    "FilterExpr",
    "FilterCompiler",
    "Not",
    "Op",
    "Or",
    "TableSpec",
    "VectorBackend",
    "VectorTable",
    "register_vector_backend",
    "LanceBackend",
]
