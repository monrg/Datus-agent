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
from .registry import register_vector_backend

LanceBackend = None
try:
    from .lance import LanceBackend  # type: ignore[assignment]
except ImportError:
    LanceBackend = None
else:
    register_vector_backend("lancedb", LanceBackend)

__all__ = [
    "And",
    "BackendCapabilities",
    "Condition",
    "FilterCompiler",
    "FilterExpr",
    "Not",
    "Op",
    "Or",
    "TableSpec",
    "VectorBackend",
    "VectorTable",
    "register_vector_backend",
]

if LanceBackend is not None:
    __all__.append("LanceBackend")
    __all__.sort()
