# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence, Union

import pyarrow as pa


class Op(str, Enum):
    EQ = "="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    IN = "IN"
    LIKE = "LIKE"


@dataclass(frozen=True)
class Condition:
    field: str
    op: Op
    value: Any


@dataclass(frozen=True)
class And:
    nodes: Sequence["FilterExpr"]


@dataclass(frozen=True)
class Or:
    nodes: Sequence["FilterExpr"]


@dataclass(frozen=True)
class Not:
    node: "FilterExpr"


# NOTE: FilterExpr is intentionally flexible for now so existing
# lancedb_conditions nodes can be passed through during migration.
FilterExpr = Union[Condition, And, Or, Not, Any]


class FilterCompiler(Protocol):
    def compile(self, expr: Optional[FilterExpr]) -> Any:
        """Compile FilterExpr to backend-native where clause."""


@dataclass(frozen=True)
class BackendCapabilities:
    vector_search: bool = True
    hybrid_search: bool = False
    fts: bool = False
    scalar_index: bool = True
    native_embedding: bool = False


@dataclass(frozen=True)
class TableSpec:
    name: str
    schema: Any
    vector_column: str
    vector_dim: int
    text_source: Optional[str] = None
    embedding_function: Optional[Any] = None


class VectorTable(Protocol):
    name: str

    def add(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Insert rows into the table."""

    def upsert(self, rows: Sequence[Mapping[str, Any]], on: str) -> None:
        """Upsert rows by a unique column."""

    def update(self, where: FilterExpr, values: Mapping[str, Any]) -> None:
        """Update rows matching filter."""

    def delete(self, where: FilterExpr) -> None:
        """Delete rows matching filter."""

    def count(self, where: Optional[FilterExpr] = None) -> int:
        """Count rows matching filter."""

    def search_all(
        self,
        where: Optional[FilterExpr] = None,
        select: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> pa.Table:
        """Return all rows matching filter."""

    def search_vector(
        self,
        vector: Sequence[float],
        top_n: int,
        where: Optional[FilterExpr] = None,
        select: Optional[Sequence[str]] = None,
    ) -> pa.Table:
        """Vector search using pre-computed embedding vector."""

    def search_text(
        self,
        text: str,
        top_n: int,
        where: Optional[FilterExpr] = None,
        select: Optional[Sequence[str]] = None,
    ) -> pa.Table:
        """Text search using backend-native embedding or FTS."""

    def search_hybrid(
        self,
        text: str,
        top_n: int,
        where: Optional[FilterExpr] = None,
        select: Optional[Sequence[str]] = None,
        reranker: Optional[Any] = None,
        vector: Optional[Sequence[float]] = None,
    ) -> pa.Table:
        """Hybrid search (vector + lexical) when supported."""

    def create_vector_index(self, **opts: Any) -> None:
        """Create vector index with backend-specific options."""

    def create_fts_index(self, fields: Sequence[str], **opts: Any) -> None:
        """Create full-text index when supported."""

    def create_scalar_index(self, fields: Sequence[str], **opts: Any) -> None:
        """Create scalar index when supported."""


class VectorBackend(Protocol):
    name: str
    caps: BackendCapabilities

    def ensure_table(self, spec: TableSpec) -> VectorTable:
        """Create table if needed and return a handle."""

    def drop_table(self, name: str) -> None:
        """Drop a table and its associated data."""

    def table_exists(self, name: str) -> bool:
        """Return True if table exists in backend."""
