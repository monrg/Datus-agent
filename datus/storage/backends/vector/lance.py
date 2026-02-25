# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

import time
from typing import Any, Mapping, Optional, Sequence

import lancedb
import pandas as pd
import pyarrow as pa
from lancedb.embeddings import EmbeddingFunctionConfig

from datus.storage.backends.vector.interfaces import (
    BackendCapabilities,
    FilterCompiler,
    FilterExpr,
    TableSpec,
    VectorBackend,
    VectorTable,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class LanceFilterCompiler(FilterCompiler):
    def compile(self, expr: Optional[FilterExpr]) -> Optional[str]:
        if expr is None:
            return None
        try:
            from datus.storage import lancedb_conditions as lc

            return lc.build_where(expr)
        except Exception as exc:
            logger.warning(f"Failed to compile filter expression for LanceDB: {exc}")
            return None


class LanceTable(VectorTable):
    def __init__(self, db: lancedb.LanceDBConnection, table: Any, spec: TableSpec, compiler: LanceFilterCompiler):
        self._db = db
        self._table = table
        self._spec = spec
        self._compiler = compiler
        self.name = spec.name

    def _compile_where(self, where: Optional[FilterExpr]) -> Optional[str]:
        return self._compiler.compile(where)

    def _fill_query(
        self,
        query_builder,
        select: Optional[Sequence[str]],
        where_clause: Optional[str],
    ):
        if where_clause:
            query_builder = query_builder.where(where_clause)
        if select:
            query_builder = query_builder.select(list(select))
        return query_builder

    def _add_with_retry(self, frame: pd.DataFrame, max_attempts: int = 3, initial_delay: float = 0.05) -> None:
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                self._table.add(frame)
                return
            except Exception as err:
                error_message = str(err)
                if "Commit conflict" not in error_message:
                    raise err
                last_error = err
                delay = initial_delay * (attempt + 1)
                logger.warning(
                    f"Commit conflict detected when writing to LanceDB table '{self.name}' "
                    f"(attempt {attempt + 1}/{max_attempts}). Retrying after {delay:.2f}s."
                )
                self._table = self._db.open_table(self.name)
                time.sleep(delay)
        assert last_error is not None
        raise last_error

    def _upsert_with_retry(
        self, frame: pd.DataFrame, on_column: str, max_attempts: int = 3, initial_delay: float = 0.05
    ) -> None:
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                self._table.merge_insert(on_column).when_matched_update_all().when_not_matched_insert_all().execute(
                    frame
                )
                return
            except Exception as err:
                error_message = str(err)
                if "Commit conflict" not in error_message:
                    raise err
                last_error = err
                delay = initial_delay * (attempt + 1)
                logger.warning(
                    f"Commit conflict detected when upserting to LanceDB table '{self.name}' "
                    f"(attempt {attempt + 1}/{max_attempts}). Retrying after {delay:.2f}s."
                )
                self._table = self._db.open_table(self.name)
                time.sleep(delay)
        assert last_error is not None
        raise last_error

    def add(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        frame = pd.DataFrame(rows)
        self._add_with_retry(frame)

    def upsert(self, rows: Sequence[Mapping[str, Any]], on: str) -> None:
        if not rows:
            return
        frame = pd.DataFrame(rows)
        self._upsert_with_retry(frame, on)

    def update(self, where: FilterExpr, values: Mapping[str, Any]) -> None:
        where_clause = self._compile_where(where)
        if not where_clause:
            return
        self._table.update(where=where_clause, values=dict(values))

    def delete(self, where: FilterExpr) -> None:
        where_clause = self._compile_where(where)
        if not where_clause:
            return
        self._table.delete(where_clause)

    def count(self, where: Optional[FilterExpr] = None) -> int:
        where_clause = self._compile_where(where)
        if where_clause:
            return self._table.count_rows(where_clause)
        return self._table.count_rows()

    def search_all(
        self,
        where: Optional[FilterExpr] = None,
        select: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> pa.Table:
        where_clause = self._compile_where(where)
        query_builder = self._table.search()
        query_builder = self._fill_query(query_builder, select, where_clause)
        if limit is None:
            limit = self.count(where)
        return query_builder.limit(limit).to_arrow()

    def search_vector(
        self,
        vector: Sequence[float],
        top_n: int,
        where: Optional[FilterExpr] = None,
        select: Optional[Sequence[str]] = None,
    ) -> pa.Table:
        where_clause = self._compile_where(where)
        query_builder = self._table.search(
            query=vector,
            query_type="vector",
            vector_column_name=self._spec.vector_column,
        )
        query_builder = self._fill_query(query_builder, select, where_clause)
        return query_builder.limit(top_n).to_arrow()

    def search_text(
        self,
        text: str,
        top_n: int,
        where: Optional[FilterExpr] = None,
        select: Optional[Sequence[str]] = None,
    ) -> pa.Table:
        where_clause = self._compile_where(where)
        query_builder = self._table.search(
            query=text,
            query_type="vector",
            vector_column_name=self._spec.vector_column,
        )
        query_builder = self._fill_query(query_builder, select, where_clause)
        return query_builder.limit(top_n).to_arrow()

    def search_hybrid(
        self,
        text: str,
        top_n: int,
        where: Optional[FilterExpr] = None,
        select: Optional[Sequence[str]] = None,
        reranker: Optional[Any] = None,
        vector: Optional[Sequence[float]] = None,
    ) -> pa.Table:
        where_clause = self._compile_where(where)
        query_builder = self._table.search(
            query=text,
            query_type="hybrid",
            vector_column_name=self._spec.vector_column,
        )
        query_builder = self._fill_query(query_builder, select, where_clause)
        if reranker:
            results = query_builder.limit(top_n * 2).rerank(reranker).to_arrow()
            return results[:top_n] if len(results) > top_n else results
        return query_builder.limit(top_n).to_arrow()

    def create_vector_index(self, **opts: Any) -> None:
        self._table.create_index(**opts)

    def create_fts_index(self, fields: Sequence[str], **opts: Any) -> None:
        if not fields:
            return
        if isinstance(fields, str):
            fields = [fields]
        self._table.create_fts_index(field_names=list(fields), **opts)

    def create_scalar_index(self, fields: Sequence[str], **opts: Any) -> None:
        if not fields:
            return
        if isinstance(fields, str):
            fields = [fields]
        for field in fields:
            self._table.create_scalar_index(field, **opts)


class LanceBackend(VectorBackend):
    name = "lancedb"
    caps = BackendCapabilities(
        vector_search=True, hybrid_search=True, fts=True, scalar_index=True, native_embedding=True
    )

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = lancedb.connect(db_path)
        self._compiler = LanceFilterCompiler()

    def ensure_table(self, spec: TableSpec) -> LanceTable:
        if spec.name in self.db.table_names(limit=100):
            table = self.db.open_table(spec.name)
        else:
            create_kwargs: dict[str, Any] = {"schema": spec.schema, "exist_ok": True}
            if spec.embedding_function and spec.text_source:
                create_kwargs["embedding_functions"] = [
                    EmbeddingFunctionConfig(
                        vector_column=spec.vector_column,
                        source_column=spec.text_source,
                        function=spec.embedding_function,
                    )
                ]
            table = self.db.create_table(spec.name, **create_kwargs)
        return LanceTable(self.db, table, spec, self._compiler)

    def drop_table(self, name: str) -> None:
        try:
            if name in self.db.table_names():
                self.db.drop_table(name)
        except Exception as exc:
            logger.warning(f"Failed to drop LanceDB table '{name}': {exc}")

    def table_exists(self, name: str) -> bool:
        try:
            return name in self.db.table_names(limit=100)
        except Exception:
            return False
