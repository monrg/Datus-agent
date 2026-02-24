# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Union

import pyarrow as pa
from lancedb.pydantic import LanceModel
from lancedb.rerankers import Reranker
from pydantic import Field

from datus.storage.backends.vector.factory import get_default_backend
from datus.storage.backends.vector.interfaces import TableSpec, VectorBackend, VectorTable
from datus.storage.embedding_models import EmbeddingModel
from datus.storage.lancedb_conditions import WhereExpr
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class StorageBase:
    """Base class for all storage components using vector backends."""

    def __init__(self, db_path: str, backend: Optional[VectorBackend] = None):
        """Initialize the storage base.

        Args:
            db_path: Path to the vector database directory
            backend: Optional vector backend instance
        """
        self.db_path = db_path
        self.backend = backend or get_default_backend(db_path)

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat()


class BaseModelData(LanceModel):
    created_at: str = Field(init=True, default="")

    class Config:
        arbitrary_types_allowed = True


class BaseEmbeddingStore(StorageBase):
    """Base class for all embedding stores using LanceDB.
    table_name: the name of the table to store the embedding
    embedding_field: the field name of the embedding
    """

    def __init__(
        self,
        db_path: str,
        table_name: str,
        embedding_model: EmbeddingModel,
        on_duplicate_columns: str = "vector",
        schema: Optional[Union[pa.Schema, LanceModel]] = None,
        vector_source_name: str = "definition",
        vector_column_name: str = "vector",
        backend: Optional[VectorBackend] = None,
    ):
        super().__init__(db_path, backend=backend)
        self.model = embedding_model
        self.batch_size = embedding_model.batch_size
        self.table_name = table_name
        self.vector_source_name = vector_source_name
        self.vector_column_name = vector_column_name
        self.on_duplicate_columns = on_duplicate_columns
        self._schema = schema
        # Delay table initialization until first use
        self.table: Optional[VectorTable] = None
        self._table_initialized = False
        self._table_lock = Lock()
        self._write_lock = Lock()

    def _ensure_table_ready(self):
        """Ensure table is ready for operations, with proper error handling."""
        if self._table_initialized:
            return

        with self._table_lock:
            if self._table_initialized:
                return

            # First check if embedding model is available
            self._check_embedding_model_ready()
            # Initialize table with embedding function
            self._ensure_table(self._schema)
            self._table_initialized = True
            logger.debug(f"Table {self.table_name} initialized successfully with embedding function")

    def _search_all(
        self, where: WhereExpr = None, select_fields: Optional[List[str]] = None, limit: Optional[int] = None
    ) -> pa.Table:
        self._ensure_table_ready()
        result = self.table.search_all(where=where, select=select_fields, limit=limit)
        if self.vector_column_name in result.column_names:
            result = result.drop([self.vector_column_name])
        return result

    def _check_embedding_model_ready(self):
        """Check if embedding model is ready for use."""
        # Check if model has failed before
        if self.model.is_model_failed:
            raise DatusException(
                ErrorCode.MODEL_EMBEDDING_ERROR,
                message=(
                    f"Embedding model '{self.model.model_name}' is not available:" f" {self.model.model_error_message}"
                ),
            )

        # Try to access the model (this will trigger lazy loading)
        try:
            _ = self.model.model
        except DatusException as e:
            # Re-raise DatusException directly to avoid nesting
            raise e
        except Exception as e:
            raise DatusException(
                ErrorCode.MODEL_EMBEDDING_ERROR,
                message=f"Embedding model '{self.model.model_name}' initialization failed: {str(e)}",
            ) from e

    def _ensure_table(self, schema: Optional[Union[pa.Schema, LanceModel]] = None):
        try:
            embedding_function = self.model.model if self.backend.caps.native_embedding else None
            spec = TableSpec(
                name=self.table_name,
                schema=schema,
                vector_column=self.vector_column_name,
                vector_dim=self.model.dim_size,
                text_source=self.vector_source_name,
                embedding_function=embedding_function,
            )
            self.table = self.backend.ensure_table(spec)
        except Exception as e:
            raise DatusException(
                ErrorCode.STORAGE_TABLE_OPERATION_FAILED,
                message_args={"operation": "create_table", "table_name": self.table_name, "error_message": str(e)},
            ) from e

    def create_vector_index(
        self,
        metric: str = "cosine",
    ):
        """
        Create a vector index (IVF_PQ or IVF_FLAT) for the table to optimize vector search.

        Args:
            metric (str): Distance metric for vector search ('cosine', 'l2', or 'dot').
                Default: 'cosine'.
            accelerator (str): Optional accelerator ('cuda' for GPU, 'mps' for MPS, None for CPU).
                Default: none.
        """
        self._ensure_table_ready()
        try:
            row_count = self.table.count()
            logger.debug(f"Creating vector index for {self.table_name} with {row_count} rows")

            # Determine index type based on dataset size
            index_type = "IVF_PQ" if row_count >= 5000 else "IVF_FLAT"
            logger.debug(f"Selected index type: {index_type}")

            # Calculate number of partitions (IVF)
            # Rule: ~sqrt(n) for large datasets, minimum 1, capped at 1024
            num_partitions = max(1, min(1024, int(row_count**0.5)))
            if row_count < 1000:
                num_partitions = max(1, row_count // 10)  # Small datasets: 10 vectors per partition
            elif row_count < 5000:
                num_partitions = max(1, row_count // 20)  # Medium datasets: 20 vectors per partition
            logger.debug(f"Number of partitions: {num_partitions}")

            # Calculate number of sub-vectors (PQ, only for IVF_PQ)
            # Rule: 8-96, based on vector dimension and dataset size
            num_sub_vectors = 32  # Default for medium datasets
            if index_type == "IVF_PQ":
                # Get vector dimension (e.g., 1024 for bge-large-en-v1.5)
                vector_dim = self.model.dim_size

                if row_count < 1000:
                    num_sub_vectors = min(16, max(8, vector_dim // 64))  # Small datasets: fewer sub-vectors
                elif row_count < 5000:
                    num_sub_vectors = min(32, max(16, vector_dim // 32))  # Medium datasets
                else:
                    num_sub_vectors = min(96, max(32, vector_dim // 16))  # Large datasets: more sub-vectors
                logger.debug(f"Number of sub-vectors: {num_sub_vectors}")

            # Create index with calculated parameters
            index_params = {
                "metric": metric,
                "vector_column_name": self.vector_column_name,
                "index_type": index_type,
                "num_partitions": num_partitions,
                "replace": True,  # Replace existing index if any
            }
            if index_type == "IVF_PQ":
                index_params["num_sub_vectors"] = num_sub_vectors
            accelerator = self.model.device
            if accelerator and accelerator == "cuda" or accelerator == "mps":
                index_params["accelerator"] = accelerator

            self.table.create_vector_index(**index_params)
            logger.debug(f"Successfully created {index_type} index for {self.table_name}")

        except Exception as e:
            # Does not affect usage, so no exception is thrown.
            logger.warning(f"Failed to create vector index for {self.table_name}: {str(e)}")

    def create_fts_index(self, field_names: Union[str, List[str]]):
        self._ensure_table_ready()
        try:
            fields = field_names if isinstance(field_names, list) else [field_names]
            self.table.create_fts_index(fields=fields, replace=True)
        except Exception as e:
            # Does not affect usage, so no exception is thrown.
            logger.warning(f"Failed to create fts index for {self.table_name} table: {str(e)}")

    def store_batch(self, data: List[Dict[str, Any]]):
        """
        Store a batch of data in the database. The following steps are performed:

            1. Encode the vector field
            2. Merge insert the data into the table

        Args:
            data: List[BaseModelData] the data to store
            on_columns: List[str] the columns to merge on duplicate
        """
        if not data:
            return
        # Ensure table is ready before storing data
        self._ensure_table_ready()

        try:
            with self._write_lock:
                data = self._with_embeddings(data)
                if len(data) <= self.batch_size:
                    self.table.add(data)
                    return
                # split the data into batches and store them
                for i in range(0, len(data), self.batch_size):
                    batch = data[i : i + self.batch_size]
                    self.table.add(batch)
        except Exception as e:
            raise DatusException(ErrorCode.STORAGE_SAVE_FAILED, message_args={"error_message": str(e)}) from e

    def store(self, data: List[Dict[str, Any]]):
        # Ensure table is ready before storing data
        self._ensure_table_ready()
        try:
            with self._write_lock:
                data = self._with_embeddings(data)
                self.table.add(data)
        except Exception as e:
            raise DatusException(ErrorCode.STORAGE_SAVE_FAILED, message_args={"error_message": str(e)}) from e

    def upsert_batch(self, data: List[Dict[str, Any]], on_column: str = "id"):
        """
        Upsert a batch of data using merge_insert (update if exists, insert if not).

        Args:
            data: List of dictionaries to upsert
            on_column: Column name to match for deduplication (default: "id")
        """
        if not data:
            return
        self._ensure_table_ready()

        # Deduplicate input data by on_column, keeping the last occurrence
        # This prevents duplicates when the same id appears multiple times in the input batch
        # Deduplicate input data by on_column, keeping the last occurrence
        if data and on_column in data[0]:
            seen = {}
            for row in data:
                key = row.get(on_column)
                seen[key] = row
            if len(seen) < len(data):
                logger.debug(f"Deduplicated {len(data) - len(seen)} records with duplicate '{on_column}' before upsert")
            data = list(seen.values())

        try:
            with self._write_lock:
                data = self._with_embeddings(data)
                if len(data) <= self.batch_size:
                    self.table.upsert(data, on_column)
                    return
                # Split the data into batches and upsert them
                for i in range(0, len(data), self.batch_size):
                    batch = data[i : i + self.batch_size]
                    self.table.upsert(batch, on_column)
        except Exception as e:
            raise DatusException(ErrorCode.STORAGE_SAVE_FAILED, message_args={"error_message": str(e)}) from e

    def search(
        self,
        query_txt: str,
        select_fields: Optional[List[str]] = None,
        top_n: Optional[int] = None,
        where: WhereExpr = None,
        reranker: Optional[Reranker] = None,
    ) -> pa.Table:
        # Ensure table is ready before searching
        self._ensure_table_ready()

        use_hybrid = self.backend.caps.hybrid_search and (
            reranker is not None or not self.backend.caps.native_embedding
        )
        if use_hybrid:
            search_result = self._search_hybrid(query_txt, reranker, select_fields, top_n, where)
        else:
            search_result = self._search_vector(query_txt, select_fields, top_n, where)
        if self.vector_column_name in search_result.column_names:
            search_result = search_result.drop([self.vector_column_name])
        return search_result

    def _search_hybrid(
        self,
        query_txt: str,
        reranker: Optional[Reranker],
        select_fields: Optional[List[str]] = None,
        top_n: Optional[int] = None,
        where: WhereExpr = None,
    ) -> pa.Table:
        try:
            if not top_n:
                top_n = self.table.count(where)
            query_vector = None
            if not self.backend.caps.native_embedding:
                query_vector = self._embed_query(query_txt)
            return self.table.search_hybrid(
                text=query_txt,
                top_n=top_n,
                where=where,
                select=select_fields,
                reranker=reranker,
                vector=query_vector,
            )
        except Exception as e:
            logger.warning(f"Failed to search hybrid: {str(e)}, use vector search instead")
            return self._search_vector(query_txt, select_fields, top_n, where)

    def _search_vector(
        self,
        query_txt: str,
        select_fields: Optional[List[str]] = None,
        top_n: Optional[int] = None,
        where: WhereExpr = None,
    ) -> pa.Table:
        try:
            if not top_n:
                top_n = self.table.count(where)
            if self.backend.caps.native_embedding:
                return self.table.search_text(query_txt, top_n=top_n, where=where, select=select_fields)
            vector = self._embed_query(query_txt)
            return self.table.search_vector(vector, top_n=top_n, where=where, select=select_fields)
        except Exception as e:
            raise DatusException(
                ErrorCode.STORAGE_SEARCH_FAILED,
                message_args={
                    "error_message": str(e),
                    "query": query_txt,
                    "where_clause": "(none)" if where is None else str(where),
                    "top_n": str(top_n or "all"),
                },
            ) from e

    def table_size(self) -> int:
        # Ensure table is ready before checking size
        self._ensure_table_ready()
        return self.table.count()

    def update(self, where: WhereExpr, update_values: Dict[str, Any], unique_filter: Optional[WhereExpr] = None):
        self._ensure_table_ready()
        if not update_values:
            return
        if where is None:
            return
        if unique_filter is not None:
            existing = self.table.count(unique_filter)
            if existing:
                raise DatusException(
                    ErrorCode.STORAGE_TABLE_OPERATION_FAILED,
                    message_args={
                        "operation": "update",
                        "table_name": self.table_name,
                        "error_message": "Conflicting rows already match unique_filter",
                    },
                )
        self.table.update(where=where, values=update_values)

    def delete(self, where: WhereExpr) -> None:
        self._ensure_table_ready()
        if where is None:
            return
        self.table.delete(where)

    def count(self, where: Optional[WhereExpr] = None) -> int:
        self._ensure_table_ready()
        return self.table.count(where)

    def _embed_query(self, text: str) -> List[float]:
        embeddings = self.model.model.generate_embeddings([text])
        return embeddings[0] if embeddings else []

    def _with_embeddings(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure vector column is populated for backends without native embedding."""
        if not rows or self.backend.caps.native_embedding:
            return rows

        to_embed: List[str] = []
        indices: List[int] = []
        for idx, row in enumerate(rows):
            if row.get(self.vector_column_name) is None:
                text_value = row.get(self.vector_source_name)
                if text_value is None:
                    text_value = ""
                to_embed.append(str(text_value))
                indices.append(idx)

        if not to_embed:
            return rows

        embeddings = self.model.model.generate_embeddings(to_embed)
        if len(embeddings) != len(to_embed):
            raise DatusException(
                ErrorCode.STORAGE_SAVE_FAILED,
                message_args={
                    "error_message": (f"Embedding count mismatch: expected {len(to_embed)} got {len(embeddings)}")
                },
            )

        for idx, vector in zip(indices, embeddings):
            rows[idx] = {**rows[idx], self.vector_column_name: vector}
        return rows
