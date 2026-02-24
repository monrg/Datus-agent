# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Document Storage Module

Provides vector storage for documents using LanceDB with full-featured schema:
- Version tracking
- Navigation path (nav_path, group_name, hierarchy)
- Titles and keywords extraction
- Deduplication via chunk_id
"""

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pyarrow as pa

from datus.storage.base import BaseEmbeddingStore
from datus.storage.document.schemas import PlatformDocChunk
from datus.storage.embedding_models import EmbeddingModel, get_document_embedding_model
from datus.storage.lancedb_conditions import And, Condition, WhereExpr, eq, in_
from datus.utils.loggings import get_logger

# Validation pattern for version strings to prevent SQL injection
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9_\-. ]+$")

logger = get_logger(__name__)


def get_platform_doc_schema(embedding_dim: int = 384) -> pa.Schema:
    """Get PyArrow schema for platform documentation table."""
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("chunk_text", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("title", pa.string()),
            pa.field("titles", pa.list_(pa.string())),
            pa.field("nav_path", pa.list_(pa.string())),
            pa.field("group_name", pa.string()),
            pa.field("hierarchy", pa.string()),
            pa.field("version", pa.string()),
            pa.field("source_type", pa.string()),
            pa.field("source_url", pa.string()),
            pa.field("doc_path", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("language", pa.string()),
            pa.field("created_at", pa.string()),
            pa.field("updated_at", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=embedding_dim)),
        ]
    )


class DocumentStore(BaseEmbeddingStore):
    """Vector store for documentation with full-featured schema."""

    TABLE_NAME = "document"

    def __init__(
        self,
        db_path: str,
        embedding_model: EmbeddingModel,
        backend: Optional[Any] = None,
    ):
        schema = get_platform_doc_schema(embedding_model.dim_size)
        super().__init__(
            db_path=db_path,
            table_name=self.TABLE_NAME,
            embedding_model=embedding_model,
            vector_source_name="chunk_text",
            vector_column_name="vector",
            on_duplicate_columns="chunk_id",
            schema=schema,
            backend=backend,
        )

    def store_chunks(self, chunks: List[PlatformDocChunk]) -> int:
        """Store documentation chunks with automatic embedding."""
        if not chunks:
            return 0

        data = [chunk.to_dict() for chunk in chunks]
        self._ensure_table_ready()

        # Delete existing chunks first to avoid duplicate chunk_id rows.
        if self.table:
            try:
                row_count = self.table.count()
            except Exception:
                row_count = 0

            if row_count > 0:
                chunk_ids = [c.chunk_id for c in chunks]
                batch_size = 500
                for i in range(0, len(chunk_ids), batch_size):
                    batch_ids = chunk_ids[i : i + batch_size]
                    try:
                        self.table.delete(in_("chunk_id", batch_ids))
                    except Exception:
                        pass

        self.store_batch(data)
        logger.info(f"Stored {len(chunks)} chunks, version '{chunks[0].version}'")
        return len(chunks)

    def store_document(
        self,
        title: str,
        hierarchy: str,
        keywords: List[str],
        language: str,
        chunk_text: str,
    ) -> None:
        """Legacy compatibility API for storing a single chunk."""
        self.store(
            [
                {
                    "title": title,
                    "titles": [title] if title else [],
                    "nav_path": [],
                    "group_name": "",
                    "hierarchy": hierarchy,
                    "version": "legacy",
                    "source_type": "legacy",
                    "source_url": "",
                    "doc_path": "",
                    "keywords": keywords,
                    "language": language,
                    "chunk_text": chunk_text,
                    "chunk_index": 0,
                    "chunk_id": "",
                    "content_hash": "",
                }
            ]
        )

    def search_docs(
        self,
        query: str,
        version: Optional[str] = None,
        top_n: int = 10,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search documentation by semantic similarity."""
        conditions: List[Condition] = []

        if version:
            conditions.append(eq("version", version))

        where: WhereExpr = None
        if len(conditions) > 1:
            where = And(conditions)
        elif len(conditions) == 1:
            where = conditions[0]

        results = self.search(
            query_txt=query,
            top_n=top_n,
            where=where,
            select_fields=select_fields,
        )
        return results.to_pylist()

    def search_similar_documents(
        self,
        query_text: str,
        select_fields: Optional[List[str]] = None,
        top_n: int = 5,
    ) -> pa.Table:
        """Legacy compatibility API returning PyArrow table."""
        self._ensure_table_ready()
        return self._search_vector(query_text, select_fields=select_fields, top_n=top_n)

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all indexed versions with chunk counts."""
        self._ensure_table_ready()
        all_data = self._search_all(select_fields=["version"])

        version_counts: Dict[str, int] = {}
        for row in all_data.to_pylist():
            version = row["version"]
            version_counts[version] = version_counts.get(version, 0) + 1

        return [{"version": ver, "chunk_count": count} for ver, count in sorted(version_counts.items())]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this document store."""
        self._ensure_table_ready()
        all_data = self._search_all(select_fields=["version", "doc_path", "created_at"])
        rows = all_data.to_pylist()

        if not rows:
            return {"total_chunks": 0, "versions": [], "doc_count": 0}

        versions = set()
        doc_paths = set()
        latest_update = None
        for row in rows:
            versions.add(row["version"])
            doc_paths.add(row["doc_path"])
            created = row.get("created_at")
            if created and (latest_update is None or created > latest_update):
                latest_update = created

        return {
            "total_chunks": len(rows),
            "versions": sorted(versions),
            "doc_count": len(doc_paths),
            "latest_update": latest_update,
        }

    def get_stats_by_version(self, version: str) -> Dict[str, Any]:
        """Get statistics for a specific version."""
        self._ensure_table_ready()
        self._validate_identifier(version, "version")
        all_data = self._search_all(where=eq("version", version), select_fields=["doc_path"])
        rows = all_data.to_pylist()
        doc_paths = {row["doc_path"] for row in rows}
        return {"total_chunks": len(rows), "doc_count": len(doc_paths)}

    @staticmethod
    def _validate_identifier(value: str, name: str) -> None:
        if not _SAFE_IDENTIFIER_RE.match(value):
            raise ValueError(
                f"Invalid {name}: '{value}'. "
                f"Only alphanumeric characters, underscores, hyphens, dots, and spaces are allowed."
            )

    def delete_docs(self, version: Optional[str] = None) -> int:
        """Delete documentation chunks."""
        self._ensure_table_ready()

        count_before = self.table.count()
        if count_before == 0:
            logger.info(f"No chunks exists for version '{version or 'all'}'")
            return 0

        if version:
            self._validate_identifier(version, "version")
            self.table.delete(eq("version", version))
            try:
                # Best-effort for backends that expose compaction APIs.
                if hasattr(self.table, "compact_files"):
                    self.table.compact_files()
                if hasattr(self.table, "cleanup_old_versions"):
                    self.table.cleanup_old_versions()
            except Exception as e:
                logger.warning(f"Post-delete cleanup failed (non-fatal): {e}")
            count_after = self.table.count()
            deleted_count = count_before - count_after
        else:
            import shutil

            shutil.rmtree(self.db_path, ignore_errors=True)
            # Reinitialize lazily
            self._table_initialized = False
            self._ensure_table_ready()
            deleted_count = count_before

        logger.info(f"Deleted {deleted_count} chunks for version '{version or 'all'}'")
        return deleted_count

    def get_all_rows(
        self,
        where: WhereExpr = None,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get all rows matching a condition."""
        self._ensure_table_ready()
        results = self._search_all(where=where, select_fields=select_fields)
        return results.to_pylist()

    def create_indices(self):
        """Create optimized indices for the table."""
        self._ensure_table_ready()
        self.create_vector_index(metric="cosine")
        self.create_fts_index(field_names=["chunk_text", "title", "hierarchy"])
        logger.info(f"Created indices for table '{self.TABLE_NAME}'")


@lru_cache(maxsize=8)
def document_store(storage_path: str) -> DocumentStore:
    """Get a cached DocumentStore instance."""
    return DocumentStore(storage_path, get_document_embedding_model())
