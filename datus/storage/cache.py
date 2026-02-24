# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, Optional

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import SubAgentConfig
from datus.storage import BaseEmbeddingStore
from datus.storage.backends.vector.factory import get_default_backend
from datus.storage.document import DocumentStore
from datus.storage.embedding_models import EmbeddingModel, get_embedding_model
from datus.storage.ext_knowledge import ExtKnowledgeStore
from datus.storage.metric.store import MetricStorage
from datus.storage.reference_sql import ReferenceSqlStorage
from datus.storage.schema_metadata import SchemaStorage
from datus.storage.schema_metadata.store import SchemaValueStorage
from datus.storage.semantic_model.store import SemanticModelStorage
from datus.storage.subject_tree.store import SubjectTreeStore
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=12)
def _cached_storage[
    T: BaseEmbeddingStore
](factory: Callable[[str, EmbeddingModel], T], path: str, model_name: str) -> T:
    return factory(path, get_embedding_model(model_name))


class StorageCacheHolder[T: BaseEmbeddingStore]:
    def __init__(
        self,
        storage_factory: Callable[[str, EmbeddingModel], T],
        agent_config: AgentConfig,
        embedding_model_conf_name: str,
        check_scope_attr: str,
        extra_kwargs_provider: Optional[Callable[[Optional[str]], dict]] = None,
    ):
        self.storage_factory = storage_factory
        self.embedding_model_conf_name = embedding_model_conf_name
        self._agent_config = agent_config
        self.check_scope_attr = check_scope_attr
        self._extra_kwargs_provider = extra_kwargs_provider
        self._instances: Dict[tuple[str, str, str], T] = {}

    def storage_instance(self, sub_agent_name: Optional[str] = None) -> T:
        storage_path = self._agent_config.rag_storage_path()
        if sub_agent_name and (config := self._agent_config.sub_agent_config(sub_agent_name)):
            sub_agent_config = SubAgentConfig.model_validate(config)
            if sub_agent_config.has_scoped_context_by(self.check_scope_attr) and getattr(
                sub_agent_config.scoped_context, self.check_scope_attr
            ):
                storage_path = self._agent_config.sub_agent_storage_path(sub_agent_name)
                logger.debug(f"Sub-agent {sub_agent_name} uses scoped storage path {storage_path}")

        extra_kwargs = {}
        if self._extra_kwargs_provider:
            extra_kwargs.update(self._extra_kwargs_provider(sub_agent_name))
        backend = get_default_backend(storage_path, agent_config=self._agent_config)
        extra_kwargs.setdefault("backend", backend)

        return self._get_or_create(storage_path, extra_kwargs)

    def _get_or_create(self, storage_path: str, extra_kwargs: dict) -> T:
        backend_name = getattr(extra_kwargs.get("backend"), "name", "default")
        cache_key = (storage_path, self.embedding_model_conf_name, backend_name)
        if cache_key in self._instances:
            return self._instances[cache_key]
        instance = self.storage_factory(
            storage_path,
            get_embedding_model(self.embedding_model_conf_name),
            **extra_kwargs,
        )
        self._instances[cache_key] = instance
        return instance

    def invalidate_path(self, storage_path: str) -> None:
        keys = [key for key in self._instances if key[0] == storage_path]
        for key in keys:
            self._instances.pop(key, None)

    def clear(self) -> None:
        self._instances.clear()


class StorageCache:
    """Cache access to global and sub-agent storage instances."""

    def __init__(self, agent_config: AgentConfig):
        self._agent_config = agent_config
        self._schema_holder = StorageCacheHolder(SchemaStorage, agent_config, "database", "tables")
        self._sample_data_holder = StorageCacheHolder(SchemaValueStorage, agent_config, "database", "tables")
        self._semantic_holder = StorageCacheHolder(
            SemanticModelStorage, agent_config, "semantic_model", "semantic_models"
        )
        self._document_holder = StorageCacheHolder(DocumentStore, agent_config, "document", "")
        self._subject_tree_store: Optional[SubjectTreeStore] = None

        def subject_tree_kwargs(_: Optional[str]) -> dict:
            return {"subject_tree_store": self.subject_tree_store()}

        self._metric_holder = StorageCacheHolder(
            MetricStorage, agent_config, "metric", "metrics", extra_kwargs_provider=subject_tree_kwargs
        )
        self._reference_sql_holder = StorageCacheHolder(
            ReferenceSqlStorage, agent_config, "reference_sql", "sqls", extra_kwargs_provider=subject_tree_kwargs
        )
        self._ext_knowledge_holder = StorageCacheHolder(
            ExtKnowledgeStore, agent_config, "ext_knowledge", "ext_knowledge", extra_kwargs_provider=subject_tree_kwargs
        )

    def schema_storage(self, sub_agent_name: Optional[str] = None) -> SchemaStorage:
        return self._schema_holder.storage_instance(sub_agent_name)

    def schema_rag(self, sub_agent_name: Optional[str] = None) -> SchemaStorage:
        return self.schema_storage(sub_agent_name)

    def schema_value_storage(self, sub_agent_name: Optional[str] = None) -> SchemaValueStorage:
        return self._sample_data_holder.storage_instance(sub_agent_name)

    def schema_value_rag(self, sub_agent_name: Optional[str] = None) -> SchemaValueStorage:
        return self.schema_value_storage(sub_agent_name)

    def metric_storage(self, sub_agent_name: Optional[str] = None) -> MetricStorage:
        return self._metric_holder.storage_instance(sub_agent_name)

    def metrics_rag(self, sub_agent_name: Optional[str] = None) -> MetricStorage:
        return self.metric_storage(sub_agent_name)

    def semantic_storage(self, sub_agent_name: Optional[str] = None) -> SemanticModelStorage:
        return self._semantic_holder.storage_instance(sub_agent_name)

    def semantic_rag(self, sub_agent_name: Optional[str] = None) -> SemanticModelStorage:
        return self.semantic_storage(sub_agent_name)

    def reference_sql_storage(self, sub_agent_name: Optional[str] = None) -> ReferenceSqlStorage:
        return self._reference_sql_holder.storage_instance(sub_agent_name)

    def reference_sql_rag(self, sub_agent_name: Optional[str] = None) -> ReferenceSqlStorage:
        return self.reference_sql_storage(sub_agent_name)

    def document_storage(self, sub_agent_name: Optional[str] = None) -> DocumentStore:
        return self._document_holder.storage_instance(sub_agent_name)

    def document_rag(self, sub_agent_name: Optional[str] = None) -> DocumentStore:
        return self.document_storage(sub_agent_name)

    def ext_knowledge_storage(self, sub_agent_name: Optional[str] = None) -> ExtKnowledgeStore:
        return self._ext_knowledge_holder.storage_instance(sub_agent_name)

    def ext_knowledge_rag(self, sub_agent_name: Optional[str] = None) -> ExtKnowledgeStore:
        return self.ext_knowledge_storage(sub_agent_name)

    def subject_tree_store(self) -> SubjectTreeStore:
        if self._subject_tree_store is None:
            backend_args = self._agent_config.relational_backend_options()
            self._subject_tree_store = SubjectTreeStore(self._agent_config.rag_storage_path(), **backend_args)
        return self._subject_tree_store

    def invalidate(self, sub_agent_name: Optional[str] = None) -> None:
        holders = (
            self._schema_holder,
            self._sample_data_holder,
            self._semantic_holder,
            self._document_holder,
            self._metric_holder,
            self._reference_sql_holder,
            self._ext_knowledge_holder,
        )
        if sub_agent_name:
            storage_path = self._agent_config.sub_agent_storage_path(sub_agent_name)
            for holder in holders:
                holder.invalidate_path(storage_path)
            return

        for holder in holders:
            holder.clear()
        self._subject_tree_store = None


_CACHE_INSTANCE = None


def get_storage_cache_instance(agent_config: AgentConfig) -> StorageCache:
    global _CACHE_INSTANCE
    if _CACHE_INSTANCE is None:
        _CACHE_INSTANCE = StorageCache(agent_config)
    return _CACHE_INSTANCE


def clear_cache():
    _cached_storage.cache_clear()
    global _CACHE_INSTANCE
    _CACHE_INSTANCE = None
