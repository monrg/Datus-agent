# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List

from lancedb.embeddings.base import TextEmbeddingFunction

from datus.storage.ext_knowledge.store import ExtKnowledgeStore
from datus.storage.storage_manager import StorageManager, build_default_artifact_stores
from datus.utils.path_manager import DatusPathManager


class DummyEmbeddingFunction(TextEmbeddingFunction):
    name: str = "dummy"

    def __init__(self, dim: int = 3, **kwargs):
        super().__init__(**kwargs)
        self._dim_size = dim

    def ndims(self) -> int:
        return int(self._dim_size)

    def generate_embeddings(self, texts: List[str], *args, **kwargs) -> List[List[float]]:
        return [[float(len(text))] * self.ndims() for text in texts]


class DummyEmbeddingModel:
    def __init__(self, dim: int = 3):
        self.model_name = "dummy"
        self._dim_size = dim
        self.batch_size = 4
        self.device = "cpu"
        self.is_model_failed = False
        self.model_error_message = ""
        self._model = DummyEmbeddingFunction(dim=dim)

    @property
    def dim_size(self) -> int:
        return self._dim_size

    @property
    def model(self):
        return self._model


def test_ext_knowledge_roundtrip_with_lance_backend(tmp_path):
    store = ExtKnowledgeStore(db_path=str(tmp_path), embedding_model=DummyEmbeddingModel(dim=3))
    store.store_knowledge(
        subject_path=["Finance"],
        name="revenue",
        search_text="total revenue",
        explanation="Total revenue definition.",
    )

    results = store.search_all_knowledge(subject_path=["Finance"])
    assert len(results) == 1
    assert results[0]["name"] == "revenue"

    vector_results = store.search_knowledge(query_text="revenue", subject_path=["Finance"], top_n=1)
    assert vector_results


def test_storage_manager_resets_ext_knowledge_and_artifacts(tmp_path):
    storage_path = tmp_path / "rag"
    storage_path.mkdir()
    datus_home = tmp_path / "home"
    datus_home.mkdir()
    namespace = "demo"

    store = ExtKnowledgeStore(db_path=str(storage_path), embedding_model=DummyEmbeddingModel(dim=3))
    store.store_knowledge(
        subject_path=["Ops"],
        name="latency",
        search_text="p95 latency",
        explanation="Latency definition.",
    )

    path_manager = DatusPathManager(str(datus_home))
    artifact_dir = path_manager.ext_knowledge_path(namespace)
    artifact_file = artifact_dir / "note.txt"
    artifact_file.write_text("test")
    assert artifact_file.exists()

    manager = StorageManager(str(storage_path), artifact_stores=build_default_artifact_stores(str(datus_home)))
    manager.reset_component("ext_knowledge", "overwrite", namespace=namespace)

    assert not manager.backend.table_exists("ext_knowledge")
    assert not artifact_dir.exists()
