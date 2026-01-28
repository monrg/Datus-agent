# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol

from datus.storage.backends.factory import get_default_backend
from datus.storage.backends.interfaces import VectorBackend
from datus.utils.loggings import get_logger
from datus.utils.path_manager import DatusPathManager

logger = get_logger(__name__)


COMPONENT_TABLES: Dict[str, tuple[str, ...]] = {
    "metadata": ("schema_metadata", "schema_value"),
    "semantic_model": ("semantic_model",),
    "metrics": ("metrics",),
    "reference_sql": ("reference_sql",),
    "ext_knowledge": ("ext_knowledge",),
    "document": ("document",),
}


class ArtifactStore(Protocol):
    def reset(self, namespace: str) -> None:
        """Clear namespace-specific artifacts."""


@dataclass
class FileSystemArtifactStore:
    name: str
    path_fn: Callable[[str], Path]

    def reset(self, namespace: str) -> None:
        path = self.path_fn(namespace)
        try:
            if path.exists():
                shutil.rmtree(path)
                logger.info(f"Deleted {self.name} directory {path}")
        except Exception as exc:
            logger.warning(f"Failed to delete {self.name} directory {path}: {exc}")


class StorageManager:
    def __init__(
        self,
        storage_path: str,
        backend: Optional[VectorBackend] = None,
        artifact_stores: Optional[Dict[str, ArtifactStore]] = None,
    ):
        self.storage_path = storage_path
        self.backend = backend or get_default_backend(storage_path)
        self.artifact_stores = artifact_stores or {}

    def drop_component_tables(self, component: str) -> None:
        tables = COMPONENT_TABLES.get(component, ())
        for table_name in tables:
            # clear data
            self.backend.drop_table(table_name)

    def reset_component(self, component: str, strategy: str, namespace: Optional[str] = None) -> None:
        if strategy != "overwrite":
            return
        self.drop_component_tables(component)
        if namespace and component in self.artifact_stores:
            self.artifact_stores[component].reset(namespace)


def build_default_artifact_stores(datus_home: Optional[str] = None) -> Dict[str, ArtifactStore]:
    path_manager = DatusPathManager(datus_home)
    return {
        "metrics": FileSystemArtifactStore("semantic_models", path_manager.semantic_model_path),
        "semantic_model": FileSystemArtifactStore("semantic_models", path_manager.semantic_model_path),
        "reference_sql": FileSystemArtifactStore("sql_summaries", path_manager.sql_summary_path),
        "ext_knowledge": FileSystemArtifactStore("ext_knowledge", path_manager.ext_knowledge_path),
    }
