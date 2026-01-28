# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from datus.storage.backends.interfaces import VectorBackend
from datus.storage.backends.lance import LanceBackend


def get_default_backend(db_path: str) -> VectorBackend:
    # TODO: route to other backends based on config
    return LanceBackend(db_path)
