# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Dynamic plugin loader for storage backends."""

from __future__ import annotations

import importlib
from typing import Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


_PLUGIN_MODULES = {
    # Backend aliases -> plugin module
    "pgvector": "datus_postgresql",
    "postgresql": "datus_postgresql",
}


def try_load_storage_plugin(backend_name: str) -> Optional[object]:
    """Attempt to load a storage backend plugin by name."""
    plugin_module = _PLUGIN_MODULES.get(backend_name, f"datus_{backend_name}")
    try:
        module = importlib.import_module(plugin_module)
    except ImportError:
        logger.debug(f"No storage plugin found for backend '{backend_name}'")
        return None
    except Exception as exc:
        logger.warning(f"Failed to import storage plugin '{plugin_module}': {exc}")
        return None

    register = getattr(module, "register", None)
    if callable(register):
        try:
            register()
        except Exception as exc:
            logger.warning(f"Storage plugin '{plugin_module}' register() failed: {exc}")
            return None

    return module
