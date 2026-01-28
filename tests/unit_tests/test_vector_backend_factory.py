import pytest

from datus.configuration.agent_config import DbConfig
from datus.storage.backends.vector import factory as vector_factory
from datus.storage.backends.vector.interfaces import BackendCapabilities


class DummyBackend:
    name = "pgvector"
    caps = BackendCapabilities(vector_search=True)

    def __init__(self, db_path: str, connection_string: str, schema: str, namespace: str):
        self.db_path = db_path
        self.connection_string = connection_string
        self.schema = schema
        self.namespace = namespace


class DummyConfig:
    def __init__(self, db_config: DbConfig, namespace: str = "ns"):
        self._db_config = db_config
        self._namespace = namespace

    def storage_backend_namespace(self, kind: str):
        return "pg_main"

    def resolve_storage_db_config(self, kind: str):
        return self._db_config

    @property
    def current_namespace(self):
        return self._namespace


def test_get_default_backend_prefers_pgvector(monkeypatch, tmp_path):
    db_config = DbConfig(
        type="postgresql",
        host="localhost",
        port="5432",
        username="user",
        password="pass",
        database="db",
        schema="custom",
    )

    monkeypatch.setattr(vector_factory, "get_vector_backend", lambda name: DummyBackend)

    backend = vector_factory.get_default_backend(str(tmp_path), agent_config=DummyConfig(db_config))

    assert isinstance(backend, DummyBackend)
    assert backend.db_path == str(tmp_path)
    assert backend.schema == "custom"
    assert backend.namespace == "ns"
    assert backend.connection_string.startswith("postgresql+psycopg2://")


@pytest.mark.parametrize("db_type", ["sqlite", "duckdb"])
def test_get_default_backend_falls_back_to_lance(db_type, tmp_path):
    db_config = DbConfig(type=db_type)
    backend = vector_factory.get_default_backend(str(tmp_path), agent_config=DummyConfig(db_config))
    assert backend.name == "lancedb"
