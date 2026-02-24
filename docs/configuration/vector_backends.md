# Vector Backend Extensions

This guide explains how vector storage backends work in Datus, how to enable PostgreSQL + pgvector, and how to extend support for other vector databases.

## Overview

Datus supports pluggable vector backends through adapter packages.

Current practical behavior in this codebase:

- Built-in default vector backend: LanceDB
- Plugin vector backend: `pgvector` (via `datus-postgresql`)
- Runtime table auto-creation for pgvector: not enabled (manual DDL required)

## Current Backend Status

| Backend | Vector Support | Notes |
|---|---|---|
| LanceDB | Yes | Built-in default |
| PostgreSQL (`pgvector`) | Yes | Provided by `datus-postgresql` plugin |
| MySQL / StarRocks / Snowflake / ClickZetta / Redshift | Not yet (vector backend) | Current adapters are relational connector focused |

## How Plugin Loading Works

Vector backend registration and loading are handled by:

- `datus/storage/backends/plugin_loader.py`
- `datus/storage/backends/vector/registry.py`
- `datus-db-adapters/datus-postgresql/datus_postgresql/__init__.py`

`pgvector` is registered by:

```python
register_vector_backend("pgvector", PgVectorBackend)
```

## PostgreSQL + pgvector Setup

### 1. Prerequisites

1. PostgreSQL instance has `vector` extension package available.
2. Your DB user can run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

If permission is denied (common on managed RDS), ask DBA/cloud admin to execute it once in the target database.

### 2. Install Required Packages

```bash
pip install -e Datus-agent \
  -e ../datus-db-adapters/datus-sqlalchemy \
  -e ../datus-db-adapters/datus-postgresql
```

### 3. Configure Namespace and Storage Backend

Example (`conf/agent.yml`):

```yaml
agent:
  namespace:
    gpdb_test:
      type: postgresql
      host: ${DB_HOST}
      port: 5432
      database: ${DB_NAME}
      username: ${DB_USER}
      password: ${DB_PASSWORD}
      schema: public
      sslmode: prefer
      enable_vector_search: true

  storage:
    backends:
      namespace: gpdb_test
```

### 4. Run DDL (Mandatory)

For pgvector in this project, create extension and tables manually:

1. Enable extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

2. Execute DDL:

- `../datus-db-adapters/datus-postgresql/datus_postgresql/ddl/all.sql`

Optional DDL rendering:

```bash
python - <<'PY'
from datus_postgresql.storage_ddl import render_all_ddl
print(render_all_ddl(schema="public", vector_dim=384))
PY
```

### 5. Bootstrap Knowledge Base

```bash
python -m datus.main bootstrap-kb \
  --components semantic_model \
  --namespace gpdb_test \
  --kb_update_strategy incremental \
  --config /absolute/path/to/conf/agent.yml \
  --semantic_yaml /absolute/path/to/semantic.yaml
```

### 6. Verification

Check extension:

```sql
SELECT extname, extversion FROM pg_extension WHERE extname='vector';
```

Check key tables:

```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema='public'
  AND table_name IN (
    'subject_nodes','tasks','feedback',
    'schema_metadata','schema_value','semantic_model',
    'metrics','reference_sql','ext_knowledge','document'
  )
ORDER BY table_name;
```

Check vector columns:

```sql
SELECT table_name, column_name, udt_name
FROM information_schema.columns
WHERE table_schema='public'
  AND column_name='vector'
ORDER BY table_name;
```

## Common Errors

### `permission denied to create extension "vector"`

Cause:
- User lacks extension-install privilege.

Fix:
- DBA/admin runs `CREATE EXTENSION IF NOT EXISTS vector;`.

### `type "vector" does not exist`

Cause:
- DDL was executed before extension was enabled.

Fix:
1. enable extension
2. rerun DDL

### `Table <name> does not exist`

Cause:
- DDL not executed in the configured database/schema.

Fix:
1. verify `database` in `agent.yml`
2. rerun `all.sql` in that database
3. verify via `information_schema.tables`

### `missing columns` during store readiness check

Cause:
- Existing table schema does not match current store schema.

Fix:
- align table structure with latest DDL (or recreate with migration plan).

## Extending Other Vector Backends

To add a new vector backend (for another database or external vector engine):

1. Create adapter package in `datus-db-adapters/datus-<backend>`.
2. Implement vector backend class following Datus vector backend interfaces.
3. Register in adapter `register()`:
   - `register_vector_backend("<backend_name>", BackendClass)`
4. Define DDL strategy:
   - manual/offline DDL (recommended first)
   - or runtime table creation if backend supports safe schema management
5. Ensure plugin loader can resolve package naming (`datus_<backend>` or alias mapping).
6. Add tests:
   - contract tests (add/upsert/search/filter/hybrid)
   - integration tests in real backend environment
7. Document installation, config, DDL, verification, and troubleshooting.

## PR Checklist

- [ ] Extension enablement steps are documented
- [ ] DDL path and execution order are documented
- [ ] Config examples include namespace + storage backend
- [ ] Verification SQL is provided
- [ ] Common failure modes and fixes are provided
- [ ] Backend registration path is documented
