# MongoDB vector store: local development rig

Everything needed to run LiteLLM against MongoDB Atlas Vector Search on one machine, with no cloud MongoDB account.

```
LiteLLM proxy (:4000, from source)  ->  litellm-mongodb sidecar (:8080, from source)  ->  Atlas Local (Docker, :27017)
LiteLLM dashboard (:3000, npm run dev)                                                     Postgres (Docker, :5432)
```

## One-time setup

1. Clone the sidecar next to this repo: `git clone https://github.com/BerriAI/litellm-mongodb ../litellm-mongodb && (cd ../litellm-mongodb && uv sync --frozen)`.
2. `make bootstrap` in this repo (uv sync, Prisma client, dashboard `npm install`).
3. Fill in `.env` at the repo root: `LITELLM_MASTER_KEY`, `DATABASE_URL`, `MONGODB_SIDECAR_API_KEY`, plus the embedding server the smoke script uses (`EMBEDDING_MODEL`, `EMBEDDING_API_BASE`, `EMBEDDING_API_KEY`). Two working options:
   - **LM Studio, fully local**: load `text-embedding-nomic-embed-text-v1.5` in LM Studio (768 dimensions) and use `openai/text-embedding-nomic-embed-text-v1.5` at `http://localhost:1234/v1`. The proxy config registers it as `nomic-embed-local`.
   - **Cloud LiteLLM router**: set `UPSTREAM_LITELLM_API_BASE` + `UPSTREAM_LITELLM_API_KEY` (a virtual key) and point `EMBEDDING_*` at it once the router has an embedding deployment (`text-embedding-3-small`).
4. Create `dev/mongodb/sidecar.env` (gitignored) with `MONGODB_CONNECTION_STRING`, the same `MONGODB_SIDECAR_API_KEY`, and `MONGODB_SIDECAR_ALLOW_DISCOVERY=true` (discovery is off by default in the sidecar; the rig opts in so the dashboard dropdowns work).

## Start the pieces

```bash
# MongoDB 8.2 with Atlas Search + Vector Search (mongot), and Postgres for the proxy
docker compose -f dev/mongodb/docker-compose.yml up -d
docker compose up -d db

# Sidecar from source with hot reload (loopback HTTP satisfies LiteLLM's HTTPS rule)
(cd ../litellm-mongodb && set -a && source ../litellm-mdb/dev/mongodb/sidecar.env && set +a && \
  uv run --no-sync uvicorn litellm_mongodb.app:create_app --factory --host 127.0.0.1 --port 8080 --reload)

# Proxy from source
uv run --no-sync litellm --config dev/mongodb/litellm_config.yaml --port 4000 --detailed_debug

# Dashboard dev server (optional, for UI work)
(cd ui/litellm-dashboard && npm run dev)
```

The Atlas Local connection string for the sidecar is
`mongodb://litellm:litellm-dev-password@127.0.0.1:27017/?directConnection=true`.

## Verify

- Sidecar health: `curl http://127.0.0.1:8080/health/readiness`
- Sidecar capabilities: `curl -H "Authorization: Bearer $MONGODB_SIDECAR_API_KEY" http://127.0.0.1:8080/v1/capabilities`
- SDK chain without a provider key (hash-based stand-in embedding): `uv run --no-sync python dev/mongodb/smoke_sdk.py`
- SDK chain with real embeddings once `UPSTREAM_LITELLM_API_KEY` is set: same command, it switches automatically.
- Sidecar integration suite against Atlas Local:
  `MONGODB_TEST_URI='mongodb://litellm:litellm-dev-password@127.0.0.1:27017/?directConnection=true' uv run --no-sync pytest tests/test_integration.py` (run inside `../litellm-mongodb`).

## Through the proxy

With the proxy on :4000 and `company-policies` registered from `dev/mongodb/litellm_config.yaml`:

```bash
# Ingest a markdown file (creates the index on first use)
curl -s http://localhost:4000/v1/rag/ingest \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -F file=@dev/mongodb/sample/travel.md \
  -F 'request={"ingest_options":{"vector_store":{"custom_llm_provider":"mongodb","vector_store_id":"policy_vector_index"}}}'

# Search
curl -s http://localhost:4000/v1/vector_stores/policy_vector_index/search \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{"query":"What is the travel policy?","max_num_results":3}'
```

## Tests to run before a PR

- Sidecar: `uv run --no-sync ruff check . && uv run --no-sync pytest -q` (in `../litellm-mongodb`).
- LiteLLM: `uv run --no-sync pytest tests/unit/llms/mongodb tests/test_litellm/vector_stores tests/test_litellm/proxy/vector_store_endpoints tests/test_litellm/proxy/rag_endpoints -q`
- LiteLLM gates: `make check` (ruff, basedpyright budget, strictness budgets on the diff).
