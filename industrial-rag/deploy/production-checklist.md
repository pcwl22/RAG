# Production deployment checklist

This checklist covers the controls that cannot be completed by application code alone. The local
Keycloak and single-node PostgreSQL/Redis services in the root Compose file are acceptance
infrastructure, not a highly available production topology.

## Identity and tenancy

- Use a managed OIDC provider or a production Keycloak cluster backed by an external HA database.
- Publish the issuer and JWKS endpoints over HTTPS and restrict accepted tokens to the `rag-api`
  audience and `RS256` algorithm.
- Map the immutable tenant UUID into the `tenant_id` claim. Map permissions to `viewer`, `editor`,
  and `admin`; do not grant `admin` as a default role.
- Register every tenant with `scripts/provision_tenant.py` before issuing tokens for it.
- Keep API-key service identities separate from human identities, scoped by
  `RAG_SERVICE_TENANT_ID` and least-privilege `RAG_SERVICE_ROLES`.

## Secrets

- Store `RAG_API_KEY`, both PostgreSQL passwords, Redis credentials, IdP administration credentials,
  and LLM provider keys in a cloud secret manager, Kubernetes Secret with encryption at rest, or an
  equivalent audited store. Do not deploy a committed `.env` file.
- Use different secrets for `POSTGRES_PASSWORD`, `POSTGRES_APP_PASSWORD`, `REDIS_PASSWORD`, and
  `RAG_API_KEY`. Rotate service secrets and IdP signing keys under a documented schedule.
- Inject `COMPOSE_REDIS_URL` as `rediss://` and configure trusted CA material when Redis crosses a
  host boundary. Never disable TLS verification in production.
- Run `python scripts/validate_production_env.py --env-file <rendered-env>` in the release job before
  deployment. The command reports variable names only and never prints secret values.

## Data resilience

- Run PostgreSQL with provider-managed high availability, point-in-time recovery, encrypted storage,
  and deletion protection. Redis is a cache/task transport and should use replication or a managed
  service; PostgreSQL remains the system of record.
- Schedule `scripts/backup_postgres.py` with a restricted backup role. Encrypt and copy archives to
  immutable off-host storage, then enforce retention at the storage layer.
- Test `scripts/restore_postgres.py` into an isolated database at least quarterly. Record recovery
  point and recovery time measurements. The script requires an exact `--confirm-database` value;
  `--clean` must be explicitly approved.
- Back up IdP configuration and signing-key recovery material separately from application data.

## Availability and operations

- Run at least two API replicas behind a TLS load balancer. Scale Celery workers separately and keep
  model memory requirements in the scheduler's resource requests and limits.
- Use `/health/live` for process restart and `/health/ready` for traffic admission. Alert on readiness,
  queue depth, PostgreSQL saturation, Redis availability, error rate, and retrieval latency.
- Centralize JSON logs with retention and access controls. Forward IdP login/admin events, secret
  access logs, deployment events, and database audit events to the same incident timeline.
- Apply network policies so only API/worker/migration identities reach PostgreSQL and Redis. The
  migration identity must not be used by runtime services.

## Release evidence

- Require backend tests, frontend tests/build/audit, Ruff, Compose rendering, evaluation asset
  validation, approved RAG quality gates, dependency scanning, and container scanning.
- Preserve reports as immutable CI artifacts and link the approved dataset hashes to the deployed
  image digest.
- Perform a canary rollout with rollback criteria. Verify one authorized, one cross-tenant-denied,
  one no-answer, one upload, and one streaming-chat flow before full traffic promotion.
