# Production deployment checklist

This checklist covers the controls that cannot be completed by application code alone. The local
Keycloak and single-node PostgreSQL/Redis services in the root Compose file are acceptance
infrastructure, not a highly available production topology.

## Runtime security gates

- Set `RAG_SECURE_MODE=true` for every shared or production API/worker process.
- Use `https://` for `OIDC_ISSUER` and `OIDC_JWKS_URL`, and `rediss://` for
  `REDIS_URL`/`COMPOSE_REDIS_URL`; the application rejects insecure URLs in
  secure mode.
- Keep `RAG_SECURE_MODE=false` only for the documented loopback/local Compose
  acceptance stack.

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
- Run `python scripts/validate_production_env.py --env-file <rendered-env> --release-snapshot` in the
  release job before deployment. This strict mode rejects normalized duplicate keys and ignores
  ambient runner overrides. The command reports variable names only and never prints secret values.
- Rotate a PostgreSQL runtime password through a new versioned runtime role. Keep the previous role
  valid until the new Deployment is ready, then revoke it in a separately audited cleanup step.

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

## Legacy PostgreSQL cutover

- Fresh databases are created directly with the final `(tenant_id, id)` primary key. Existing
  databases use append-only stage markers: add the nullable/defaulted tenant column and `NOT VALID`
  constraints, backfill with `FOR UPDATE SKIP LOCKED` in committed batches, validate the constraints,
  set `NOT NULL` from the validated check proof, and build the unique composite index concurrently.
  A killed or timed-out Job resumes from the verified marker and remaining rows.
- The production migration Job intentionally sets `POSTGRES_ALLOW_LEGACY_PK_CUTOVER=false`. On an
  id-only legacy database it completes the online preparation and then stops before changing the
  primary key. This is an expected release blocker, not permission to bypass the gate.
- Schedule the one-time cutover only after a tested backup and restore, drain every API/worker and
  other writer that may still use `ON CONFLICT (id)`, and run the migration with
  `POSTGRES_ALLOW_LEGACY_PK_CUTOVER=true`. Keep traffic drained until the whole migration and schema
  health check succeed, then deploy only tenant-aware Pods. The cutover reuses the already-valid
  unique index, sets a 1–60 second bounded lock timeout (5 seconds by default), and records completion
  in the same short transaction. A lock timeout rolls back the primary-key change and is safe to retry.
- Do not describe this one-time boundary as strictly zero-downtime: `DROP/ADD PRIMARY KEY USING INDEX`
  still needs a brief `ACCESS EXCLUSIVE` catalog lock, and forced RLS is enabled before tenant-aware
  traffic resumes. Tune `POSTGRES_MIGRATION_BATCH_SIZE` between 1 and 10,000 after load testing; do not
  increase the cutover lock timeout to wait through uncontrolled production traffic.

## Availability and operations

- Run at least two API replicas behind a TLS load balancer. Scale Celery workers separately and keep
  model memory requirements in the scheduler's resource requests and limits.
- Use `/health/live` for process restart and `/health/ready` for traffic admission. Alert on readiness,
  queue depth, PostgreSQL saturation, Redis availability, error rate, and retrieval latency.
- Centralize JSON logs with retention and access controls. Forward IdP login/admin events, secret
  access logs, deployment events, and database audit events to the same incident timeline.
- Apply network policies so only API/worker/migration identities reach PostgreSQL and Redis. The
  migration identity must not be used by runtime services.
- Set `INGRESS_PROXY_CIDR` to the narrow ingress-controller Pod CIDR. Nginx trusts forwarded client
  addresses only from that network; never use `0.0.0.0/0` or `::/0`.

## Release evidence

- For private repositories, enable GitHub Advanced Security and set the repository variable
  `GITHUB_ADVANCED_SECURITY_ENABLED=true` before treating CodeQL and Dependency Review as provider-native
  required checks. Until that license is available, the mandatory portable Ruff, mypy, pip/npm audit,
  Trivy, lock validation, and SBOM gates remain active and their artifacts must be retained.
- Require backend tests, frontend tests/build/audit, Ruff, Compose rendering, evaluation asset
  validation, approved RAG quality gates, dependency scanning, and container scanning.
- Resolve and review `requirements-runtime.lock.txt`, `requirements-gpu.lock.txt`,
  `requirements-torch-cpu.lock.txt`, `requirements-torch-cu126.lock.txt`, and
  `requirements-evaluation.lock.txt`; each installable artifact is version- and SHA-256-pinned.
  Torch is never resolved from the public PyPI default during a container build: API images use
  the digest-pinned CUDA base and workers use the CPU carrier lock. CI audits every deployed lock,
  validates the carrier/application split, and builds both production paths.
- Preserve reports as immutable CI artifacts and link the approved dataset hashes to the deployed
  image digest.
- Generate a fresh canonical UUIDv4 `RELEASE_ID` for every protected release. Never reuse an ID
  with changed inputs. Public resource suffixes/annotations derive only from that non-secret ID;
  ConfigMaps and Secrets are immutable and reject content changes under a reused ID.
- Manage Namespace, ServiceAccount, Service, Ingress, PDB, and NetworkPolicy as a separately
  reviewed platform baseline. The application release must only diff these resources and fail on
  missing/drifted state; platform bootstrap/change automation owns their mutation and rollback.
  For the first cluster, run the documented `production-platform` protected operation with a
  platform-only service identity, server-side dry-run, approved diff, stable
  `industrial-rag-platform` field manager, readback, and zero-diff proof before dispatching the
  application release. Never grant the application release identity permission to bootstrap them.
- Build and scan every production Dockerfile in CI. Require the exact protected Cosign certificate
  identity, source repository label, release commit label, OCI digest, and model-manifest digest
  before rendering manifests.
- Treat the approved model bundle as an external prerequisite. This repository does not contain a
  `publish-model-bundle` workflow and the application-image workflow must not imply that it trained,
  downloaded, or independently approved those weights. Before dispatching
  `.github/workflows/publish-production-images.yml`, place the approved bundle under the protected
  `ghcr.io/<owner>/<repository>/...@sha256:<digest>` namespace and obtain the exact SHA-256 of
  `/models/model-manifest.json`.
- Protect the `production-image-publish` environment with required reviewers and configure
  `MODEL_BUNDLE_CERTIFICATE_IDENTITY` as the exact keyless signer URI (an internal or external
  GitHub workflow is allowed), `MODEL_BUNDLE_SOURCE_REPOSITORY` as that workflow's exact GitHub
  repository URL, and `MODEL_BUNDLE_SOURCE_REVISION` as the full 40-character source commit. The
  workflow rejects mutable/cross-namespace images, verifies that exact signer and source labels,
  binds the signing certificate's workflow-SHA claim, validates every file hash in the manifest,
  then builds the API, worker, frontend, and repository-bound model images only from `main` at the
  exact dispatch SHA.
- Preserve the successful `industrial-rag-production-image-publication` artifact. Its
  `release-images.env` contains only digest-pinned release inputs; its record, Trivy reports,
  CycloneDX SBOMs, signatures, and SLSA v1 attestations bind all four images to the protected
  publication workflow. Set the release environment's `COSIGN_CERTIFICATE_IDENTITY` to the exact
  `https://github.com/<owner>/<repository>/.github/workflows/publish-production-images.yml@refs/heads/main`
  identity and copy the reviewed digest values into the protected release snapshot. Publication
  does not automatically authorize deployment or replace the protected RAG quality gates.
- Perform a canary rollout with rollback criteria. Verify one authorized, one cross-tenant-denied,
  one no-answer, one upload, and one source-bound SSE answer before full traffic promotion.
- Provision two dedicated OIDC client-credentials identities for the protected canary. The primary
  identity needs `admin` in one registered tenant so the fixture can be uploaded and deleted; the secondary needs `viewer` in a different
  registered tenant. Store both secrets and the approved base64 UTF-8 fixture only in the protected
  release snapshot. A missing credential, same-tenant claim, failed upload, leaked cross-tenant
  result, changed refusal contract, or malformed/error SSE event must block promotion.
  The tenant-scoped successful task result must return its `document_id`; deletion plus a verified
  `404` repeat-delete is part of the gate, and cleanup failure blocks promotion.
- Keep release-suffixed ConfigMaps and Secrets referenced by the current and retained rollback
  ReplicaSets. Do not prune them in the release workflow. Garbage-collect them only in a separate
  audited job after proving that no Deployment or ReplicaSet references each candidate resource.
- Require database migrations to remain backward-compatible with the currently running application:
  migration and canary failures deliberately leave the stable Deployments untouched, while the new
  versioned resources remain inert and available for forensic evidence or a controlled retry.
