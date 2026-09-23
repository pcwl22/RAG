# Production Kustomize contract

`base/` and `production/` contain only Kubernetes primitives. They deliberately
do not deploy PostgreSQL, Redis, or Keycloak. Those services must be supplied as
HA/managed endpoints through the external Secret Manager contract.

Before applying the production overlay, the release job must:

1. Render a Secret Manager snapshot into immutable-by-name Secrets named
   `rag-runtime-<release-suffix>`, `rag-migration-<release-suffix>`, and
   `rag-canary-credentials-<release-suffix>` (the snapshot is never committed, printed, or
   uploaded as an artifact). The same suffix is applied to runtime, frontend,
   and canary-script ConfigMaps.
2. Provision the TLS Secret `rag-ingress-tls` and replace the example OIDC,
   ingress-host, image, and model-bundle values with approved values.
3. Replace every zero image digest, `MODEL_BUNDLE_DIGEST`, and
   `MODEL_MANIFEST_SHA256` placeholder from the same validated snapshot.
4. Snapshot the current Deployment UID/generation/revision and prove all of its
   ConfigMap/Secret references still exist before changing any cluster state.
5. Run the migration Job, wait for the functional `rag-canary` to succeed, then
   roll the Deployments and verify rollback criteria.

The checked-in migration Job refuses the one-time legacy id-only primary-key
cutover by default. Its first run may safely finish the batched/validated/concurrent
preparation and then fail at that gate. Follow the traffic-drain procedure in
`../production-checklist.md`, explicitly set
`POSTGRES_ALLOW_LEGACY_PK_CUTOVER=true` only for the approved cutover Job, and
keep writers drained until schema health succeeds. Fresh databases and databases
already using `(tenant_id, id)` do not require this override.

Every protected snapshot must contain a newly generated, canonical lowercase
UUIDv4 `RELEASE_ID`. Resource names and the public Pod annotation are derived
only from a domain-separated SHA-256 of this non-secret ID; no digest of secret
values is exposed. Reusing a `RELEASE_ID` with changed content fails because
all release ConfigMaps and Secrets set `immutable: true`.

The protected GitHub Actions environment is named `production` and expects the
Secret Manager contract as the masked `RAG_PRODUCTION_ENV_B64` environment
secret. It is a base64-encoded dotenv snapshot containing the values checked by
`scripts/validate_production_env.py`, including the three immutable application
image references, `MODEL_BUNDLE_IMAGE`, its OCI `MODEL_BUNDLE_DIGEST`, the
byte-level `MODEL_MANIFEST_SHA256`, and the managed service endpoints. The
release snapshot is parsed once with duplicate-key rejection and is authoritative
over ambient runner variables. The release runner must have the `rag-production` label,
network access to those endpoints, and read-only authentication for the
configured container registry (pre-provisioned on the runner or injected by a
protected runner service); no secret snapshot is committed or uploaded as an
artifact.

The runtime Secret keys are (the concrete Secret name carries the release suffix):

`rag-runtime`: `RAG_API_KEY`, `RAG_METRICS_TOKEN`, `POSTGRES_HOST`,
`POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER` (mapped from
`POSTGRES_RUNTIME_USER`), `POSTGRES_PASSWORD` (mapped from
`POSTGRES_APP_PASSWORD`), `RAG_SERVICE_TENANT_ID`, `RAG_SERVICE_ROLES`,
`REDIS_URL`, `DEEPSEEK_API_KEY`, `DEEPSEEK_API_URL`, `DEEPSEEK_MODEL`,
`OIDC_ISSUER`, `OIDC_AUDIENCE`, `OIDC_JWKS_URL`, `S3_ENDPOINT_URL`, `S3_BUCKET`,
`S3_REGION`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`. Frontend OIDC values,
S3 addressing/failure-prefix values, and the trusted ingress CIDR are rendered
into ConfigMaps rather than copied into the backend Secret.

`rag-migration`: `POSTGRES_ADMIN_USER`, `POSTGRES_ADMIN_PASSWORD`,
`POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_RUNTIME_USER`, and
`POSTGRES_RUNTIME_PASSWORD`. The migration Job maps the admin credentials to
`POSTGRES_USER`/`POSTGRES_PASSWORD`; API and Worker receive only the runtime
credentials.

`rag-canary-credentials`: `CANARY_OIDC_TOKEN_URL`, `CANARY_OIDC_CLIENT_AUTH_METHOD`, two
distinct client-credential identities (`CANARY_PRIMARY_*` and
`CANARY_SECONDARY_*`), `CANARY_UPLOAD_CONTENT_B64`,
`CANARY_NO_ANSWER_QUERY`, and `CANARY_NO_ANSWER_EXPECTED_TEXT`. Optional
`CANARY_OIDC_SCOPE` and `CANARY_OIDC_AUDIENCE` values are forwarded to the
token request. The primary client must receive a registered tenant claim plus
`admin` access (upload plus mandatory fixture deletion); the secondary client must receive a different registered
tenant claim plus at least `viewer` access. The token endpoint must share the
OIDC issuer origin, and the uploaded fixture must be 80 bytes through 256 KiB
of base64-encoded UTF-8. Missing, duplicate, insecure, placeholder, or
same-client values stop the release before Kubernetes is modified.

The canary starts the candidate API image locally and verifies readiness,
client-credentials authentication, a real asynchronous text upload, primary
tenant retrieval, direct task-ID denial and empty retrieval from the secondary
tenant, the deterministic no-answer contract, and a complete SSE answer with
bound sources. It then deletes the tenant-scoped `document_id` returned by the
successful task and proves a repeated delete is `404`; cleanup failure blocks
the release. The same unpredictable random marker is included in the filename,
content, and `source_id`, so replacement semantics cannot overwrite a
legitimate tenant document that happens to share a fixed public metadata value.
The canary has no API-key fallback and emits no credential values.

Release ConfigMaps and Secrets are append-only during deployment. The workflow
does not use `kubectl --prune` and does not delete versioned release resources.
Before migration or canary work it snapshots the old Deployment references and
checks every dependency exists; before the stable apply it rechecks the old
UID/generation. If the stable rollout fails, rollback is allowed only while the
candidate UID/generation is unchanged and the exact previous resource refs
still exist. Retain resources referenced by any live ReplicaSet; garbage
collection must be a separate audited operation that first proves no
Deployment or retained ReplicaSet references the candidate names.

Namespace, ServiceAccount, Service, Ingress, PodDisruptionBudget, and
NetworkPolicy objects are a separately governed platform baseline. The
application release performs a server-side `kubectl diff` and fails on a
missing object or any drift; it never applies these shared traffic/policy
resources. The platform bootstrap/change pipeline must render the same approved
overlay and apply `platform-baseline.yaml` before the first application release.
Use server-side apply with the stable `industrial-rag-platform` field manager;
the release diff uses that same manager to avoid ownership-dependent results.
This ownership split means migration, canary, or stable failures cannot leave a
partially updated Ingress, Service, PDB, or NetworkPolicy behind.

Each workload uses one exact, tokenless ServiceAccount: `rag-api`, `rag-worker`,
and `rag-frontend` serve the corresponding Deployments, `rag-migrate` is reserved
for the PostgreSQL migration Job, and `rag-canary` is reserved for the functional
canary Job. Both ServiceAccounts and Pods disable automatic API-token mounting;
using the namespace `default` identity is a validation failure. Pod seccomp is
`RuntimeDefault`, and every declared container must explicitly run as non-root,
disable privilege escalation, and drop all Linux capabilities.

### First-cluster platform bootstrap (separate protected operation)

The first application release is intentionally unable to bootstrap a cluster.
Run the following as a separate reviewed automation job protected by a distinct
`production-platform` environment and concurrency group. Its identity must have
write access only to the six platform kinds above; the application release
identity must not have that permission.

1. Check out the reviewed commit and materialize the same protected release
   snapshot into a mode-`0600` temporary file. Validate it with
   `validate_production_env.py --release-snapshot` and render the production
   overlay with `render_production_manifests.py`.
2. Select **only** `Namespace`, `ServiceAccount`, `Service`, `Ingress`,
   `PodDisruptionBudget`, and `NetworkPolicy` using
   `select_kubernetes_manifest.py`. Review and retain the resulting
   `platform-baseline.yaml` as the change artifact.
3. Provision `rag-ingress-tls` through the cluster's certificate/secret
   controller, then run a server-side dry-run and human-approved diff.
4. Apply exactly that file with
   `kubectl apply --server-side --field-manager=industrial-rag-platform -f platform-baseline.yaml`.
   Read every object back and require a zero `kubectl diff` before marking the
   bootstrap complete.
5. Remove the protected snapshot. Roll back a platform change only from its
   separately reviewed prior baseline; never invoke the application Deployment
   rollback as a substitute.

Subsequent platform changes follow the same operation. The protected application
release remains diff-only for these objects on both bootstrap and upgrade paths.

The release workflow verifies keyless signatures with Cosign against the exact
protected `COSIGN_CERTIFICATE_IDENTITY`, binds application image labels to the
release commit, and binds the model image to both its OCI and manifest digests. It produces
Trivy SARIF, dependency-audit CycloneDX, source SBOM, per-image SBOM, quality,
retrieval, holdout, manifest, and signature artifacts. A missing or failed
artifact blocks the release.

The zero-digest/example values are intentional release blockers, not deployable
defaults. This prevents a local or unapproved image from being promoted by
accident.

The protected model-source build creates a descriptor without reading model
weights. Regenerate and validate it with:

```bash
cd industrial-rag
python scripts/build_model_bundle_manifest.py \
  --output /tmp/model-manifest.json
python scripts/validate_model_bundle_manifest.py /tmp/model-manifest.json
cmp /tmp/model-manifest.json model-sources/model-manifest.json
```

`MODEL_BUNDLE_DIGEST` is the descriptor-only OCI image digest;
`MODEL_MANIFEST_SHA256` is the SHA-256 of the exact source manifest. They are
deliberately separate release inputs. The `model-source-init` containers verify
the latter, download the two immutable Hugging Face commits, verify their tree
digests, and populate `/app/models`. API and Worker mount that cache read-only
and verify the descriptor again before opening PostgreSQL or loading a model.
