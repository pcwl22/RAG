# Production image publishing

`.github/workflows/publish-production-images.yml` is the only repository-owned
path that turns an approved model bundle and an exact source commit into
production API, Worker, and Frontend image references.

## One-time GitHub configuration

1. Create the `production-image-publish` environment. Require independent
   reviewers, prevent self-review, and restrict deployment branches to `main`.
   The workflow also rejects any source ref other than `refs/heads/main`.
2. Add the protected environment variable
   `MODEL_BUNDLE_CERTIFICATE_IDENTITY`. Its value must be the exact keyless
   certificate identity of the approved internal or external model-builder
   workflow, for example:

   ```text
   https://github.com/MODEL_OWNER/MODEL_REPOSITORY/.github/workflows/publish-model-bundle.yml@refs/tags/model-v1
   ```

   Regular expressions and wildcard identities are not accepted. Also set
   `MODEL_BUNDLE_SOURCE_REPOSITORY` to that builder repository's exact URL and
   `MODEL_BUNDLE_SOURCE_REVISION` to the approved 40-character commit SHA. The
   workflow requires the image's source and revision labels to match both.
3. Link the `api`, `worker`, `frontend`, and `model-bundle` GHCR packages to this
   repository so its scoped `GITHUB_TOKEN` can push signatures and attestations.
   Configure the immutable-tag policy where the registry supports it. Releases
   never consume tags; tags only aid discovery.
4. In the protected `production` release environment, set
   `COSIGN_CERTIFICATE_IDENTITY` to the exact publisher identity:

   ```text
   https://github.com/OWNER/REPOSITORY/.github/workflows/publish-production-images.yml@refs/heads/main
   ```

## Publishing

Start **Publish production images** from the `main` branch and supply both:

- `model_bundle_image`: an approved, pullable OCI image from an internal or
  external registry, pinned as `registry/path@sha256:<64 hex>`
- `model_manifest_sha256`: the `sha256:<64 hex>` digest of
  `/models/model-manifest.json` in that image

The repository intentionally does not invent or download model weights. Its
prerequisite is a separately approved, digest-pinned model bundle produced by the
exact protected identity above. The job verifies that upstream signature, the
exact source and revision labels, the manifest, and both model file trees. It then
copies the verified `/models` tree into a repository-bound production model
image. That derived image carries the application repository and exact
`github.sha` labels required by `release.yml`; the external source repository,
revision, signer, and digest remain recorded in `release-images.json` and every
application provenance predicate. The job then uses the repository-owned digest
to build with these production Dockerfiles:

An external/private registry must already be authenticated on the protected
runner before dispatch. The upstream registry namespace is not treated as an
identity signal; trust comes from the exact Cosign workflow identity plus the
source-repository, source-revision, manifest, and file-tree checks. Only the
verified copy is published into this repository's GHCR namespace and handed to
the application release.

- `industrial-rag/docker/api/Dockerfile.production`
- `industrial-rag/docker/worker/Dockerfile.production`
- `industrial-rag/docker/model-bundle/Dockerfile.production`
- `rag-frontend/Dockerfile.production`

The backend images embed the approved model files and carry labels for the exact
model-bundle digest, model-manifest digest, repository URL, and `github.sha`.
Every application image is pulled back by digest and its labels, numeric runtime
user, and embedded manifest are checked before it can be signed.

Critical or high Trivy findings fail the job. A failed scan leaves only unsigned
registry content, which the protected release rejects. After all scans pass, the
job keyless-signs the three application digests and the verified repository-owned
model mirror, attaches CycloneDX SBOM and SLSA v1 provenance attestations, and
verifies them again using the publisher's exact workflow identity.

## Release hand-off

Download the `industrial-rag-production-image-publication` artifact and archive
`release-images.json` with the approval record. Merge the six lines from
`release-images.env` into the protected Secret Manager snapshot used by
`release.yml`:

```text
API_IMAGE=ghcr.io/owner/repository/api@sha256:...
WORKER_IMAGE=ghcr.io/owner/repository/worker@sha256:...
FRONTEND_IMAGE=ghcr.io/owner/repository/frontend@sha256:...
MODEL_BUNDLE_IMAGE=ghcr.io/owner/repository/model-bundle@sha256:...
MODEL_BUNDLE_DIGEST=sha256:...
MODEL_MANIFEST_SHA256=sha256:...
```

That release-input artifact is uploaded only after signature and attestation
verification succeeds. A failed run may upload a separately named diagnostics
artifact, but it deliberately contains no `release-images.env` or
`release-images.json` hand-off file.

Do not copy the `git-<sha>` discovery tags into a release snapshot. The release
contract intentionally accepts only OCI digest references and independently
checks their signatures, source revision, source repository, and model binding.
