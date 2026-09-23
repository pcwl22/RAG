# Production image publishing

Model weights are **not** uploaded to GitHub, GHCR, or application images.
The release chain publishes only a small signed source descriptor and downloads
the weights from immutable Hugging Face revisions inside the target runtime.

## Approved model sources

The canonical descriptor is model-sources/model-manifest.json:

- BGE-M3:
  https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181
- BGE reranker v2 M3:
  https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e

Each entry binds the model ID, exact 40-character commit, human-readable source
and download URLs, local path, and deterministic downloaded-tree SHA-256. The
runtime refuses a mutable branch/tag, a different host, a changed descriptor,
or downloaded content with a different tree digest.

## Protected publication chain

Two protected workflows form the image-publication chain:

1. .github/workflows/publish-model-bundle.yml runs on GitHub's hosted Ubuntu
   runner and publishes approved-model-source, a scratch image containing
   only /models/model-manifest.json. It explicitly proves the artifact has no
   /models/bge-m3 tree, adds io.industrial-rag.model-weights=absent, and
   keyless-signs the digest.
2. .github/workflows/publish-production-images.yml verifies that signature,
   exact source commit, descriptor digest, and no-weights label. It mirrors the
   descriptor into the repository namespace and builds API, Worker, and
   Frontend images. API and Worker contain the descriptor at
   /app/model-source-manifest.json, not model weights.

The historical MODEL_BUNDLE_* environment names remain for release-artifact
compatibility. They now identify a descriptor-only OCI artifact.

## One-time GitHub configuration

1. Create the production-model-bundle environment, require reviewers, prevent
   self-review where available, and restrict deployments to main. No
   self-hosted runner, local weight mount, or proxy variable is required.
2. Create the production-image-publish environment with independent reviewers
   and the same main restriction.
3. Set MODEL_BUNDLE_CERTIFICATE_IDENTITY to the exact publisher identity:

   ~~~text
   https://github.com/OWNER/REPOSITORY/.github/workflows/publish-model-bundle.yml@refs/heads/main
   ~~~

   Set MODEL_BUNDLE_SOURCE_REPOSITORY to the exact repository URL and
   MODEL_BUNDLE_SOURCE_REVISION to the approved 40-character source commit.
4. Link the approved-model-source, model-bundle, api, worker, and frontend GHCR
   packages to the repository. Tags aid discovery only; every release consumes
   immutable @sha256: references.
5. In the protected production environment, set COSIGN_CERTIFICATE_IDENTITY to:

   ~~~text
   https://github.com/OWNER/REPOSITORY/.github/workflows/publish-production-images.yml@refs/heads/main
   ~~~

## Publishing

From main, start **Publish approved Hugging Face model source descriptor**.
There are no dispatch inputs and no local model files are read. Preserve the
industrial-rag-approved-model-source evidence artifact. Its model-source.env
contains the digest-pinned descriptor and exact signer/source values.

Then start **Publish production images** with:

- model_bundle_image: the approved-model-source@sha256:... reference
- model_manifest_sha256: the SHA-256 reported by the source publication

The production workflow scans, attests, signs, and verifies the descriptor plus
the three application images. The API and Worker image labels remain bound to
the exact descriptor digest and manifest digest.

## Runtime download

Kubernetes uses a model-source-init init container before each API/Worker pod
and once for the two-container Canary pod. It:

1. verifies the embedded descriptor byte digest;
2. downloads both exact revisions with huggingface_hub.snapshot_download;
3. removes downloader metadata from the content tree;
4. rejects symlinks and verifies each deterministic tree SHA-256;
5. atomically promotes the completed directory and writes a verified marker.

The init container is the only writer. Application containers mount the 16 GiB
model-cache volume read-only. The portable baseline uses emptyDir, so a new pod
downloads its own cache. A production cluster may replace that volume with a
pre-provisioned, access-controlled persistent cache, but must preserve the
single-writer initialization and read-only consumer mounts.

Outbound HTTPS to huggingface.co must be available during pod initialization.
No Hugging Face token is used for these public models. The Canary deadline is
longer to account for the first 6.88 GiB cold download, and the Hub file
download timeout is raised to 600 seconds for slow large-file transfers.

## Release hand-off

Archive release-images.json from the
industrial-rag-production-image-publication artifact and merge these immutable
values into the protected release snapshot:

~~~text
API_IMAGE=ghcr.io/owner/repository/api@sha256:...
WORKER_IMAGE=ghcr.io/owner/repository/worker@sha256:...
FRONTEND_IMAGE=ghcr.io/owner/repository/frontend@sha256:...
MODEL_BUNDLE_IMAGE=ghcr.io/owner/repository/model-bundle@sha256:...
MODEL_BUNDLE_DIGEST=sha256:...
MODEL_MANIFEST_SHA256=sha256:...
~~~

The model reference above is descriptor-only. Failed publication runs never
produce the release hand-off file.
