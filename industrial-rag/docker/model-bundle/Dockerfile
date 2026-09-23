# syntax=docker/dockerfile:1.7
# Descriptor-only image. It intentionally contains no model weights; runtime
# init containers download the pinned Hugging Face revisions in this manifest.
FROM scratch
ARG SOURCE_REVISION
ARG SOURCE_REPOSITORY
ARG MODEL_MANIFEST_SHA256
LABEL org.opencontainers.image.revision=$SOURCE_REVISION \
      org.opencontainers.image.source=$SOURCE_REPOSITORY \
      org.opencontainers.image.model-manifest-sha256=$MODEL_MANIFEST_SHA256 \
      io.industrial-rag.model-weights=absent
COPY model-manifest.json /models/model-manifest.json
