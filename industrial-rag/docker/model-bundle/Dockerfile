# syntax=docker/dockerfile:1.7
# Build context is supplied by the protected release environment and must
# contain models/bge-m3, models/bge-reranker-v2-m3 and model-manifest.json.
FROM scratch
ARG SOURCE_REVISION
ARG SOURCE_REPOSITORY
ARG MODEL_MANIFEST_SHA256
LABEL org.opencontainers.image.revision=$SOURCE_REVISION \
      org.opencontainers.image.source=$SOURCE_REPOSITORY \
      org.opencontainers.image.model-manifest-sha256=$MODEL_MANIFEST_SHA256
COPY models/bge-m3 /models/bge-m3
COPY models/bge-reranker-v2-m3 /models/bge-reranker-v2-m3
COPY model-manifest.json /models/model-manifest.json
