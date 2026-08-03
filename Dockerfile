# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.27@sha256:4d01caf3b22dfd11003455e2e68153da08c4ee1fa54fdbd166c6282d22693419

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/bodycomposition \
    UV_CACHE_DIR=/var/cache/uv
WORKDIR /src

# Resolve the large, frozen runtime layer before copying frequently changing code.
COPY pyproject.toml uv.lock README.md LICENSE THIRD_PARTY_NOTICES.md CITATION.cff CHANGELOG.md SECURITY.md CONTRIBUTING.md MANIFEST.in ./
COPY scripts/sanitize_container_environment.py ./scripts/sanitize_container_environment.py
RUN --mount=type=cache,target=/var/cache/uv \
    uv sync --frozen --no-dev --no-install-project --no-editable

COPY BodyComposition ./BodyComposition
RUN --mount=type=cache,target=/var/cache/uv \
    uv sync --frozen --no-dev --no-editable --compile-bytecode \
    --reinstall-package BodyComposition

# Some frozen dependencies bundle unused image-quality checkpoints and test
# scans. Remove only the known assets, then reject any unexpected model or
# medical-image file. SPINEPS 2.0.0 creates its package-local fallback
# directory at import even when explicit model paths are supplied.
RUN mkdir -p /opt/bodycomposition/lib/python3.11/site-packages/spineps/models \
    && /opt/bodycomposition/bin/python scripts/sanitize_container_environment.py /opt/bodycomposition

FROM ${PYTHON_IMAGE} AS runtime
ARG VERSION=1.0.0rc1
ARG GIT_SHA=unknown
ARG LOCK_SHA256=unknown
ARG SOURCE_SHA256=unknown
ARG SOURCE_DATE_EPOCH=0
ARG SOURCE_DIRTY=true

LABEL org.opencontainers.image.title="BodyComposition" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.source="https://github.com/fohofmann/BodyComposition" \
      org.opencontainers.image.description="Validated CT body-composition research pipeline" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="BodyComposition contributors" \
      org.bodycomposition.uv-lock-sha256="${LOCK_SHA256}" \
      org.bodycomposition.source-sha256="${SOURCE_SHA256}" \
      org.bodycomposition.source-date-epoch="${SOURCE_DATE_EPOCH}" \
      org.bodycomposition.source-dirty="${SOURCE_DIRTY}" \
      org.bodycomposition.cuda-runtime="13.0" \
      org.bodycomposition.model-weights="not-included"

ENV PATH=/opt/bodycomposition/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/bodycomposition \
    BODYCOMPOSITION_MODEL_ROOT=/models \
    BODYCOMPOSITION_OUTPUT_ROOT=/output \
    BODYCOMPOSITION_GIT_SHA=${GIT_SHA} \
    BODYCOMPOSITION_UV_LOCK_SHA256=${LOCK_SHA256} \
    BODYCOMPOSITION_SOURCE_SHA256=${SOURCE_SHA256} \
    BODYCOMPOSITION_SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    BODYCOMPOSITION_SOURCE_DIRTY=${SOURCE_DIRTY} \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=600 \
    XDG_CACHE_HOME=/tmp/bodycomposition-cache \
    MPLCONFIGDIR=/tmp/bodycomposition-matplotlib \
    SKIMAGE_DATADIR=/tmp/bodycomposition-skimage

COPY --from=builder /opt/bodycomposition /opt/bodycomposition
COPY LICENSE THIRD_PARTY_NOTICES.md /usr/share/licenses/bodycomposition/
COPY CITATION.cff CHANGELOG.md SECURITY.md /usr/share/doc/bodycomposition/

RUN mkdir -p /home/bodycomposition /input /output /models /tmp/bodycomposition-cache /tmp/bodycomposition-matplotlib /tmp/bodycomposition-skimage \
    && chown -R 10001:10001 /home/bodycomposition /input /output /models /tmp/bodycomposition-*

USER 10001:10001
WORKDIR /work
VOLUME ["/input", "/output", "/models"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD ["bodycomposition", "version", "--json"]
ENTRYPOINT ["bodycomposition"]
CMD ["--help"]
