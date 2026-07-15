# syntax=docker/dockerfile:1.7
ARG PLATFORM=linux/amd64
ARG UV_VERSION=0.11.27

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM --platform=${PLATFORM} nvcr.io/nvidia/pytorch:25.03-py3

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Berlin \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PROJECT_ENVIRONMENT=/opt/bodycomposition \
    PATH=/opt/bodycomposition/bin:/usr/local/bin:/usr/bin:/bin \
    SKIMAGE_DATADIR=/app/models/tmp \
    MPLCONFIGDIR=/app/models/tmp

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Resolve the frozen dependency layer before copying the changing source tree.
COPY pyproject.toml uv.lock README.md LICENSE THIRD_PARTY_NOTICES.md ./
RUN uv python install 3.11 \
    && uv sync --frozen --no-dev --no-install-project

COPY . /app
RUN uv sync --frozen --no-dev \
    && mkdir -p /app/data /app/logs /app/models/tmp

# Model weights are deliberately never baked into the public image. Hydrate the
# mounted model directory from the original upstream sources with
# bodycomposition_download_models before inference.
VOLUME ["/app/data", "/app/logs", "/app/models"]

LABEL org.opencontainers.image.title="BodyComposition" \
      org.opencontainers.image.version="0.3" \
      org.opencontainers.image.authors="Felix Hofmann" \
      org.opencontainers.image.source="https://github.com/fohofmann/BodyComposition" \
      org.opencontainers.image.description="CT body-composition analysis with orientation QC and pinned SPINEPS/VERIDAH vertebral labeling." \
      org.opencontainers.image.licenses="Apache-2.0"

ENTRYPOINT ["bodycomposition"]
CMD ["--help"]
