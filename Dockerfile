# syntax=docker/dockerfile:1
ARG PLATFORM=linux/amd64
FROM --platform=${PLATFORM} nvcr.io/nvidia/pytorch:24.06-py3

# Set environment
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Berlin \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    SKIMAGE_DATADIR="/app/models/tmp/" \
    MPLCONFIGDIR="/app/models/tmp/"

# Install packages and clean up in a single layer
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    git libsm6 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy files and install app
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e /app

# reinstall opencv-python-headless
RUN pip uninstall --no-cache-dir -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir opencv-python-headless==4.8.0.74

# copy custom trainers, hard coded python3.10
# needed if somebody wants to use stanford model, 16000 epochs, oh boy...
COPY /suppl/custom_trainers/* /usr/local/lib/python3.10/dist-packages/nnunet/training/network_training/

# Create directories as mounting points
RUN mkdir -p data logs models

# Labels
LABEL org.opencontainers.image.title="BodyComposition" \
      org.opencontainers.image.version="0.2" \
      org.opencontainers.image.authors="Felix Hofmann" \
      org.opencontainers.image.source=https://github.com/fohofmann/BodyComposition \
      org.opencontainers.image.description="Pipeline for the segmentation of muscle and adipose tissue with postprocessing for mapping to vertebral levels and calculation of cross-sectional areas." \
      org.opencontainers.image.licenses=Apache-2.0