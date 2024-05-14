# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:23.12-py3

# set environment
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV SKIMAGE_DATADIR="/app/models/tmp/"
ENV MPLCONFIGDIR="/app/models/tmp/"

# install packages
RUN apt update && apt upgrade -y
RUN apt-get -qq install -y --no-install-recommends \
    git-all ffmpeg libsm6 libxext6 xvfb nvidia-driver-525 python3-venv slurm-client

# install python packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    numpy pandas nibabel SimpleITK pydicom psutil pyradiomics

# copy files into app direatory
COPY . /app
RUN pip install --no-cache-dir -e /app

# copy custom trainers, hard coded python3.10
# needed if somebody wants to use stanford model, 16000 epochs, oh boy...
COPY /suppl/custom_trainers /usr/local/lib/python3.10/dist-packages/nnunet/training/custom_trainers

# reinstall opencv-python-headless
RUN pip uninstall --no-cache-dir -y opencv-python
RUN pip uninstall --no-cache-dir -y opencv-python-headless
RUN pip install --no-cache-dir opencv-python-headless==4.8.0.74

# create directories as mounting points
WORKDIR /app
RUN mkdir -p data
RUN mkdir -p logs
RUN mkdir -p models

# clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# labels
LABEL org.opencontainers.image.title="BodyComposition"
LABEL org.opencontainers.image.version="0.1"
LABEL org.opencontainers.image.source=https://github.com/fohofmann/BodyComposition
LABEL org.opencontainers.image.description="Pipeline for the segmentation of muscle and adipose tissue with postprocessing for mapping to vertebral levels and calculation of cross-sectional areas."
LABEL org.opencontainers.image.licenses=Apache-2.0