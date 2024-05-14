#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from setuptools import find_packages, setup

setup(
    name="BodyComposition",
    version="0.1",
    author="Hofmann FO",
    url="https://github.com/fohofmann/BodyComposition",
    description="Pipeline for the segmentation of muscle and adipose tissue with postprocessing for mapping to vertebral levels and calculation of cross-sectional areas.",
    packages=find_packages(exclude=("data", "docs", "logs", "models", "suppl")),
    python_requires=">=3.9",
    install_requires=[
        "requests==2.27.1",
        "psutil",
        "numpy",
        "pandas",
        "nibabel",
        "pydicom",
        "SimpleITK",
        "tqdm",
        "nnunet",
        "nnunetv2",
        "totalsegmentator @ git+https://github.com/wasserth/TotalSegmentator.git",
    ],
)
