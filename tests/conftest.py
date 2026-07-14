from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

from BodyComposition.utils.config import update_config, validate_config
from BodyComposition.utils.nifti import NiftiDataContainer


@pytest.fixture
def base_config():
    config = update_config({}, Path("config/config.yaml"))
    config = update_config(config, Path("config/labels.yaml"))
    validate_config(config)
    return config


@pytest.fixture
def pipeline_stub(base_config):
    return SimpleNamespace(config=base_config, timestamp=1234567890, device="cpu")


@pytest.fixture
def sitk_image_factory():
    return make_sitk_image


@pytest.fixture
def container_factory():
    return make_container


def make_sitk_image(
    array_zyx: np.ndarray,
    *,
    spacing_xyz=(1.0, 1.0, 1.0),
    origin_lps_xyz=(0.0, 0.0, 0.0),
    direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
) -> sitk.Image:
    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing(tuple(float(value) for value in spacing_xyz))
    image.SetOrigin(tuple(float(value) for value in origin_lps_xyz))
    image.SetDirection(tuple(float(value) for value in direction_lps))
    return image


def make_container(path: Path, array_zyx: np.ndarray, **geometry) -> NiftiDataContainer:
    container = NiftiDataContainer(path)
    container.img = make_sitk_image(array_zyx, **geometry)
    return container
