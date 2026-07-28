"""Three-dimensional image container with explicit SimpleITK axis semantics."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from nibabel import Nifti1Image

from BodyComposition.utils.geometry import GeometryError, ImageGeometry

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


class NiftiDataContainer:
    """Store a three-dimensional image with zyx arrays and xyz metadata."""

    def __init__(
        self,
        path: str | Path,
        *,
        dtype: np.dtype | type | str | None = None,
    ):
        self._path = Path(path)
        self._data = None
        self._origin = None
        self._direction = None
        self._spacing = None

        self.dtype = None if dtype is None else np.dtype(dtype)
        if self.dtype is not None and not np.issubdtype(self.dtype, np.integer):
            raise TypeError("An explicit container dtype must be an integer label dtype.")

    def __repr__(self):
        dtype = self._data.dtype if self._data is not None else self.dtype
        return f"NiftiDataContainer(loaded={self._data is not None}, dtype={dtype or 'preserve'})"

    @property
    def path(self):
        return self._path

    @property
    def spacing(self):
        if self._spacing is None and self.path.exists():
            self.load_from_file()
        return self._spacing

    @property
    def shape(self):
        if self.data is None:
            return None
        return self.data.shape

    @property
    def origin(self):
        if self._origin is None and self.path.exists():
            self.load_from_file()
        return self._origin

    @property
    def direction(self):
        if self._direction is None and self.path.exists():
            self.load_from_file()
        return self._direction

    def exists(self):
        return self._data is not None or self.path.exists()

    def clear(self):
        """Release loaded pixels while retaining physical metadata."""

        self._data = None

    @property
    def img(self):
        """Return a SimpleITK image from the zyx array and xyz metadata."""

        if self.data is None:
            return None
        if self._origin is None or self._direction is None or self._spacing is None:
            raise ValueError("Metadata missing; cannot create a SimpleITK image.")

        image = sitk.GetImageFromArray(self.data)
        image.SetDirection(self._direction)
        image.SetOrigin(self._origin)
        image.SetSpacing(self._spacing)
        return image

    @img.setter
    def img(self, value: sitk.Image | Nifti1Image):
        """Load a SimpleITK or nibabel image while enforcing existing metadata."""

        if isinstance(value, Nifti1Image):
            data = np.asanyarray(value.dataobj).transpose((2, 1, 0))
            affine_lps = LPS_TO_RAS @ np.asarray(value.affine, dtype=float)
            linear_lps = affine_lps[:3, :3]
            spacing = np.linalg.norm(linear_lps, axis=0)
            if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
                raise GeometryError(f"Invalid NIfTI affine spacing: {spacing}.")
            direction = (linear_lps / spacing).flatten()
            origin = affine_lps[:3, 3]
        elif isinstance(value, sitk.Image):
            data = sitk.GetArrayFromImage(value)
            direction = value.GetDirection()
            origin = value.GetOrigin()
            spacing = value.GetSpacing()
        else:
            raise ValueError(f"Unknown image type: {type(value)}")

        direction = tuple(float(item) for item in direction)
        origin = tuple(float(item) for item in origin)
        spacing = tuple(float(item) for item in spacing)

        if self._direction is not None and not np.allclose(self._direction, direction):
            raise ValueError(f"Directions do not match: {self._direction} != {direction}")
        if self._origin is not None and not np.allclose(self._origin, origin):
            raise ValueError(f"Origins do not match: {self._origin} != {origin}")
        if self._spacing is not None and not np.allclose(self._spacing, spacing):
            raise ValueError(f"Spacings do not match: {self._spacing} != {spacing}")

        self._direction = direction
        self._origin = origin
        self._spacing = spacing
        self.data = data

    @property
    def imgNifti1(self):
        """Return a nibabel image while preserving the SimpleITK physical domain."""

        if self.data is None:
            return None
        if self._origin is None or self._direction is None or self._spacing is None:
            raise ValueError("Metadata missing; cannot create a Nifti1Image.")

        affine_lps = self.geometry.affine_lps
        affine_ras = LPS_TO_RAS @ affine_lps

        # SimpleITK exposes zyx arrays; nibabel expects xyz arrays.
        data_xyz = self.data.transpose((2, 1, 0))
        return Nifti1Image(data_xyz, affine_ras)

    @property
    def data(self):
        """Return the array in explicit SimpleITK zyx order."""

        if self._data is None:
            if self.path.exists():
                self.load_from_file()
            else:
                return None
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        """Set a zyx array without implicit cropping."""

        array = np.asarray(value)
        if self._data is not None and self._data.shape != array.shape:
            raise ValueError(f"Numpy shapes do not match: {self._data.shape} != {array.shape}")
        if self.dtype is None:
            self._data = np.array(array, copy=True)
            return

        integer_or_bool = np.issubdtype(array.dtype, np.integer) or np.issubdtype(
            array.dtype,
            np.bool_,
        )
        if np.iscomplexobj(array) or not (
            integer_or_bool or np.issubdtype(array.dtype, np.floating)
        ):
            raise ValueError("Label arrays must contain real numeric data.")
        if not np.all(np.isfinite(array)):
            raise ValueError("Label arrays must contain only finite values.")
        if not integer_or_bool and not np.all(array == np.rint(array)):
            raise ValueError("Label arrays must contain integer-valued data.")
        limits = np.iinfo(self.dtype)
        if array.size and (np.min(array) < limits.min or np.max(array) > limits.max):
            raise ValueError(f"Label values do not fit the configured {self.dtype} dtype.")
        self._data = array.astype(self.dtype, copy=True)

    @property
    def meta(self):
        """Return ``(origin, spacing, direction)`` for another container."""

        return (self.origin, self.spacing, self.direction)

    @property
    def geometry(self) -> ImageGeometry:
        """Return validated image geometry with explicit axis naming."""

        return ImageGeometry.from_container(self)

    @meta.setter
    def meta(self, value: tuple):
        """Set physical metadata, requiring consistency with existing values."""

        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError("Metadata must be an (origin, spacing, direction) tuple.")
        if self._origin is not None and not np.allclose(self._origin, value[0]):
            raise ValueError(f"Origins do not match: {self._origin} != {value[0]}")
        if self._spacing is not None and not np.allclose(self._spacing, value[1]):
            raise ValueError(f"Spacings do not match: {self._spacing} != {value[1]}")
        if self._direction is not None and not np.allclose(self._direction, value[2]):
            raise ValueError(f"Directions do not match: {self._direction} != {value[2]}")

        self._origin = tuple(float(item) for item in value[0])
        self._spacing = tuple(float(item) for item in value[1])
        self._direction = tuple(float(item) for item in value[2])

    def save_to_file(self):
        """Write the loaded image to ``path``."""

        image = self.img
        if image is None:
            raise ValueError("Nothing to save.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(self.path))

    def load_from_file(self):
        """Load ``path`` through SimpleITK."""

        if not self.path.exists():
            raise FileNotFoundError(f"File not available at {self.path}.")
        self.img = sitk.ReadImage(str(self.path))

    def validate(self, *, finite_pixels: bool = True) -> None:
        """Validate pixels and physical geometry before scientific processing."""

        self.geometry.validate()
        if finite_pixels and not np.all(np.isfinite(self.data)):
            raise GeometryError(f"Image contains non-finite pixels: {self.path}.")
