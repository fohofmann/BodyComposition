from pathlib import Path

import numpy as np
import SimpleITK as sitk
from nibabel import Nifti1Image

from BodyComposition.utils.geometry import GeometryError, ImageGeometry

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


class NiftiDataContainer:
    """Store a three-dimensional image with explicit zyx arrays and xyz metadata."""

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
        else:
            return self.data.shape # if bbox: inside only
        
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
        """Clears _data, usefull for making some space, keeps metadata."""
        self._data = None



    @property
    def img(self):
        """Return a SimpleITK image from the explicit zyx array and xyz metadata."""
        if self.data is None:
            return None   
        elif self._origin is None or self._direction is None or self._spacing is None:
            raise ValueError('metadata missing, can not create SimpleITK object.')

        img_tmp = sitk.GetImageFromArray(self.data)
        img_tmp.SetDirection(self._direction)
        img_tmp.SetOrigin(self._origin)
        img_tmp.SetSpacing(self._spacing)

        return img_tmp
    

    @img.setter
    def img(self, value: sitk.Image | Nifti1Image):
        """Load Image, either SimpleITK or Nifti1Image: check and adjust origin, direction, and shape if needed."""

        # load data and metadata
        if isinstance(value, Nifti1Image):
            tmp_data = np.asanyarray(value.dataobj).transpose((2, 1, 0))
            affine_lps = LPS_TO_RAS @ np.asarray(value.affine, dtype=float)
            linear_lps = affine_lps[:3, :3]
            tmp_spacing = np.linalg.norm(linear_lps, axis=0)
            if not np.all(np.isfinite(tmp_spacing)) or np.any(tmp_spacing <= 0):
                raise GeometryError(f"Invalid NIfTI affine spacing: {tmp_spacing}.")
            tmp_direction = (linear_lps / tmp_spacing).flatten()
            tmp_origin = affine_lps[:3, 3]

        elif isinstance(value, sitk.Image):
            tmp_data = sitk.GetArrayFromImage(value)
            tmp_direction = value.GetDirection()
            tmp_origin = value.GetOrigin()
            tmp_spacing = value.GetSpacing()
        else:
            raise ValueError(f'Unknown type for image: {type(value)}')
        
        # checks
        tmp_direction = tuple(float(item) for item in tmp_direction)
        tmp_origin = tuple(float(item) for item in tmp_origin)
        tmp_spacing = tuple(float(item) for item in tmp_spacing)

        if self._direction is not None and not np.allclose(self._direction, tmp_direction):
            raise ValueError(f'Directions do not match: {self._direction} != {tmp_direction}')
        if self._origin is not None and not np.allclose(self._origin, tmp_origin):
            raise ValueError(f'Origins do not match: {self._origin} != {tmp_origin}')
        if self._spacing is not None and not np.allclose(self._spacing, tmp_spacing):
            raise ValueError(f'Spacings do not match: {self._spacing} != {tmp_spacing}')


        self._direction = tmp_direction
        self._origin = tmp_origin
        self._spacing = tmp_spacing
        self.data = tmp_data


    @property
    def imgNifti1(self):
        """Return a nibabel image while preserving the SimpleITK physical domain."""
        if self.data is None:
            return None   
        elif self._origin is None or self._direction is None or self._spacing is None:
            raise ValueError('Metadata missing, cannot create Nifti1Image object.')
        
        affine_lps = self.geometry.affine_lps
        affine_ras = LPS_TO_RAS @ affine_lps
        
        # SimpleITK exposes zyx arrays; nibabel expects xyz arrays.
        data_tmp = self.data.transpose((2, 1, 0))

        return Nifti1Image(data_tmp, affine_ras)
    

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
        """Set an array in explicit SimpleITK zyx order without implicit cropping."""
        array = np.asarray(value)
        if self._data is not None and self._data.shape != array.shape:
            raise ValueError(f'Numpy shapes do not match: {self._data.shape} != {array.shape}')
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
            raise ValueError(
                f"Label values do not fit the configured {self.dtype} dtype."
            )
        self._data = array.astype(self.dtype, copy=True)



    @property
    def meta(self):
        """Return metadata as tuple, used during import by other instances."""
        return (self.origin, self.spacing, self.direction)


    @property
    def geometry(self) -> ImageGeometry:
        """Return validated image geometry with explicit axis naming."""
        return ImageGeometry.from_container(self)


    @meta.setter
    def meta(self, value: tuple):
        """Set metadata. If already set, check if consistent. If not set, set it."""
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError('Metadata must be an (origin, spacing, direction) tuple.')
        if self._origin is not None and not np.allclose(self._origin, value[0]):
            raise ValueError(f'Origins do not match: {self._origin} != {value[0]}')
        elif self._spacing is not None and not np.allclose(self._spacing, value[1]):
            raise ValueError(f'Spacings do not match: {self._spacing} != {value[1]}')
        elif self._direction is not None and not np.allclose(self._direction, value[2]):
            raise ValueError(f'Directions do not match: {self._direction} != {value[2]}')

        self._origin = tuple(float(item) for item in value[0])
        self._spacing = tuple(float(item) for item in value[1])
        self._direction = tuple(float(item) for item in value[2])



    def save_to_file(self):
        """Save data to nifti file: if NA, error. if AV, use sitk getter."""
        img_tmp = self.img
        if img_tmp is None:
            raise ValueError('Nothing to save.')
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(img_tmp, str(self.path))

    def load_from_file(self):
        """Load data from nifti file: if NA, error. if AV, use sitk setter"""
        if not self.path.exists():
            raise FileNotFoundError(f'File not available at {self.path}.')
        else:
            img_tmp = sitk.ReadImage(str(self.path))
            self.img = img_tmp


    def validate(self, *, finite_pixels: bool = True) -> None:
        """Validate pixels and physical geometry before scientific processing."""
        self.geometry.validate()
        if finite_pixels and not np.all(np.isfinite(self.data)):
            raise GeometryError(f'Image contains non-finite pixels: {self.path}.')
