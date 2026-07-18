from pathlib import Path

import numpy as np
import SimpleITK as sitk
from nibabel import Nifti1Image

from BodyComposition.utils.geometry import GeometryError, ImageGeometry

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


class NiftiDataContainer:
    """
    Class for loading and handling Nifti data using SimpleITK.
    The origin, direction, and spacing properties of the image are stored and used for file handling.
    directions: np: z, y, x, sitk: x, y, z, nifti: x, y, z
    """
    
    def __init__(self, path: str | Path):

        # save path, can not be changed
        self._path = Path(path)

        # empty data and metadata
        self._data = None
        self._origin = None
        self._direction = None
        self._spacing = None

        # set datatype
        if any(keyword in self._path.parent.name for keyword in ['label', 'mask']):
            self.dtype = np.uint8
        else:
            self.dtype = np.int16 # int8 would be enough for CT scans, but flexibility if using other dtypes


    def __repr__(self):
        return f"NiftiDataContainer(loaded={self._data is not None}, dtype={self.dtype})"

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
        
        # Transpose the data from z, y, x (SITK numpy) to x, y, z (Nifti1Image expected orientation)
        data_tmp = self.data.transpose((2, 1, 0))

        return Nifti1Image(data_tmp.astype(self.dtype, copy=False), affine_ras)
    

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
        if self._data is not None and self._data.shape != value.shape:
            raise ValueError(f'Numpy shapes do not match: {self._data.shape} != {value.shape}')
        self._data = value.astype(self.dtype)



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
            #print(f'Image loaded from {self.path}:\n - Origin: {img_tmp.GetOrigin()} \n - Direction: {img_tmp.GetDirection()} \n - Spacing: {img_tmp.GetSpacing()}')
            self.img = img_tmp


    def validate(self, *, finite_pixels: bool = True) -> None:
        """Validate pixels and physical geometry before scientific processing."""
        self.geometry.validate()
        if finite_pixels and not np.all(np.isfinite(self.data)):
            raise GeometryError(f'Image contains non-finite pixels: {self.path}.')



    def remap(self, mapping: dict):
        """Remap labels: replace values in data numpy usimg mapping dictionary."""
        data_tmp = self.data # load np
        if data_tmp is None:
            raise ValueError('No data available for remapping.')
        else:
            # create mapping array for relabeling, not existing labels are replaced with 0, then fancy indexing
            labels_max = max(max(mapping.keys()), np.max(data_tmp))
            relabel_array = np.zeros(labels_max+1, dtype=np.uint8)
            for key, value in mapping.items():
                relabel_array[key] = value
            self.data = relabel_array[data_tmp]
