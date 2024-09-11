from typing import Union
from pathlib import Path
import SimpleITK as sitk
from nibabel import Nifti1Image
import numpy as np

class NiftiDataContainer():
    """
    Class for loading and handling Nifti data using SimpleITK.
    The origin, direction, and spacing properties of the image are stored and used for file handling.
    If a bounding box (bbox) is set, the image data is cropped, and the origin is adjusted accordingly.
    directions: np: z, y, x, sitk: x, y, z, nifti: x, y, z
    """
    
    def __init__(self, path: Union[str, Path]):

        # save path, can not be changed
        self._path = Path(path)

        # empty data and metadata
        self._data = None
        self._origin = None
        self._direction = None
        self._spacing = None
        self._bbox = None

        # set datatype
        if any(keyword in path.parent.name for keyword in ['label', 'mask']):
            self.dtype = np.uint8
        else:
            self.dtype = np.int16 # int8 would be enough for CT scans, but flexibility if using other dtypes


    def __repr__(self):
        return f'NiftiDataContainer(path={self.path})'

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

        if self._bbox is None:
            return self._origin
        else:
            # Compute the new origin: the start of the bounding box in physical space
            bbox = self._bbox
            # Bounding box is in the order (z, y, x) for numpy arrays, convert to (x, y, z) for physical space
            new_origin_offset = np.array([bbox[4], bbox[2], bbox[0]]) * self._spacing[::-1]
            # Convert self._direction (which is a tuple) into a numpy array and reshape to 3x3
            direction_matrix = np.array(self._direction).reshape(3, 3)
            # Apply the direction matrix to the new offset in physical space
            new_origin = np.array(self._origin) + np.dot(direction_matrix, new_origin_offset)

            return new_origin

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
        """Get SimpleITK image: if bbox is set, export only region within bbox and adjust origin accordingly."""
        if self.data is None:
            return None   
        elif self._origin is None or self._direction is None or self._spacing is None:
            raise ValueError(f'metadata missing, can not create SimpleITK object.')

        if self._bbox is None:
            # No bbox, return the full image
            img_tmp = sitk.GetImageFromArray(self.data)
            img_tmp.SetDirection(self._direction)
            img_tmp.SetOrigin(self._origin)
            img_tmp.SetSpacing(self._spacing)
        else:
            # If bbox is set, extract the subregion
            bbox = self._bbox
            cropped_np = self._data[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] # np: z, y, x

            # Create sitk image from cropped data
            img_tmp = sitk.GetImageFromArray(cropped_np)
            img_tmp.SetDirection(self._direction)
            img_tmp.SetOrigin(self.origin) # origin is adjusted to bbox
            img_tmp.SetSpacing(self._spacing)

        return img_tmp
    

    @img.setter
    def img(self, value: Union[sitk.Image, Nifti1Image]):
        """Load Image, either SimpleITK or Nifti1Image: check and adjust origin, direction, and shape if needed."""

        # load data and metadata
        if isinstance(value, Nifti1Image):
            tmp_data = value.get_fdata().transpose((2, 1, 0)) # Nifti1Image = physical space -> SITK np: z, y, x
            tmp_affine = value.affine
            tmp_direction = tmp_affine[:3, :3] 
            tmp_origin = tmp_affine[:3, 3].tolist()
            tmp_direction = (tmp_direction / value.header.get_zooms()).flatten().tolist() # adjust the direction matrix for spacing, physical space
            tmp_spacing = [float(i) for i in value.header.get_zooms()[::-1]] # Nifti1Image = physical space -> SITK np: z, y, x
        elif isinstance(value, sitk.Image):
            tmp_data = sitk.GetArrayFromImage(value)
            tmp_direction = self._direction = value.GetDirection()
            tmp_origin = self._origin = value.GetOrigin()
            tmp_spacing = value.GetSpacing()
        else:
            raise ValueError(f'Unknown type for image: {type(value)}')
        
        # checks
        if self._bbox is not None and (self._origin is None or self._spacing is None or self._direction is None):
            raise ValueError(f'Bounding box can only be set, if metadata are available.')
        if self._direction is not None and not np.allclose(self._direction, tmp_direction):
            raise ValueError(f'Directions do not match: {self._direction} != {tmp_direction}')
        if self._origin is not None and not np.allclose(self._origin, tmp_origin):
            raise ValueError(f'Origins do not match: {self._origin} != {tmp_origin}')
        if self._spacing is not None and not np.allclose(self._spacing, tmp_spacing):
            raise ValueError(f'Spacings do not match: {self._spacing} != {tmp_spacing}')


        if self._bbox is None:
            # save data & metadata
            self._direction = tmp_direction
            self._origin = tmp_origin
            self._spacing = tmp_spacing
            self.data = tmp_data
        
        else:
            # save data numpy only, metadata are already available (requirement of bbox)
            self.data = tmp_data


    @property
    def imgNifti1(self):
        """Get Nifti1Image: if bbox is set, export only region within bbox and adjust origin accordingly."""
        if self.data is None:
            return None   
        elif self._origin is None or self._direction is None or self._spacing is None:
            raise ValueError(f'metadata missing, can not create Nifti1Image object.')

        # Create the affine matrix
        tmp_direction = np.array(self.direction).reshape(3, 3)  # Reshape into a 3x3 matrix
        tmp_origin = self.origin  # Origin is a 3-element vector
        tmp_spacing = self.spacing[::-1]  # Spacing is a 3-element vector, reverse it to match the physical space
        affine_tmp = np.eye(4)  # Initialize a 4x4 identity matrix
        affine_tmp[:3, :3] = tmp_direction * tmp_spacing # Set the upper 3x3 part to the scaled direction matrix
        affine_tmp[:3, 3] = tmp_origin  # Set the translation (origin) part
        
        # transpose data np
        data_tmp = self.data.transpose((2, 1, 0)) # SITK np: z, y, x -> Nifti1Image = physical space

        return Nifti1Image(data_tmp, affine_tmp)
    

    @property
    def data(self):
        """Get numpy: if no bbox: all. if bbox: only inside."""

        if self._data is None:
            if self.path.exists():
                self.load_from_file()
            else:
                return None

        if self._bbox is None:
            return self._data
        else:
            bbox = self._bbox
            return self._data[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]

    @data.setter
    def data(self, value: np.ndarray):
        """Set numpy: check shape. if no bbox: all. if bbox: only inside."""
        bbox = self._bbox
        if bbox is None:
            if self._data is not None and self._data.shape != value.shape:
                raise ValueError(f'Numpy shapes do not match: {self._data.shape} != {value.shape}')
            self._data = value.astype(self.dtype)
        else:
            bbox_shape = (bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]) # np: z, y, x
            if bbox_shape != value.shape:
                raise ValueError(f'Numpy shapes do not match: {bbox_shape} != {value.shape}')
            self._data[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = value.astype(self.dtype)



    @property
    def bbox(self):
        return self._bbox

    @bbox.setter
    def bbox(self, value: Union[np.ndarray, list]):
        """Sets or resets bounding box: check (array (2,3), metadata must be av."""
        if value is None:
            self._bbox = None
        elif not isinstance(value, (list, np.ndarray)) or len(value) != 6:
            raise ValueError(f'Bounding box must be a list of 6 elements or None.')
        elif self.origin is None or self.spacing is None or self.direction is None:
            raise ValueError(f'Bounding box can only be set, if metadata of original image already available.')
        else:
            self._bbox = value



    @property
    def meta(self):
        """Return metadata as tuple, used during import by other instances."""
        return (self.origin, self.spacing, self.direction)


    @meta.setter
    def meta(self, value: tuple):
        """Set metadata. If already set, check if consistent. If not set, set it."""
        if self._origin is not None and self._origin != value[0]:
            raise ValueError(f'Origins do not match: {self._origin} != {value[0]}')
        elif self._spacing is not None and self._spacing != value[1]:
            raise ValueError(f'Spacings do not match: {self._spacing} != {value[1]}')
        elif self._direction is not None and self._direction != value[2]:
            raise ValueError(f'Directions do not match: {self._direction} != {value[2]}')

        self._origin, self._spacing, self._direction = value



    def save_to_file(self):
        """Save data to nifti file: if NA, error. if AV, use sitk getter."""
        img_tmp = self.img
        if img_tmp is None:
            raise ValueError(f'Nothing to save.')
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



    def remap(self, mapping: dict):
        """Remap labels: replace values in data numpy usimg mapping dictionary."""
        data_tmp = self.data # load np
        if data_tmp is None:
            raise ValueError(f'No data available for remapping.')
        else:
            # create mapping array for relabeling, not existing labels are replaced with 0, then fancy indexing
            labels_max = max(max(mapping.keys()), np.max(data_tmp))
            relabel_array = np.zeros(labels_max+1, dtype=np.uint8)
            for key, value in mapping.items():
                relabel_array[key] = value
            self.data = relabel_array[data_tmp]



    def as_closest_canonical(self):
        """Transform data within bbox to canonical orientation.
        Currently one way function, all changes to data are permanently.
        Multiple reorientations should be avoided to reduce affine inaccuracies that are caused by rounding."""

        # load nib
        if self._data is not None and self._origin is not None and self._direction is not None:
            img_tmp = self.img
        elif self.path.exists():
            img_tmp = sitk.ReadImage(str(self.path))
        else:
            raise ValueError(f'Data not complete, can not reorientate')

        # reorientate sitk
        img_tmp_reoriented = sitk.DICOMOrient(img_tmp, 'RAS')
        data_tmp_reoriented = sitk.GetArrayFromImage(img_tmp_reoriented).astype(self.dtype)

        # reset existing bbox & metadata, set np
        self.bbox = None
        self._origin = img_tmp_reoriented.GetOrigin()
        self._direction = img_tmp_reoriented.GetDirection()
        self._spacing = img_tmp_reoriented.GetSpacing()
        self._data = data_tmp_reoriented
