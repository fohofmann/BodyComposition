from typing import Union
from pathlib import Path
from nibabel import Nifti1Image, load as nib_load, save as nib_save, as_closest_canonical
import numpy as np

class NiftiDataContainer():
    """
    Class for loading and handling the Nifti data.
    Principles: 
    - data stored as numpy only, nibabel object is not stored.
    - orientation is as in original file
    - metadata stored separetely, and used to for exports and imports (especially to bbox)
    - data are not loaded upon initialization, but only when needed; some actions only require file path.
    """
    
    def __init__(self, path: Union[str, Path]):

        # save path, can not be changed
        self._path = Path(path)

        # empty data and metadata
        self._data_np = None
        self._affine = None
        self._shape = None
        self._spacing = None
        self._bbox = None

        # set datatype
        if any(keyword in path.parent.name for keyword in ['label', 'mask']):
            self.dtype = np.uint8
        else:
            self.dtype = np.int16 # enough for CT scans, but flexibility if using other dtypes


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
        if self.data_np is None:
            return None
        else:
            return self.data_np.shape
        
    @property
    def affine(self):
        # not available: try to load from file
        if self._affine is None and self.path.exists():
            self.load_from_file()

        bbox = self._bbox
        if bbox is None:
            return self._affine
        else:
            affine_cropped = self._affine.copy()
            bbox_ras = np.array([bbox[0], bbox[2], bbox[4]])
            affine_cropped[:3, 3] = np.dot(self._affine[:3, :3], bbox_ras) + self._affine[:3, 3]
            return affine_cropped
        
    
    def exists(self):
        return self._data_np is not None or self.path.exists()
        
    def clear(self):
        """Clears data_np, usefull for making some space, keeps metadata."""
        self._data_np = None



    @property
    def data_nib(self):
        """Get nibabel: transforms np to nib, if bbox adjust affine"""

        # some checks
        if self.data_np is None:
            return None
        if self._affine is None:
            raise ValueError(f'metadata missing, can not create nibabel object.')
        
        # export
        return Nifti1Image(self.data_np, self.affine)
            

    @data_nib.setter
    def data_nib(self, value: Nifti1Image):
        """Set nibabel: depending on bbox, check adjust affine and shape."""

        if self._bbox is None:
            # check affine and spacing
            # data_np setter checks shape
            if self._affine is not None and not np.array_equal(self._affine, value.affine):
                raise ValueError(f'Affines do not match: {self._affine} != {value.affine}')
            elif self._spacing is not None and self._spacing != value.header.get_zooms():
                raise ValueError(f'Spacings do not match: {self._spacing} != {value.header.get_zooms()}')
            
            # save data & metadata
            self._affine = value.affine
            self._spacing = value.header.get_zooms()
            self.data_np = value.get_fdata()
        
        else:

            # check affine and spacing
            # data_np setter checks shape
            if not np.array_equal(self._affine, value.affine):
                raise ValueError(f'Affines do not match: {self.affine} != {value.affine}')
            elif self.spacing != value.header.get_zooms():
                raise ValueError(f'Shapes do not match: {self.spacing} != {value.header.get_zooms()}')
        
            # save data
            # metadata are already available (requirement of bbox)
            self.data_np = value.get_fdata()



    @property
    def data_np(self):
        """Get numpy: if no bbox: all. if bbox: only inside."""

        if self._data_np is None:
            if self.path.exists():
                self.load_from_file()
            else:
                return None

        if self._bbox is None:
            return self._data_np
        else:
            bbox = self._bbox
            return self._data_np[bbox[0]:bbox[1],
                                 bbox[2]:bbox[3],
                                 bbox[4]:bbox[5]]

    @data_np.setter
    def data_np(self, value: np.ndarray):
        """Set numpy: check shape. if no bbox: all. if bbox: only inside."""

        bbox = self._bbox
        if bbox is None:
            if self._shape is None:
                self._shape = value.shape
            elif self._shape != value.shape:
                raise ValueError(f'Numpy shapes do not match: {self._shape} != {value.shape}')
            self._data_np = value.astype(self.dtype)

        else:
            bbox_shape = (bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]) # RAS+
            if bbox_shape != value.shape:
                raise ValueError(f'Numpy shapes do not match: {bbox_shape} != {value.shape}')
            self._data_np[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = value.astype(self.dtype)



    @property
    def bbox(self):
        return self._bbox

    @bbox.setter
    def bbox(self, value: np.ndarray):
        """Sets or resets bounding box: check (array (2,3), metadata must be av."""
        if value is None:
            self._bbox = None
        elif not isinstance(value, list) or len(value)!=6: 
            raise ValueError(f'Bounding box must be a list of 6 elements or None.')
        elif any(element is None for element in self.meta):
            raise ValueError(f'Bounding box can only be set, if metadata of original image already available.')
        else:
            self._bbox = value



    @property
    def meta(self):
        """Return metadata as tuple, used during import by other instances."""
        return (self.affine, self.shape, self.spacing)


    @meta.setter
    def meta(self, value: tuple):
        """Set metadata. If already set, check if consistent. If not set, set it."""
        if self._affine is not None and self._affine != value[0]:
            raise ValueError(f'Affines do not match: {self._affine} != {value[0]}')
        elif self._shape is not None and self._shape != value[1]:
            raise ValueError(f'Shapes do not match: {self._shape} != {value[1]}')
        elif self._spacing is not None and self._spacing != value[2]:
            raise ValueError(f'Spacings do not match: {self._spacing} != {value[2]}')
        self._affine, self._shape, self._spacing = value



    def save_to_file(self):
        """Save data to nifti file: if NA, error. if AV, use nib getter."""
        data_nib = self.data_nib
        if data_nib is None:
            raise ValueError(f'Nothing to save.')  
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            nib_save(data_nib, self.path)

    def load_from_file(self):
        """Load data from nifti file: if NA, error. if AV, use nib setter"""
        if not self.path.exists():
            raise FileNotFoundError(f'File not available at {self.path}.')
        else:       
            self.data_nib = nib_load(self.path)



    def remap(self, mapping: dict):
        """Remap labels: replace values in data_np usimg mapping dictionary."""
        tmp_data_np = self.data_np # load np
        if tmp_data_np is None:
            raise ValueError(f'No data available for remapping.')
        else:
            # create mapping array for relabeling, not existing labels are replaced with 0, then fancy indexing
            labels_max = max(max(mapping.keys()), np.max(tmp_data_np))
            relabel_array = np.zeros(labels_max+1, dtype=np.uint8)
            for key, value in mapping.items():
                relabel_array[key] = value
            self.data_np = relabel_array[tmp_data_np]



    def as_closest_canonical(self):
        """Transform data within bbox to canonical orientation.
        Currently one way function, all changes to data are permanently.
        Multiple reorientations should be avoided to reduce affine inaccuracies that are caused by rounding."""

        # load nib
        if self._data_np is not None and self._affine is not None:
            data_nib = self.data_nib
        elif self.path.exists():
            data_nib = nib_load(self.path)
        else:
            raise ValueError(f'Data not complete, can not reorientate')

        # reorientate nib
        data_reoriented_nib = as_closest_canonical(data_nib)
        data_reoriented_np = data_reoriented_nib.get_fdata().astype(self.dtype)

        # reset existing bbox & metadata, set np
        self.bbox = None
        self._affine = data_reoriented_nib.affine
        self._shape = data_reoriented_np.shape
        self._spacing = data_reoriented_nib.header.get_zooms()
        self._data_np = data_reoriented_np