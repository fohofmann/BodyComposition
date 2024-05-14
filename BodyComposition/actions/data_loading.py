# libraries
from typing import Union, List
import logging
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.nifti import NiftiDataContainer
from pathlib import Path
import csv
import pydicom
from datetime import datetime
import re

# action class
class LoadNifti(PipelineAction):
    """Action class for loading NiftiDataContainers if not available due to segmentation action."""

    def __init__(self, pipeline, collection: str = None, io_inputs: Union[str, List[str]] = None):
        super().__init__(pipeline)

        # dictionary for collections
        collections = {
            'SarcopeniaTotalSegmentator': ['labels/{caseid}_tseg-spine.nii.gz',
                                           'labels/{caseid}_tseg-vertebralbodies.nii.gz',
                                           'labels/{caseid}_tseg-bodytrunk.nii.gz',
                                           'labels/{caseid}_tseg-tissue.nii.gz'],
        }

        # define io
        tmp_io = []
        if collection is not None:
            tmp_io += collections[collection]
        if isinstance(io_inputs, str):
            tmp_io += [io_inputs]
        elif isinstance(io_inputs, list):
            tmp_io += io_inputs
        self.io_inputs = tmp_io
        self.io_outputs = tmp_io


    def __call__(self, memory):
        """Segment case."""
        super().__call__(memory)

        # loop over all io_inputs
        for io_input in self.io_inputs:
            if not io_input in memory:
                input_path = memory['workspace']/io_input.format(caseid=memory['id'])
                memory[io_input] = NiftiDataContainer(input_path)
                logging.info(f'  loaded {memory[io_input]}')


# action class
class LoadMetadata(PipelineAction):
    """Action class for loading metadata."""

    def __init__(self, pipeline, input: str = None):
        super().__init__(pipeline)

        # io to pipeline
        self.io_inputs = [] # optional, does not
        self.io_outputs = ['tmp/metadata']
        self.input_name = str(input)


    def __call__(self, memory):
        """Segment case."""
        super().__call__(memory)

        # create path
        input_parent = memory['tmp/index'].path.parent.parent
        metadata_file_path = Path(input_parent, self.input_name.format(caseid=memory['id']))

        # if metadata available, import
        if not metadata_file_path.exists():
            logging.info(f'  metadata ({metadata_file_path}) not found, import skipped.')
            scan_date = None
            pat_sex = None
            pat_size = None
            pat_weight = None

        elif metadata_file_path.suffix == '.csv':
            with open(metadata_file_path, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                metadata_dict = {rows[0]: rows[1] for rows in reader if len(rows) > 1}
            scan_date = metadata_dict.get('AcquisitionDate', None)
            pat_sex = metadata_dict.get('PatientSex', None)
            pat_size = metadata_dict.get('PatientSize', None)
            pat_weight = metadata_dict.get('PatientWeight', None)

        elif metadata_file_path.suffix == '.dcm':
            metadata = pydicom.dcmread(metadata_file_path)
            scan_date = metadata.AcquisitionDate
            pat_sex = metadata.PatientSex
            pat_size = metadata.PatientSize
            pat_weight = metadata.PatientWeight

        else:
            raise AssertionError(f'Unknown metadata file type: {metadata_file_path.suffix}')

        # transform scan_date
        if scan_date:
            scan_date = datetime.strptime(scan_date, '%Y%m%d').strftime('%Y-%m-%d')

        # get patient id
        match = re.match(r'([a-zA-Z0-9]+)-([a-zA-Z0-9]+)?-?([a-zA-Z0-9]+)?', memory['id'])
        if match:
            pat_prefix, pat_id, pat_suffix = match.groups()
        else:
            pat_prefix, pat_id, pat_suffix = "", memory['id'], ""

        # create dictionary
        metadata = {
            'pat_id': pat_id,
            'pat_prefix' : pat_prefix,
            'pat_suffix': pat_suffix,
            'scan_date': scan_date,
            'pat_sex': pat_sex,
            'pat_size': pat_size,
            'pat_weight': pat_weight,
            'scan_date': scan_date,
        }

        # save to memory
        memory['tmp/metadata'] = metadata
        logging.info(f' loaded to memory:tmp/metadata')
