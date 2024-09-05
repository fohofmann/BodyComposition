#!/usr/bin/env python

# import libraries
import argparse
from time import time
import SimpleITK as sitk
import pydicom
import logging
import multiprocessing
import traceback
import pandas as pd
from pathlib import Path
import os

# config
config_min_num_slices = 30

# function to write to error csv, add new line
def track_warnings_as_csv(path, id, error):
    with open(path, 'a') as f:
        f.write(id + ',' + error + '\n')

# function to extract metadata up to a certain key
def extract_metadata(dicom):
    metadata = {}
    for attr in dir(dicom):
        if attr == 'PixelData':
            break
        if not attr.startswith('_'):
            try:
                value = getattr(dicom, attr)
                if not callable(value):
                    metadata[attr] = value
            except AttributeError:
                pass  # Ignore attributes that don't exist
    return metadata

def main():
    """
    Toolkit for transforming DICOM files to NIfTI files, and extracting metadata.
    Usage: bin/prepare_dcm_to_nifti.py -w /path/to/root/dir/ -i input/dcm -o input/nii
    """
        
    # start timer
    glob_time = time()

    # argument parser
    parser = argparse.ArgumentParser(description='toolkit to transform DICOM files to NIfTI files, and extract metadata.')
    parser.add_argument('--workspace', '-w', type=str,
                        required=True, help='path to workspace.')
    parser.add_argument('--input', '-i', type=str,
                        required=True, help='path to subdirectory selected for processing, relative to workspace.')
    parser.add_argument('--output', '-o', type=str,
                        required=True, help='path to subdirectory for processed files, relative to workspace.')
    parser.add_argument('--override', action='store_true', 
                        default=False, help='Override warnings and errors.')
    parser.add_argument('--overwrite', action='store_true',
                        default=False, help='Overwrite existing files.')
    args = parser.parse_args()

    # logging
    path_log = Path(args.workspace, args.output, 'DcmToNifti.log')
    path_warnings = Path(args.workspace, args.output, 'DcmToNifti_warnings.csv')
    path_log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=path_log)

    # logging.info
    logging.info('DATA PREPROCESSING')
    logging.info(f' workspace: {args.workspace}')
    logging.info(f' input: {args.input}')
    logging.info(f' output: {args.output}')

    # full paths, create output directories
    path_input = Path(args.workspace, args.input)

    # check if input directory exists
    if not path_input.exists():
        raise ValueError(f'Input directory `{path_input}` does not exist.')

    # identify all directories containing more than one file
    path_input_dirs = []
    for root, dirs, files in os.walk(path_input):
        if len(files) > 1:
            path_input_dirs.append(Path(root))
    logging.info(f' found {len(path_input_dirs)} directories containing more than one file')

    # remove all cases, that are already processed
    if not args.overwrite:
        path_input_dirs = [path_input_dir for path_input_dir in path_input_dirs if not Path(args.workspace, args.output, 'images', path_input_dir.parent.name + '.nii.gz').exists()]
        logging.info(f' found {len(path_input_dirs)} directories that are not processed yet')

    # multiprocessing
    process = ProcessLoader(args.workspace, args.output, path_warnings=path_warnings, override=args.override)
    n_processes = min(multiprocessing.cpu_count(), len(path_input_dirs))
    logging.info(f" spawn processes at {n_processes}/{multiprocessing.cpu_count()} CPUs\n")
    with multiprocessing.Pool(processes=n_processes, maxtasksperchild=20) as p:
       p.map(process, path_input_dirs)

    # logging.info
    logging.info(f"FINISHED {len(path_input_dirs)} cases ({time()-glob_time:.2f}s)")


class ProcessLoader:
    def __init__(self, path_data, path_output, path_warnings, override=False):

        path_output_images = Path(path_data, path_output, 'images')
        path_output_metadata = Path(path_data, path_output, 'metadata')
        path_output_images.mkdir(parents=True, exist_ok=True)
        path_output_metadata.mkdir(parents=True, exist_ok=True)

        self.root = path_data
        self.output_images = path_output_images
        self.output_metadata = path_output_metadata
        self.path_warnings = path_warnings
        self.override = override

        logging.info(f" initializing method process, creating directories")

    def __call__(self, path_input_dir: Path):
        time_start = time()
        worker_name = multiprocessing.current_process().name
        logging.getLogger().setLevel(logging.INFO)
        logging.debug(f"Processing {path_input_dir} @{worker_name}")

        try:
            input_pat_id = path_input_dir.parent.name
            path_metadata = self.output_metadata / (input_pat_id + ".csv")
            path_image = self.output_images / (input_pat_id + ".nii.gz")
            
            # logging
            logging.debug(f'processing directory {path_input_dir}')
            logging.debug(f' patient ID: `{input_pat_id}`')

            # load dicoms from input directory
            reader = sitk.ImageSeriesReader()
            tmp_dicoms = reader.GetGDCMSeriesFileNames(str(path_input_dir))
            tmp_series = [pydicom.filereader.dcmread(tmp_dicom) for tmp_dicom in tmp_dicoms]

            # create boolean lists for primary and axial images
            tmp_series_primary = [hasattr(tmp, 'ImageType') and any(s.lower() in ['primary', 'original'] for s in tmp.ImageType) for tmp in tmp_series]
            tmp_series_axial = [hasattr(tmp, 'ImageOrientationPatient') and tmp.ImageOrientationPatient == [1, 0, 0, 0, 1, 0] for tmp in tmp_series]

            # select series with primary image type and axial orientation
            series_selected = [tmp for i, tmp in enumerate(tmp_series) if tmp_series_primary[i] and tmp_series_axial[i]]

            # if no series with primary image type and axial orientation, select series with axial orientation
            if not series_selected:
                series_selected = [tmp for i, tmp in enumerate(tmp_series) if tmp_series_axial[i]]

            # if no series with axial orientation, select series with primary image type, assuming that it is axial
            if not series_selected:
                track_warnings_as_csv(self.path_warnings, input_pat_id, 'no axial orientation')
                if not self.override:
                    logging.warning(f" {input_pat_id}: no series with axial orientation found, skipping. Use --override if you wanna try anyway.")
                    return
                series_selected = [tmp for i, tmp in enumerate(tmp_series) if tmp_series_primary[i]]
                for tmp in series_selected:
                    tmp.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                logging.info(f" {input_pat_id}: no series with axial orientation found. As --override is set, trying to set it anyway.")
            
            # if no series with primary image type, select all
            if not series_selected:
                track_warnings_as_csv(self.path_warnings, input_pat_id, 'no primary image')
                if not self.override:
                    logging.warning(f" {input_pat_id}: no series with axial orientation or primary image type found, skipping. Use --override if you wanna try anyway.")
                    return
                series_selected = tmp_series
                for tmp in series_selected:
                    tmp.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                logging.warning(f" {input_pat_id}: no series with primary image type found. As --override is set, trying to set it anyway.")

            # skip if series contains less than files than needed
            if len(series_selected) < config_min_num_slices:
                track_warnings_as_csv(self.path_warnings, input_pat_id, 'too few slices')
                logging.info(f" {input_pat_id}: number of slices is {len(series_selected)}, < {config_min_num_slices}, skipping\n")
                return

            # Extract metadata from the first and last DICOM files
            first_dicom_metadata = extract_metadata(series_selected[0])
            last_dicom_metadata = extract_metadata(series_selected[-1])
            combined_metadata = first_dicom_metadata.copy()
            for key, value in last_dicom_metadata.items():
                if value != "":
                    combined_metadata[key] = value
            metadata_df = pd.DataFrame.from_records([combined_metadata]).T
            metadata_df.to_csv(path_metadata, index=True, header=False)
            logging.debug(f" metadata exported as {path_metadata}")

            # transform dicoms to nifti
            reader.SetFileNames([tmp.filename for tmp in series_selected])
            tmp_image = reader.Execute()
            sitk.WriteImage(tmp_image, path_image)
            logging.debug(f" nifti exported as {path_image}")
            logging.info(f" {input_pat_id}: exported to {path_image} ({time()-time_start:.2f}s)")
        
        except Exception as e:
            logging.error(f"{path_image} failed:\n {e}\n {traceback.format_exc()}\n")

if __name__ == "__main__":
    main() # parser is in main to be available when using pyproject.toml entrypoint