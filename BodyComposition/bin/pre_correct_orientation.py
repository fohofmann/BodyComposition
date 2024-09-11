#!/usr/bin/env python
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pydicom
import argparse
import pandas as pd


def main():
    """
    Main function to process DICOM files and correct orientation if needed.
    """
    parser = argparse.ArgumentParser(description='Check and correct DICOM orientation and image type for files with warnings.')
    parser.add_argument('-i', '--input_dir', required=True, help='Path to the root directory containing DICOM files.')
    parser.add_argument('-w', '--warnings_file', required=True, help='Path to the CSV file containing image ids with errors.')
    parser.add_argument('-o', '--output_dir', required=True, help='Path to the output directory for corrected NIfTI files.')
    parser.add_argument('-m', '--method', default='upsidedown', choices=['upsidedown', 'rotate180'],
                        help='Method to correct the orientation of the DICOM images.')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    warnings_df = pd.read_csv(args.warnings_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in warnings_df.iterrows():
        patient_id = row['patient_id']
        warning = row['warning']
        patient_dir = input_dir / patient_id

        print(f"\nProcessing patient {patient_id} with warning: {warning}")

        # Find all DICOM files in the directory
        dicom_files = []
        for file_path in patient_dir.rglob('*'):
            if file_path.is_file() and (file_path.suffix.lower() == '.dcm' or file_path.suffix == ''):
                try:
                    dicom = pydicom.dcmread(str(file_path))
                    if hasattr(dicom, 'PixelData'):
                        dicom_files.append(str(file_path))
                except:
                    pass  # Not a valid DICOM file, skip it

        if not dicom_files:
            print(f"- no DICOM files found in `{patient_dir}`. Skipping.")
            continue

        # Read the DICOM series
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_files)

        try:
            image_sitk = reader.Execute()
        except RuntimeError as e:
            print(f"- Error reading DICOM series: {e}")
            continue

        # Convert the image to a numpy array
        image_np = sitk.GetArrayFromImage(image_sitk)
        image_origin = image_sitk.GetOrigin()
        image_spacing = image_sitk.GetSpacing()
        image_direction = image_sitk.GetDirection()
        print(f"Image orientation: {image_direction}")

        # Apply the orientation transformation
        if args.method == 'upsidedown':
            # upside down
            image_reorientated = image_sitk
            image_reorientated.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, -1])
            image_reorientated.SetOrigin(image_origin)
            image_reorientated.SetSpacing(image_spacing)
        elif args.method == 'rotate180':
            # rotate image 180 degrees around the z-axis
            image_reorientated = image_sitk
            image_reorientated.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, -1])
            image_reorientated.SetOrigin(image_origin)
            image_reorientated.SetSpacing(image_spacing)
        else:
            print(f"- Invalid method: {args.method}. Skipping.")
            continue

        output_path = output_dir / f"{patient_id}.nii.gz"
        sitk.WriteImage(image_reorientated, str(output_path))
        print(f"Saved corrected NIfTI file to {output_path}")

if __name__ == "__main__":
    main()