# general libraries
import sys
import tempfile
from pathlib import Path
import multiprocessing
import SimpleITK as sitk
from time import time
import logging
import contextlib
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.geometry import assert_same_physical_domain
from BodyComposition.utils.nifti import NiftiDataContainer, resample_label_to_reference
from BodyComposition.utils.logging import LoggingWriter, log_gpu_usage

# library nnUNet
sl = LoggingWriter(logging.DEBUG)
with contextlib.redirect_stdout(sl), contextlib.redirect_stderr(sl):
    from nnunet.inference.predict import predict_from_folder

# action class
class SegmBOA(PipelineAction):
    """
    Body Regions Segmentation using nnUNet and BOA|Essen
    Body Composition Analysis: https://github.com/UMEssen/Body-and-Organ-Analyzer
    Model weights: https://zenodo.org/record/7918824/, doi: 10.5281/zenodo.7918824
    Model weights source 2: https://github.com/UMEssen/Body-and-Organ-Analysis/releases/download/BCA-BodyRegionsWeights-v0.1.0/Task542_BCA_inference.zip
    License: MIT, https://opensource.org/license/MIT (Zenodo) and Apache License 2.0, https://choosealicense.com/licenses/apache-2.0/ (GitHub)
    Description: compartiments from pretrained nnUNet:
        1: "subcutaneous_tissue",
        2: "muscle",
        3: "abdominal_cavity",
        4: "thoracic_cavity",
        5: "bone",
        6: "glands",
        7: "pericardium",
        8: "breast_implant",
        9: "mediastinum",
        10: "brain",
        11: "nervous_system",
    """

    def __init__(self, pipeline, image: str):
        super().__init__(pipeline)

        # define io
        self.input_image_name = image
        self.output_label_name = 'labels/{caseid}_boa.nii.gz'
        self.io_inputs = [image]
        self.io_outputs = [self.output_label_name]
        self.io_reset_outputs = [self.output_label_name]
        if self.config['segmentation']['save_label']:
            self.io_persisted_outputs = [self.output_label_name]
        self.licenses = ['boa', 'nnunet']

        # set nnunet weight dir, device
        self.model_path = str(pipeline.config['paths']['weights']['boa'])
        self.device = pipeline.device


    def __call__(self, memory):
        """Segment case."""
        super().__call__(memory)
        time_start = time()

        # create case specific nifti data container for output
        output_label_path = memory['workspace']/self.output_label_name.format(caseid=memory['id'])
        output_label = memory[self.output_label_name] = NiftiDataContainer(output_label_path)

        # check if segmentation already available
        if output_label.exists() and self.config['run']['skip']:
            logging.info(f' output: {output_label} available, skipping')
        else:
            # load input container, log
            input_image = memory[self.input_image_name]
            input_image.validate()
            logging.info(f' input: {input_image}')
        
            # do segmentation
            logging.info(f' running segmentation using nnUNetv1|Body and Organ Analysis')

            # temporary directory to catch output
            with tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp_folder:
                tmp_dir = Path(tmp_folder)
                logging.debug(f" temporary directory: {tmp_dir}")

                # load, reorientate & save image temporarily
                img_tmp = input_image.img
                img_tmp = sitk.DICOMOrient(img_tmp, 'LPS')
                sitk.WriteImage(img_tmp, str(tmp_dir / "s01_0000.nii.gz"))

                # redirect stdout and stderr from nnunet to logging
                sl = LoggingWriter(logging.DEBUG)
                with contextlib.redirect_stdout(sl), contextlib.redirect_stderr(sl):
                    # Ref: https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/nnunet/inference/predict.py

                    # avoid pickle error w/ nnunetv1 on macOS
                    if sys.platform == 'darwin':
                        multiprocessing.set_start_method('fork', force=True)
                        logging.warning(f'multiprocessing method temporarily forced to `fork` to avoid pickle error @arm64.')

                    predict_from_folder(model=self.model_path,
                                        input_folder=str(tmp_dir), output_folder=str(tmp_dir),
                                        folds=[0,1,2,3,4],
                                        save_npz=False, num_threads_preprocessing=1, num_threads_nifti_save=1,
                                        lowres_segmentations=None,
                                        part_id=0, num_parts=1, tta=False, mixed_precision=True,
                                        overwrite_existing=True, mode="fastest", overwrite_all_in_gpu=True if self.device.type=='cuda' else False,
                                        step_size=0.5, checkpoint_name="model_final_checkpoint",
                                        segmentation_export_kwargs=None, disable_postprocessing=False)
                    log_gpu_usage()

                    # back to spawn to avoid speed loss if nnUNetv2 is used later
                    if sys.platform == 'darwin':
                        multiprocessing.set_start_method('spawn', force=True)
                
                # load and save output to memory
                segm_tmp = sitk.ReadImage(str(tmp_dir/'s01.nii.gz'))

                # Restore the original physical grid without interpolating label values.
                segm_tmp = resample_label_to_reference(segm_tmp, input_image.img)

                # save to memory
                output_label.img = segm_tmp
                assert_same_physical_domain(
                    input_image.geometry,
                    output_label.geometry,
                    reference_name="input image",
                    candidate_name="BOA label",
                )

            # logging
            logging.info(f' finished segmentation ({time() - time_start:.2f}s)')
            logging.info(f'  output: memory{output_label.path}')

            # saving
            if self.config['segmentation']['save_label']:
                output_label.save_to_file()
                logging.info(f'  file saved')
