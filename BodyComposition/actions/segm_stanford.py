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
from BodyComposition.utils.nifti import NiftiDataContainer
from BodyComposition.utils.logging import LoggingWriter, log_gpu_usage

# library nnUNet
sl = LoggingWriter(logging.DEBUG)
with contextlib.redirect_stdout(sl), contextlib.redirect_stderr(sl):
    from nnunet.inference.predict import predict_from_folder

# action class
class SegmStanfordSpine(PipelineAction):
    """
    Spine segmentation using Stanford Spine model.
    Model weights v1: https://huggingface.co/louisblankemeier/spine_v1 (license unclear, not used here)
    Model weights v2: https://huggingface.co/louisblankemeier/stanford_spine
    License: Apache License 2.0, https://choosealicense.com/licenses/apache-2.0/
    Labels: like verse labels
    """

    def __init__(self, pipeline, image: str):
        super().__init__(pipeline)

        # define io
        self.input_image_name = image
        self.output_label_name = 'labels/{caseid}_stanford-spine.nii.gz'
        self.io_inputs = [image]
        self.io_outputs = [self.output_label_name]
        self.licenses = ['stanford-spine', 'nnunet']

        # set nnunet weight dir, device
        self.model_path = str(pipeline.config['paths']['weights']['stanford-spine'])
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
            logging.info(f' input: {input_image}')
        
            # do segmentation
            logging.info(f' running segmentation using nnUNetv1|StanfordSpinev2')

            # temporary directory to catch output
            with tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp_folder:
                tmp_dir = Path(tmp_folder)
                logging.debug(f" temporary directory: {tmp_dir}")

                # load, reorientate & save image temporarily
                img_tmp = input_image.img
                img_tmp = sitk.DICOMOrient(img_tmp, 'RAS')
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
                                        folds=[0],
                                        save_npz=False, num_threads_preprocessing=1, num_threads_nifti_save=1,
                                        lowres_segmentations=None,
                                        part_id=0, num_parts=1, tta=False, mixed_precision=True,
                                        overwrite_existing=True, mode="fastest", overwrite_all_in_gpu=True if self.device.type=='cuda' else False,
                                        step_size=0.5, checkpoint_name="model_latest",
                                        segmentation_export_kwargs=None, disable_postprocessing=False)
                    log_gpu_usage()

                    # back to spawn to avoid speed loss if nnUNetv2 is used later
                    if sys.platform == 'darwin':
                        multiprocessing.set_start_method('spawn', force=True)
                
                # load and save output to memory
                segm_tmp = sitk.ReadImage(str(tmp_dir/'s01.nii.gz'))

                # Resample the image back to its original orientation using the original direction, origin, and spacing
                segm_tmp = sitk.Resample(
                    segm_tmp,
                    transform=sitk.Transform(3, sitk.sitkIdentity),
                    outputDirection=input_image.direction,
                    outputOrigin=input_image.origin,
                    outputSpacing=input_image.spacing,
                    size=input_image.img.GetSize(),
                    interpolator=sitk.sitkLinear
                )

                # save to memory
                output_label.img = segm_tmp

            # logging
            logging.info(f' finished segmentation ({time() - time_start:.2f}s)')
            logging.info(f'  output: memory{output_label.path}')

            # saving
            if self.config['segmentation']['save_label']:
                output_label.save_to_file()
                logging.info(f'  file saved')



# action class
class SegmStanfordTissue(PipelineAction):
    """
    Tissue segmentation using Stanford Tissue model.
    Model weights: https://huggingface.co/stanfordmimi/multilevel_muscle_adipose_tissue/
    License: Apache License 2.0, https://choosealicense.com/licenses/apache-2.0/
    Labels: 1:SAT, 2:VAT, 3:IMAT, 4:Muscle
    """

    def __init__(self, pipeline, image: str):
        super().__init__(pipeline)

        # define io
        self.input_image_name = image
        self.output_label_name = 'labels/{caseid}_stanford-tissue.nii.gz'
        self.io_inputs = [image]
        self.io_outputs = [self.output_label_name]
        self.licenses = ['stanford-tissue', 'nnunet']

        # set nnunet weight dir, device
        self.model_path = str(pipeline.config['paths']['weights']['stanford-tissue'])
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
            logging.info(f' input: {input_image}')
        
            # do segmentation
            logging.info(f' running segmentation using nnUNetv1|StanfordTissue')

            # temporary directory to catch output
            with tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp_folder:
                tmp_dir = Path(tmp_folder)
                logging.debug(f" temporary directory: {tmp_dir}")

                # load, reorientate & save image temporarily
                img_tmp = input_image.img
                img_tmp = sitk.DICOMOrient(img_tmp, 'RAS')
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
                                        folds="all",
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
                
                # load and save output
                segm_tmp = sitk.ReadImage(str(tmp_dir/'s01.nii.gz'))
                segm_tmp.SetDirection(input_image.direction)
                segm_tmp.SetOrigin(input_image.origin)
                segm_tmp.SetSpacing(input_image.spacing)
                output_label.img = segm_tmp

            # logging
            logging.info(f' finished segmentation ({time() - time_start:.2f}s)')
            logging.info(f'  output: memory{output_label.path}')

            # saving
            if self.config['segmentation']['save_label']:
                output_label.save_to_file()
                logging.info(f'  file saved')