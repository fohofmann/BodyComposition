# general libraries
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.nifti import NiftiDataContainer
from time import time

# logging
import logging
import contextlib
from BodyComposition.utils.logging import LoggingWriter, log_gpu_usage

# library nnUNet
sl = LoggingWriter(logging.DEBUG)
with contextlib.redirect_stdout(sl), contextlib.redirect_stderr(sl):
    from nnunet.inference.predict import predict_from_folder

# other
import sys
import tempfile
from pathlib import Path
import multiprocessing
from nibabel import load as nib_load, save as nib_save
from totalsegmentator.alignment import as_closest_canonical, undo_canonical

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
        self.licenses = ['stanford', 'nnunet']

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
                input_image_nib = input_image.data_nib
                nib_save(as_closest_canonical(input_image_nib), tmp_dir / "s01_0000.nii.gz")

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
                                        step_size=0.5, checkpoint_name="model_best",
                                        segmentation_export_kwargs=None, disable_postprocessing=False)
                    log_gpu_usage()

                    # back to spawn to avoid speed loss if nnUNetv2 is used later
                    if sys.platform == 'darwin':
                        multiprocessing.set_start_method('spawn', force=True)
                
                # load and save output
                output_label.data_nib = nib_load(tmp_dir/'s01.nii.gz')

            # logging
            logging.info(f' finished segmentation ({time() - time_start:.2f}s)')
            logging.info(f'  output: memory{output_label.path}')

            # saving
            if self.config['segmentation']['save_label']:
                output_label.save_to_file()
                logging.info(f'  file saved')