# general libraries
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.nifti import NiftiDataContainer
from time import time

# logging
import logging
import contextlib
from BodyComposition.utils.logging import LoggingWriter, log_gpu_usage

# action class
class SegmIntVertebrae(PipelineAction):
    """
    Spine Segmentation using nnUNet
    Ref: https://huggingface.co/fhofmann/VertebralBodies-nnUNet-Task601

    """

    def __init__(self, pipeline, image: str, model: str = 'ResEncM'):
        super().__init__(pipeline)

        # define io
        self.input_image_name = image
        self.output_label_name = 'labels/{caseid}_int-vertebrae.nii.gz'
        self.io_inputs = [image]
        self.io_outputs = [self.output_label_name]
        self.licenses = ['nnunet', 'nnunet_resenc', 'intvertebrae', 'verse', 'BOA', 'totalsegmentator']

        # select correct model
        if model == 'ResEncM':
            logging.info('  using fast model: ResEncM, fold=all')
            model_path = self.config['paths']['weights']['int-vertebrae'] + '/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres'
            model_folds = 'all'
        elif model == 'ResEncL':
            logging.info('  using standard model: ResEncL, fold=[0,1,2,3,4]')
            model_path = self.config['paths']['weights']['int-vertebrae'] + '/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres'
            model_folds = [0, 1, 2, 3, 4]
        else:
            raise ValueError(f'unknown model: {model}')

        # redirect stdout and stderr to logging
        sl = LoggingWriter(logging.DEBUG)
        with contextlib.redirect_stdout(sl), contextlib.redirect_stderr(sl):

            #
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

            # load predictor, only once
            self.predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=False,
                perform_everything_on_device=True,
                device=pipeline.device,
                verbose=True,
                verbose_preprocessing=True,
                allow_tqdm=False
            )

            self.predictor.initialize_from_trained_model_folder(
                str(model_path),
                use_folds=model_folds,
                checkpoint_name='checkpoint_final.pth',
            )
            

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

            # load metadata from input image
            output_label.meta = input_image.meta

            # transpose image for SITK:
            # models were trained using SimpleITKIO, NiftiDataContainer currently uses nibabel.
            # ToDo: use SimpleITKIO for all nifti operations, also in NiftiDataContainer
            # but: TotalSegmentator only takes Nifti1Imag directly -> transformations there, or only paths
            tmp_img = input_image.data_np.transpose((2, 1, 0))[None]
            tmp_props = {'spacing': [float(i) for i in input_image.spacing[::-1]]}
            
            # do segmentation
            logging.info(f' running segmentation using nnUNetv2|internal-vertebrae')
            sl = LoggingWriter(logging.DEBUG)
            with contextlib.redirect_stdout(sl), contextlib.redirect_stderr(sl):

                tmp_segm = self.predictor.predict_single_npy_array(tmp_img, tmp_props, None, None, False)
                log_gpu_usage()

            # revert transpose, save segmentation
            output_label.data_np = tmp_segm.transpose((2, 1, 0))

            # logging
            logging.info(f' finished segmentation ({time() - time_start:.2f}s)')
            logging.info(f' output: memory:{output_label.path}')

            # saving
            if self.config['segmentation']['save_label']:
                output_label.save_to_file()
                logging.info(f' saved file')
