# libraries
import logging
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.geometry import assert_same_physical_domain
from BodyComposition.utils.nifti import NiftiDataContainer
from time import time
from BodyComposition.utils.masks import filter_hu, remove_small_objects
from scipy.ndimage import median_filter as ndi_median_filter
import numpy as np


# action class
class MasksBoaTissue(PipelineAction):
    """Action class for postprocessing tissue segmentations derived from Stanford Tissue model"""

    def __init__(self, pipeline, image: str):
        super().__init__(pipeline)

        # io names & to pipeline
        self.input_label_tissue_name = 'labels/{caseid}_boa.nii.gz'
        self.output_mask_name = 'masks/{caseid}_boa.nii.gz'
        self.io_inputs = [image, self.input_label_tissue_name]
        self.io_outputs = [self.output_mask_name]
        self.io_reset_outputs = [self.output_mask_name]

        # input
        self.input_image_name = image

        # load config
        self.config_tissue = self.config['tissue']
        if self.config_tissue['save_mask']:
            self.io_persisted_outputs = [self.output_mask_name]

        # overwrite default mappings
        self.LBL_TISSUE = pipeline.config['LBL_TISSUE'] = pipeline.config['LBL_TISSUE_BOA']
        self.LBL_TISSUE_R = {v: k for k, v in self.LBL_TISSUE.items()}
        logging.info(f'  tissue labels are mapped to `LBL_TISSUE_BOA` mapping: {self.LBL_TISSUE}')

    def __call__(self, memory):
        """Segment case."""
        super().__call__(memory)
        time_start = time()

        # create output: mask, = empty dc + header from input
        output_mask_path = memory['workspace']/self.output_mask_name.format(caseid=memory['id'])
        output_mask = memory[self.output_mask_name] = NiftiDataContainer(output_mask_path)

        # if mask already available, skip all
        if output_mask.exists() and self.config['run']['skip']:
            logging.info(f' output: {output_mask.path} available, skipping')
            return

        # load input: label, tissue segmentation
        input_label_tissue = memory[self.input_label_tissue_name]
        input_image = memory[self.input_image_name]
        input_label_tissue.validate()
        input_image.validate()
        assert_same_physical_domain(
            input_image.geometry,
            input_label_tissue.geometry,
            reference_name="input image",
            candidate_name="BOA tissue label",
        )
        logging.info(f' load {input_label_tissue}')
        logging.debug(f'  tissue: origin={input_label_tissue.origin}, shape={input_label_tissue.shape}, direction={input_label_tissue.direction}')

        # copy content and header from input
        output_np = input_label_tissue.data.copy()
        output_mask.meta = input_label_tissue.meta

        # if any HU-based filter active:
        if any([self.config_tissue[tissue]['filter_hu'] for tissue in ['imat', 'sm', 'vat', 'sat']]):
            
            # load image np
            image_np = input_image.data
            logging.debug(f'  HU filter(s) active, loaded image')
            logging.debug(f'   image: origin={input_image.origin}, shape={input_image.shape}, direction={input_image.direction}')

            # denoising: clip outliers
            if self.config_tissue['hu_denoise']['filter_outliers']:
                hu_range = self.config_tissue['hu_denoise']['filter_outliers_range']
                image_np = np.clip(image_np, min(hu_range), max(hu_range))
                logging.debug(f'  clipped HU values, range {hu_range}')
            
            # denoising: median filter
            if self.config_tissue['hu_denoise']['filter_median']:
                image_np = ndi_median_filter(image_np, size=self.config_tissue['hu_denoise']['filter_median_kernel'])   
                logging.debug(f'  applied HU median filter, kernel={self.config_tissue["hu_denoise"]["filter_median_kernel"]}')

        # filter: intermuscular adipose tissue IMAT; special case as introducing extra label
        if self.config_tissue['imat']['filter_hu']:
            logging.info(f" HU filter muscle compartment")
            # create AT mask, doing this first to account for smaller objects at the edge of SM segmentation
            mask_tmp = filter_hu(image_np, self.config_tissue['imat']['filter_hu_range'])
            if self.config_tissue['imat']['filter_size']:
                remove_small_objects(mask_np = mask_tmp,
                                     spacing_xyz=input_image.spacing,
                                     limit_size_version=self.config_tissue['imat']['filter_size_version'],
                                     limit_size_2D=self.config_tissue['imat']['filter_size_2D'],
                                     limit_size_3D=self.config_tissue['imat']['filter_size_3D'])
                
            # intersection of AT and SM -> add to IMAT
            mask_tmp = np.isin(output_np, [self.LBL_TISSUE_R['SM']]) & mask_tmp
            output_np[mask_tmp] = self.LBL_TISSUE_R['IMAT']
            logging.debug(f"  identified everything in IMAT HU-range within label SM as IMAT (={self.LBL_TISSUE_R['IMAT']})")


        # filter: skeletal muscle SM
        if self.config_tissue['sm']['filter_hu']:
            logging.info(f" HU filter muscle compartment(s)")
            mask_tmp = filter_hu(image_np, self.config_tissue['sm']['filter_hu_range'])
            mask_tmp_not = np.isin(output_np, [self.LBL_TISSUE_R['SM']]) & np.logical_not(mask_tmp)
            if self.config_tissue['sm']['filter_size']:
                remove_small_objects(mask_np = mask_tmp_not,
                                     spacing_xyz=input_image.spacing,
                                     limit_size_version=self.config_tissue['sm']['filter_size_version'],
                                     limit_size_2D=self.config_tissue['sm']['filter_size_2D'],
                                     limit_size_3D=self.config_tissue['sm']['filter_size_3D'])
            output_np[mask_tmp_not] = 0
            logging.debug(f"  removed everything out of SM HU-range from label SM")


        # filter: visceral adipose tissue aVAT and tVAT
        if self.config_tissue['vat']['filter_hu']:
            logging.info(f" HU filter visceral compartment(s)")
            mask_tmp = filter_hu(image_np, self.config_tissue['vat']['filter_hu_range'])
            mask_tmp_not = np.isin(output_np, [self.LBL_TISSUE_R['ABDOMEN'], self.LBL_TISSUE_R['THORAX'], self.LBL_TISSUE_R['MEDIASTINUM']]) & np.logical_not(mask_tmp)
            if self.config_tissue['vat']['filter_size']:
                remove_small_objects(mask_np = mask_tmp_not,
                                     spacing_xyz=input_image.spacing,
                                     limit_size_version=self.config_tissue['vat']['filter_size_version'],
                                     limit_size_2D=self.config_tissue['vat']['filter_size_2D'],
                                     limit_size_3D=self.config_tissue['vat']['filter_size_3D'])
            output_np[mask_tmp_not] = 0
            logging.debug(f"  removed everything out of VAT HU-range from label ABDOMEN, THORAX, MEDIASTINUM")


        # filter: subcutaneous adipose tissue SAT
        if self.config_tissue['sat']['filter_hu']:
            logging.info(f" HU filter subcutaneous compartment")
            mask_tmp = filter_hu(image_np, self.config_tissue['sat']['filter_hu_range'])
            mask_tmp_not = np.isin(output_np, [self.LBL_TISSUE_R['SAT']]) & np.logical_not(mask_tmp)
            if self.config_tissue['sat']['filter_size']:
                remove_small_objects(mask_np = mask_tmp_not,
                                     spacing_xyz=input_image.spacing,
                                     limit_size_version=self.config_tissue['sat']['filter_size_version'],
                                     limit_size_2D=self.config_tissue['sat']['filter_size_2D'],
                                     limit_size_3D=self.config_tissue['sat']['filter_size_3D'])
            output_np[mask_tmp_not] = 0
            logging.debug(f"  removed everything out of SAT HU-range from label SAT")

        # logging
        output_mask.data = output_np
        logging.info(f' output: memory:{output_mask} ({time()-time_start:.2f}s)')

        # save mask if active
        if self.config_tissue['save_mask']:
            output_mask.save_to_file()
            logging.info(f'  file saved')
