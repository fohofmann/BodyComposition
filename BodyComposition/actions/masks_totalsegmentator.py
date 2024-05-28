# libraries
import logging
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.nifti import NiftiDataContainer
from time import time
from BodyComposition.utils.masks import filter_hu, remove_small_objects, filter_keep_largest, fill_holes
from scipy.ndimage import median_filter as ndi_median_filter
import numpy as np


# action class
class MasksTotalSegmentatorSpine(PipelineAction):
    """Action class for postprocessing spine segmentations derived from TotalSegmentator"""

    def __init__(self, pipeline, reduce_to_vb: bool = True):
        super().__init__(pipeline)

        # io names & to pipeline
        self.input_label_name = 'labels/{caseid}_tseg-spine.nii.gz'
        self.output_mask_name = 'masks/{caseid}_tseg-vertebrae.nii.gz'
        self.io_inputs = [self.input_label_name]
        self.io_outputs = [self.output_mask_name]

        # reduce to vertebral bodies
        if reduce_to_vb:
            self.input_label_vb_name = 'labels/{caseid}_tseg-vertebralbodies.nii.gz'
            self.io_inputs.append(self.input_label_vb_name)
        self.reduce_to_vb = reduce_to_vb

        # create mappings to internal labels
        LBL_VERTEBRALBODIES = pipeline.config['LBL_VERTEBRALBODIES']
        LBL_VERTEBRALBODIES_R = {v: k for k, v in LBL_VERTEBRALBODIES.items()}
        LBL_tseg = {43:'T1',42:'T2',41:'T3',40:'T4',39:'T5',38:'T6',37:'T7',36:'T8',35:'T9',34:'T10',33:'T11',32:'T12',
                    31:'L1',30:'L2',29:'L3',28:'L4',27:'L5',
                    25:'SACRUM',26:'SACRUM'}
        self.mapping_vertebralbodies = {k: LBL_VERTEBRALBODIES_R[v] for k, v in LBL_tseg.items() if v in LBL_VERTEBRALBODIES_R}
        self.LBL_VERTEBRALBODIES_SACRUM = LBL_VERTEBRALBODIES_R["SACRUM"]
        self.LBL_VERTEBRALBODIESONLY = 1

    def __call__(self, memory):
        """Do."""
        super().__call__(memory)
        time_start = time()

        # create output: mask, = empty dc + header from input
        output_mask_path = memory['workspace']/self.output_mask_name.format(caseid=memory['id'])
        output_mask = memory[self.output_mask_name] = NiftiDataContainer(output_mask_path)

        # if mask already available, skip all
        if output_mask.exists() and self.config['run']['skip']:
            logging.info(f' output: {output_mask.path} available, skipping')
            return
        
        # load input: label, spine segmentation
        input_label = memory[self.input_label_name]
        input_label.remap(mapping=self.mapping_vertebralbodies)
        logging.info(f' loaded and remapped {input_label}')

        # copy content and header from input
        output_mask.data_np = input_label.data_np
        output_mask.meta = input_label.meta

        # remove all but vertebral bodies
        if self.reduce_to_vb:
            input_label_vb = memory[self.input_label_vb_name]
            output_mask.data_np[input_label_vb.data_np != self.LBL_VERTEBRALBODIESONLY] = 0 # set everything outside of vertebralbodies to zero
            output_mask.data_np[input_label.data_np == self.LBL_VERTEBRALBODIES_SACRUM] = self.LBL_VERTEBRALBODIES_SACRUM # include sacrum and S1 to mask vertebralbodies
            logging.info(f'  removed everything except vertebral bodies')

        # save output
        logging.info(f' output: memory:{output_mask} ({time()-time_start:.2f}s)')

         # save mask if active
        if self.config['vertebrae']['save_mask']:
            output_mask.save_to_file()
            logging.info(f'  file saved')


# action class
class MasksTotalSegmentatorTissue(PipelineAction):
    """Action class for postprocessing tissue segmentations derived from TotalSegmentator"""

    def __init__(self, pipeline, image: str, iliopsoas: bool = True, bodytrunk: bool = True):
        super().__init__(pipeline)

        # io names & to pipeline
        self.input_label_tissue_name = 'labels/{caseid}_tseg-tissue.nii.gz'
        self.output_mask_name = 'masks/{caseid}_tseg-tissue.nii.gz'
        self.io_inputs = [image, self.input_label_tissue_name]
        self.io_outputs = [self.output_mask_name]

        # input
        self.input_image_name = image

        # separate label for iliopsoas muscle
        if iliopsoas:
            self.input_label_iliopsoas_name = 'labels/{caseid}_tseg-iliopsoas.nii.gz'
            self.io_inputs.append(self.input_label_iliopsoas_name)
        self.iliopsoas = iliopsoas

        # remove extremities label from tissue segmentation
        if bodytrunk:
            self.input_label_bodytrunk_name = 'labels/{caseid}_tseg-bodytrunk.nii.gz'
            self.io_inputs.append(self.input_label_bodytrunk_name)
        self.bodytrunk = bodytrunk

        # load config
        self.config_tissue = self.config['tissue']

        # overwrite default mappings
        self.LBL_TISSUE = pipeline.config['LBL_TISSUE'] = pipeline.config['LBL_TISSUE_TSEG']
        self.LBL_TISSUE_R = {v: k for k, v in self.LBL_TISSUE.items()}
        self.LBL_PSOAS = [88,89]
        self.LBL_BODYTRUNK = 1
        logging.info(f'  tissue labels are mapped to `LBL_TISSUE_TSEG` mapping: {self.LBL_TISSUE}')

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
        # remapping not necessary, as labels using LBL_TISSUE_TSEG
        logging.info(f' load {input_label_tissue}')

        # copy content and header from input
        output_np = input_label_tissue.data_np
        output_mask.meta = input_label_tissue.meta
        
        # iliopsoas
        if self.iliopsoas:
            input_label_iliopsoas = memory[self.input_label_iliopsoas_name]
            tmp_np = np.isin(input_label_iliopsoas.data_np, self.LBL_PSOAS)
            output_np[tmp_np] = self.LBL_TISSUE_R['PSOAS']
            logging.info(f' added new label for PSOAS (={self.LBL_TISSUE_R["PSOAS"]})')

        # if any HU-based filter active:
        if any([self.config_tissue[tissue]['filter_hu'] for tissue in ['imat', 'sm', 'vat', 'sat']]):
            
            # load image np
            input_image = memory[self.input_image_name]
            image_np = input_image.data_np
            logging.debug(f'  HU filter(s) active, loaded image')

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
                                     image_zooms=input_image.spacing,
                                     limit_size_version=self.config_tissue['imat']['filter_size_version'],
                                     limit_size_2D=self.config_tissue['imat']['filter_size_2D'],
                                     limit_size_3D=self.config_tissue['imat']['filter_size_3D'])
                
            # intersection of AT and SMTOTAL (= SM + PSOAS) -> IMAT
            mask_tmp = np.isin(output_np, [self.LBL_TISSUE_R['SM'],self.LBL_TISSUE_R['PSOAS']]) & mask_tmp
            output_np[mask_tmp] = self.LBL_TISSUE_R['IMAT']
            logging.info(f"  positive filter -> added new label for IMAT (={self.LBL_TISSUE_R['IMAT']})")


        # filter: skeletal muscle SM
        if self.config_tissue['sm']['filter_hu']:
            logging.info(f" HU filter muscle compartment(s)")
            mask_tmp = filter_hu(image_np, self.config_tissue['sm']['filter_hu_range'])
            mask_tmp_not = np.isin(output_np, [self.LBL_TISSUE_R['SM'], self.LBL_TISSUE_R['PSOAS']]) & np.logical_not(mask_tmp)
            if self.config_tissue['sm']['filter_hu_size']:
                remove_small_objects(mask_np = mask_tmp_not,
                                     image_zooms=input_image.spacing,
                                     limit_size_version=self.config_tissue['sm']['filter_hu_size_version'],
                                     limit_size_2D=self.config_tissue['sm']['filter_hu_size_2D'],
                                     limit_size_3D=self.config_tissue['sm']['filter_hu_size_3D'])
            output_np[mask_tmp_not] = 0
            logging.info(f"  negative filter -> SM")


        # filter: visceral adipose tissue VAT 
        if self.config_tissue['vat']['filter_hu']:
            logging.info(f" HU filter visceral compartment")
            mask_tmp = filter_hu(image_np, self.config_tissue['vat']['filter_hu_range'])
            mask_tmp_not = np.isin(output_np, self.LBL_TISSUE_R['VAT']) & np.logical_not(mask_tmp)
            if self.config_tissue['vat']['filter_hu_size']:
                remove_small_objects(mask_np = mask_tmp_not,
                                     image_zooms=input_image.spacing,
                                     limit_size_version=self.config_tissue['vat']['filter_hu_size_version'],
                                     limit_size_2D=self.config_tissue['vat']['filter_hu_size_2D'],
                                     limit_size_3D=self.config_tissue['vat']['filter_hu_size_3D'])
            output_np[mask_tmp_not] = 0
            logging.info(f"  negative filter -> VAT")


        # filter: subcutaneous adipose tissue SAT
        if self.config_tissue['sat']['filter_hu']:
            logging.info(f" HU filter subcutaneous compartment")
            mask_tmp = filter_hu(image_np, self.config_tissue['sat']['filter_hu_range'])
            mask_tmp_not = np.isin(output_np, self.LBL_TISSUE_R['SAT']) & np.logical_not(mask_tmp)
            if self.config_tissue['sat']['filter_hu_size']:
                remove_small_objects(mask_np = mask_tmp_not,
                                     image_zooms=input_image.spacing,
                                     limit_size_version=self.config_tissue['sat']['filter_hu_size_version'],
                                     limit_size_2D=self.config_tissue['sat']['filter_hu_size_2D'],
                                     limit_size_3D=self.config_tissue['sat']['filter_hu_size_3D'])
            output_np[mask_tmp_not] = 0
            logging.info(f"  negative filter -> SAT")


        # remove extremities, ignore everything but bodytrunk
        if self.bodytrunk:
            input_label_bodytrunk = memory[self.input_label_bodytrunk_name]
            tmp_mask = (input_label_bodytrunk.data_np==self.LBL_BODYTRUNK) # 1=bodytrunk, remove other labels
            filter_keep_largest(tmp_mask)
            fill_holes(tmp_mask)
            output_np[np.logical_not(tmp_mask)] = 0
            logging.info(f" removed extremities")

        # logging
        output_mask.data_np = output_np
        logging.info(f' output: memory:{output_mask} ({time()-time_start:.2f}s)')

        # save mask if active
        if self.config_tissue['save_mask']:
            output_mask.save_to_file()
            logging.info(f'  file saved')
