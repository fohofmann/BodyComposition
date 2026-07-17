# libraries
import logging
from BodyComposition.pipeline import PipelineAction
from BodyComposition.tissue import cleanup_tissue_mask, prepare_classification_images
from BodyComposition.utils.geometry import assert_same_physical_domain
from BodyComposition.utils.nifti import NiftiDataContainer
from time import time
from BodyComposition.utils.masks import filter_hu, filter_keep_largest, fill_holes
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
        self.io_reset_outputs = [self.output_mask_name]
        if self.config['vertebrae']['save_mask']:
            self.io_persisted_outputs = [self.output_mask_name]

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
        input_label.validate()
        output_mask.meta = input_label.meta
        output_mask.data = input_label.data
        output_mask.remap(mapping=self.mapping_vertebralbodies)
        logging.info(f' loaded {input_label} and remapped output labels')

        # remove all but vertebral bodies
        if self.reduce_to_vb:
            input_label_vb = memory[self.input_label_vb_name]
            input_label_vb.validate()
            assert_same_physical_domain(
                input_label.geometry,
                input_label_vb.geometry,
                reference_name="TotalSegmentator spine label",
                candidate_name="TotalSegmentator vertebral-body label",
            )
            sacrum_mask = output_mask.data == self.LBL_VERTEBRALBODIES_SACRUM
            output_mask.data[input_label_vb.data != self.LBL_VERTEBRALBODIESONLY] = 0 # set everything outside of vertebralbodies to zero
            output_mask.data[sacrum_mask] = self.LBL_VERTEBRALBODIES_SACRUM # include sacrum and S1 to mask vertebralbodies
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
        self.io_reset_outputs = [self.output_mask_name]

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
        if self.config_tissue['save_mask']:
            self.io_persisted_outputs = [self.output_mask_name]

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
        input_image = memory[self.input_image_name]
        input_label_tissue.validate()
        input_image.validate()
        assert_same_physical_domain(
            input_image.geometry,
            input_label_tissue.geometry,
            reference_name="input image",
            candidate_name="TotalSegmentator tissue label",
        )
        logging.info(f' load {input_label_tissue}')
        logging.debug(f'  tissue: origin={input_label_tissue.origin}, shape={input_label_tissue.shape}')

        # copy content and header from input
        output_np = input_label_tissue.data.copy()
        output_mask.meta = input_label_tissue.meta
        
        # iliopsoas
        if self.iliopsoas:
            input_label_iliopsoas = memory[self.input_label_iliopsoas_name]
            input_label_iliopsoas.validate()
            assert_same_physical_domain(
                input_label_tissue.geometry,
                input_label_iliopsoas.geometry,
                reference_name="TotalSegmentator tissue label",
                candidate_name="TotalSegmentator iliopsoas label",
            )
            tmp_np = np.isin(input_label_iliopsoas.data, self.LBL_PSOAS)
            output_np[tmp_np] = self.LBL_TISSUE_R['PSOAS']
            logging.info(f' added new label for PSOAS (={self.LBL_TISSUE_R["PSOAS"]})')

        # if any HU-based filter active:
        if any([self.config_tissue[tissue]['filter_hu'] for tissue in ['imat', 'sm', 'vat', 'sat']]):
            
            # load image np
            image_by_tissue = prepare_classification_images(
                input_image.data,
                input_image.spacing,
                self.config_tissue['hu_denoise'],
            )
            logging.debug(f'  HU filter(s) active, loaded image')
            logging.debug(f'   image: origin={input_image.origin}, shape={input_image.shape}')

        # filter: intermuscular adipose tissue IMAT; special case as introducing extra label
        if self.config_tissue['imat']['filter_hu']:
            logging.info(f" HU filter muscle compartment")
            compartment = np.isin(
                output_np,
                [self.LBL_TISSUE_R['SM'], self.LBL_TISSUE_R['PSOAS']],
            )
            mask_tmp = compartment & filter_hu(
                image_by_tissue['imat'],
                self.config_tissue['imat']['filter_hu_range'],
            )
            cleanup_tissue_mask(
                mask_tmp,
                self.config_tissue['imat'],
                input_image.spacing,
                support_mask=compartment,
            )
            output_np[mask_tmp] = self.LBL_TISSUE_R['IMAT']
            logging.debug(f"  identified everything in IMAT HU-range within label SM and PSOAS as IMAT (={self.LBL_TISSUE_R['IMAT']})")


        # filter: skeletal muscle SM
        if self.config_tissue['sm']['filter_hu']:
            logging.info(f" HU filter muscle compartment(s)")
            compartment = np.isin(
                output_np,
                [self.LBL_TISSUE_R['SM'], self.LBL_TISSUE_R['PSOAS']],
            )
            mask_tmp = compartment & filter_hu(
                image_by_tissue['sm'],
                self.config_tissue['sm']['filter_hu_range'],
            )
            cleanup_tissue_mask(
                mask_tmp,
                self.config_tissue['sm'],
                input_image.spacing,
                support_mask=compartment,
            )
            output_np[compartment & ~mask_tmp] = 0
            logging.debug(f"  removed everything out of SM HU-range from label SM and PSOAS")


        # filter: visceral adipose tissue VAT 
        if self.config_tissue['vat']['filter_hu']:
            logging.info(f" HU filter visceral compartment")
            compartment = output_np == self.LBL_TISSUE_R['VAT']
            mask_tmp = compartment & filter_hu(
                image_by_tissue['vat'],
                self.config_tissue['vat']['filter_hu_range'],
            )
            cleanup_tissue_mask(
                mask_tmp,
                self.config_tissue['vat'],
                input_image.spacing,
                support_mask=compartment,
            )
            output_np[compartment & ~mask_tmp] = 0
            logging.debug(f"  removed everything out of VAT HU-range from label VAT")


        # filter: subcutaneous adipose tissue SAT
        if self.config_tissue['sat']['filter_hu']:
            logging.info(f" HU filter subcutaneous compartment")
            compartment = output_np == self.LBL_TISSUE_R['SAT']
            mask_tmp = compartment & filter_hu(
                image_by_tissue['sat'],
                self.config_tissue['sat']['filter_hu_range'],
            )
            cleanup_tissue_mask(
                mask_tmp,
                self.config_tissue['sat'],
                input_image.spacing,
                support_mask=compartment,
            )
            output_np[compartment & ~mask_tmp] = 0
            logging.debug(f"  removed everything out of SAT HU-range from label SAT")


        # remove extremities, ignore everything but bodytrunk
        if self.bodytrunk:
            input_label_bodytrunk = memory[self.input_label_bodytrunk_name]
            input_label_bodytrunk.validate()
            assert_same_physical_domain(
                input_label_tissue.geometry,
                input_label_bodytrunk.geometry,
                reference_name="TotalSegmentator tissue label",
                candidate_name="TotalSegmentator body-trunk label",
            )
            tmp_mask = (input_label_bodytrunk.data==self.LBL_BODYTRUNK) # 1=bodytrunk, remove other labels
            filter_keep_largest(tmp_mask)
            fill_holes(tmp_mask)
            output_np[np.logical_not(tmp_mask)] = 0
            logging.info(f" removed extremities")

        # logging
        output_mask.data = output_np
        logging.info(f' output: memory:{output_mask} ({time()-time_start:.2f}s)')

        # save mask if active
        if self.config_tissue['save_mask']:
            output_mask.save_to_file()
            logging.info(f'  file saved')
