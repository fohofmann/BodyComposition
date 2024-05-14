# libraries
import logging
from BodyComposition.pipeline import PipelineAction
from time import time
import numpy as np
from scipy.ndimage.measurements import center_of_mass

# action class
class CalcVertebralLevel(PipelineAction):
    """Action class for extracting vertebrae levels from spine segmentation."""

    def __init__(self, pipeline, mask: str):
        super().__init__(pipeline)

        # io to pipeline
        self.input_mask_name = mask
        self.io_inputs = [mask]
        self.io_outputs = ['tmp/vertebrae_values', 'tmp/vertebrae_meta']

        # configs
        self.config_vertebrae = pipeline.config['vertebrae']


    # max counts per slive, with and without window
    def get_max_counts(self, data: np.array, index: int, window_size: int, min_voxels: int = 0, deprioritize_labels: list = []):

        # limit window size to data size
        if index-window_size < 0:
            helper_start = 0
        else:
            helper_start = index-window_size

        if index+window_size+1 > data.shape[2]:
            helper_end = data.shape[2]
        else:
            helper_end = index+window_size+1

        # select window 
        indices = range(helper_start, helper_end)
        data_window = data[:, :, indices]

        # count values
        values, counts = np.unique(data_window, return_counts=True)

        # ignore 0s as maximums, but keep them if nothing else is there 
        counts[values == 0] = 0

        # ignore values with less than min_voxels
        counts[counts < min_voxels] = 0

        # ignore labels
        for deprioritize_label in deprioritize_labels:
            counts[values == deprioritize_label] = 1

        # return max counts
        return values[np.argmax(counts)]
    

    # identify non-monotonical vertebrae
    # standard RAS+ = inferior to superior, vertebrae cranio-caudal
    def get_not_monotonical(self, vertebrae: np.array):
        # empty array for indices
        indices = []

        # loop through verterbrae
        for i, value in enumerate(vertebrae):
                
            # skip if first slice
            if i == 0:
                continue
    
            # check if not monotonical
            if value > vertebrae[i-1]:
                indices.append(i)
        
        return indices


    def __call__(self, memory):
        """Segment case."""
        super().__call__(memory)
        time_start = time()

        # load mask
        input_mask = memory[self.input_mask_name]
        input_mask.as_closest_canonical()
        logging.info(f' loaded {input_mask}, reorientated to canonical')
        
        # create empty output array
        # RAS+ format
        #   rows: 2 = inferior to superior, starting w 0
        #   4 columns: 0=slice, 1=dominating vertebrae, 2=median of distr in cranio-caudal direction, 3=vertebrea center of mass
        res_labels_np = np.zeros(shape=(input_mask.shape[-1], 4), dtype=np.uint16) 
        res_labels_np[:, 0] = range(input_mask.shape[-1]) # column 0: slice

        # load vertrebrae as numpy
        time_start_i = time()
        labels_all = res_labels_np[:, 1] # column 1: dominating vertebrae

        # STEP 1: find value with max counts without using windows
        for z in range(input_mask.shape[2]):
            labels_all[z] = self.get_max_counts(input_mask.data_np, z, 0, self.config_vertebrae['min_voxels_per_vertebra'], self.config_vertebrae['deprioritize_labels'])
        logging.info(f' computed dominating vertebrae levels ({time() - time_start_i:.1f}s)')

        # vertebrae between first and last defined level to subarray
        helper_localizer = np.where(labels_all != 0)[0]
        if helper_localizer.size == 0:
            raise ValueError(f' No vertebrae found in {self.input_mask_name}. A common cause is, that the image is not properly aligned or oriented.')
        labels_vertebrae = labels_all[helper_localizer[0]:helper_localizer[-1]+1]
        data_vertebrae = input_mask.data_np[:, :, helper_localizer[0]:helper_localizer[-1]+1]

        # STEP 2: fill undefined levels between first and last defined level
        if self.config_vertebrae['fill_undefined_levels']:
            time_start_i = time()
            helper_unlabeled = np.where(labels_vertebrae == 0)[0]
            helper_windowsize = 1

            while helper_unlabeled.size > 0:
                for z in helper_unlabeled:
                    labels_vertebrae[z] = self.get_max_counts(data_vertebrae, z, helper_windowsize, self.config_vertebrae['min_voxels_per_vertebra'], self.config_vertebrae['deprioritize_labels'])
                helper_unlabeled = np.where(labels_vertebrae == 0)[0]
                helper_windowsize += 1
            
            if helper_windowsize > 1:
                logging.info(f' filled undefined vertebrae levels ({time() - time_start_i:.1f}s)')

        # STEP 3: check and correct monotonicality
        if self.config_vertebrae['correct_monotonicity']:
            time_start_i = time()
            helper_notmonotonical = self.get_not_monotonical(labels_vertebrae)
            helper_windowsize = 1

            while len(helper_notmonotonical) > 0:
                for z in helper_notmonotonical:

                    # define range (helper_start / helper_end) depending on window size
                    if z-helper_windowsize < 0:
                        helper_start = 0
                    else:
                        helper_start = z-helper_windowsize

                    if z+helper_windowsize+1 > labels_vertebrae.size:
                        helper_end = labels_vertebrae.size
                    else:
                        helper_end = z+helper_windowsize+1

                    # loop through range and find max counts
                    for z2 in range(helper_start, helper_end):
                        labels_vertebrae[z2] = self.get_max_counts(data_vertebrae, z2, helper_windowsize, self.config_vertebrae['min_voxels_per_vertebra'], self.config_vertebrae['deprioritize_labels'])

                helper_notmonotonical = self.get_not_monotonical(labels_vertebrae)
                helper_windowsize += 1

                # set limit
                if helper_windowsize > self.config_vertebrae['correct_monotonicity_max_windowsize']:
                    break

            if helper_windowsize > 1:
                logging.info(f' corrected non-monotonical vertebrae levels ({time() - time_start_i:.1f}s)')

        # STEP 4: find center of weight for each vertebrae
        time_start_i = time()

        # load vertebrae data and output as numpy
        labels_all = res_labels_np[:, 2] # column 2: vertebrae center

        # select all identified vertrebrae
        vertebrae = np.unique(input_mask.data_np)
        vertebrae = vertebrae[vertebrae != 0]

        # loop through all vertrebrae, find median of voxel distribution in cranio-caudal direction
        for vertebra in vertebrae:
            helper_counts = np.sum(input_mask.data_np == vertebra, axis=(0, 1))
            helper_cumsum = np.cumsum(helper_counts)
            helper_total = np.sum(helper_counts)
            helper_index = np.where(helper_cumsum >= helper_total/2)[0][0]

            # skip if not dominating vertebrae
            if res_labels_np[int(helper_index),1] != vertebra:
                continue

            labels_all[int(helper_index)] = vertebra # column 2 in results: vertebrae center
        logging.info(f' vertebrae centers (cranio-caudal = z-axis) calculated ({time() - time_start_i:.1f}s).')

        # STEP 5: find centroid / center of mass for each vertrebra of vertebrae
        time_start_i = time()

        # load vertebrae data and output as numpy
        labels_all = res_labels_np[:, 3] # column 3: vertebrae center of mass / centroid

        # select all identified vertrebrae, not needed as already done in step 4
        # vertebrae = np.unique(input_mask.data_np)
        # vertebrae = vertebrae[vertebrae != 0]

        # loop through all vertrebrae, find center of mass in cranio-caudal direction
        if self.config_vertebrae['center_of_mass']:
            for vertebra in vertebrae:
                _, _, helper_index = center_of_mass(input_mask.data_np == vertebra)
                helper_index = int(round(helper_index))
                # skip if not dominating vertebrae
                if res_labels_np[helper_index,1] != vertebra:
                    continue
                labels_all[helper_index] = vertebra
            logging.info(f' vertebrae center of mass (cranio-caudal = z-axis) calculated ({time() - time_start_i:.1f}s).')
        else:
            labels_all[:] = np.nan
            
        # save data to pipeline
        memory['tmp/vertebrae_values'] = res_labels_np
        memory['tmp/vertebrae_meta'] = input_mask.meta
        logging.info(f' output: memory:tmp/vertebrae_values, shape {res_labels_np.shape} ({time()-time_start:.2f}s)') 