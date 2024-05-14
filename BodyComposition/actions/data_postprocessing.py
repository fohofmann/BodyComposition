# libraries
import logging
from BodyComposition.pipeline import PipelineAction
from time import time
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Union


# action class
class DataCombine(PipelineAction):
    """
    Action class combining, filtering, aggregating and exporting data
    """

    def __init__(self, pipeline):
        super().__init__(pipeline)

        # io to pipeline
        self.io_inputs = ['tmp/tissue_values','tmp/tissue_meta','tmp/vertebrae_values','tmp/vertebrae_meta']
        self.io_outputs = ['tmp/bodycomposition']
        
        # load labels as constants
        self.LBL_VERTEBRALBODIES = pipeline.config['LBL_VERTEBRALBODIES']
        self.LBL_TISSUE = pipeline.config['LBL_TISSUE']


    def __call__(self, memory):
        """Combine, filter, aggregate and exporting data."""
        super().__call__(memory)
        time_start = time()

        # checks before combination
        tissue_meta = memory['tmp/tissue_meta']
        vertebrae_meta = memory['tmp/vertebrae_meta']
        if not np.allclose(tissue_meta[0], vertebrae_meta[0]):
            raise ValueError(f'Affines of tissue and vertebrae masks do not match:\n{tissue_meta[0]} vs \n{vertebrae_meta[0]}')
        elif not np.array_equal(tissue_meta[1], vertebrae_meta[1]):
            raise ValueError(f'Shapes of tissue and vertebrae masks do not match:\n{tissue_meta[1]} vs \n{vertebrae_meta[1]}')
        elif not np.allclose(tissue_meta[2], vertebrae_meta[2]):
            raise ValueError(f'Spacings of tissue and vertebrae masks do not match:\n{tissue_meta[2]} vs \n{vertebrae_meta[2]}')

        # transform labels numpy to pandas
        vertebrae_df = pd.DataFrame(memory['tmp/vertebrae_values'], columns=["Slice", "Level", "Center", "Centroid"])
        vertebrae_df["Tag"] = None
           
        # transform csa numpy to pandas
        names_columns = ["CSA_" + col for col in self.LBL_TISSUE.values()]
        tissue_df = pd.DataFrame(memory['tmp/tissue_values'], columns=names_columns)
        tissue_df = tissue_df / 100 # adapt csa values (transform mm2 to cm2)

        # concatenate, reverse order of rows
        results_df = pd.concat([vertebrae_df, tissue_df], axis=1)
        results_df = results_df.iloc[::-1].reset_index(drop=True)

        # replace labels with real levels
        dict_vertebrae = self.LBL_VERTEBRALBODIES
        results_df['Level'] = results_df['Level'].replace(dict_vertebrae)
        results_df['Center'] = results_df['Center'].replace(dict_vertebrae)
        results_df['Centroid'] = results_df['Centroid'].replace(dict_vertebrae)

        # save to memory
        memory['tmp/bodycomposition'] = results_df
        logging.info(f' saved to memory:tmp/bodycomposition ({time()-time_start:.2f}s)')


# action class
class DataSubset(PipelineAction):
    """
    Action class subsetting data
    """

    def __init__(self, pipeline,
                 ref: str, level: Union[str, list],
                 input_df='tmp/bodycomposition', output_df='tmp/bodycomposition'):
        super().__init__(pipeline)

        # io to pipeline
        self.io_inputs = [input_df]
        self.io_outputs = [output_df]
        self.input_df_name = input_df
        self.output_df_name = output_df

        # definition of valid arguments
        valid_ref = ['Center', 'Level', 'Centroid', 'Tag']
        valid_levels = pipeline.config['LBL_VERTEBRALBODIES'].values()
        
        # check subset definitions
        if ref:
            # ref must be a string and included in valid_ref
            if not isinstance(ref, str):
                raise ValueError(f'Argument `ref` must be a string.')
            if ref not in valid_ref:
                raise ValueError(f'Argument `ref` must be {valid_ref}.')
            
            # level must be a string or list and included in valid_levels
            if isinstance(level, str):
                if level == 'ALL':
                    level = valid_levels
                    logging.info(f' subset: `ALL` = all valid levels, excluding undefinied levels')
                elif level == 'L':
                    level = [x for x in valid_levels if x.startswith('L')]
                    logging.info(f' subset: `L` = all lumbar levels')
                else:
                    logging.info(f' subset: `{level}`.')
                    level = [level]
            elif not isinstance(level, list):
                raise ValueError(f'Argument `level` must be a string or list.')

            if not all(x in valid_levels for x in level):
                raise ValueError(f'Argument `level` includes undefined structures.')
        else:
            logging.info(f' subset: not defined = all data, including undefinied levels.')
        
        # set subset setting variables
        self.ref = ref
        self.level = level

    def __call__(self, memory):
        """Do."""
        super().__call__(memory)
        time_start = time()

        # load df
        output_df = memory[self.input_df_name]

        if not isinstance(output_df, pd.DataFrame):
            raise ValueError(f'Input must be a pandas dataframe.')
        elif output_df.empty:
            logging.info(f' skipped: no data')
        elif not self.ref:
            logging.info(f' skipped: inactive')
        else:
            output_df = output_df[output_df[self.ref].isin(self.level)]
            logging.info(f' subset: finished ({time()-time_start:.2f}s)')
        
        # save to memory
        memory[self.output_df_name] = output_df
        logging.info(f' output: memory:{self.output_df_name} ({time()-time_start:.2f}s)')


# action class
class DataAggregate(PipelineAction):
    """
    Action class for aggregating data
    """

    def __init__(self, pipeline,
                 method: str=None, ref: str=None, tag_mapping: dict=None,
                 input_df='tmp/bodycomposition', output_df='tmp/bodycomposition'):
        super().__init__(pipeline)

        # io to pipeline
        self.io_inputs = [input_df]
        self.io_outputs = [output_df]
        self.input_df_name = input_df
        self.output_df_name = output_df

        # definition of valid arguments
        valid_method = ['mean', 'median', 'sum']
        valid_ref = ['Center', 'Level', 'Centroid', 'Tag']
        valid_levels = pipeline.config['LBL_VERTEBRALBODIES'].values()

        # check aggregate definitions
        if method:
            # method must be a string and included in valid_origins
            if method not in valid_method:
                raise ValueError(f'Argument `method` must be a valid pandas.agg function (e.g., `mean`, `median` or `sum`).')

            if ref:
                # ref must be a string and included in valid_origins
                if ref not in valid_ref:
                    raise ValueError(f'Argument `ref` must be {valid_ref}.')
                                                 
                # check aggregate groups
                if ref == "Tag":
                    if not isinstance(tag_mapping, dict):
                        raise ValueError(f'If grouping by a new tag, argument `tag_mapping` must be a dictionary matching `ref` (=key) and a user-specified tag (=value).')
                    if not all(key in valid_levels for key in tag_mapping.keys()):
                        raise ValueError(f'Invalid key for grouping. Valid keys are {valid_levels}')
                    logging.info(f'  using user-specified tag: levels = `{pd.Series(tag_mapping.values()).unique()}`')
                    logging.info(f'  aggregation: method `{method}`, grouped by user-specified tag')
                else:
                    logging.info(f'  aggregation: method `{method}`, grouped by `{ref}`')
            else:
                logging.info(f'  aggregation: method `{method}`, no group -> grouping at patient level')

        # set aggregate variables
        self.ref = ref
        self.tag_mapping = tag_mapping
        self.method = method

    def __call__(self, memory):
        """Do."""
        super().__call__(memory)
        time_start = time()

        # load df
        input_df = memory[self.input_df_name]
    
        # checks 
        if not isinstance(input_df, pd.DataFrame):
            raise ValueError(f' input must be a pandas dataframe.')
        elif input_df.empty:
            output_df = input_df
            logging.info(f' skipped: no data')
        elif not self.method:
            output_df = input_df
            logging.info(f' skipped: inactive')
        else:

            # only aggregate measurements
            col_wo_aggregation = ['Slice', 'Center', 'Level', 'Centroid', 'Tag']
            def single_or_none(x):
                return x.iloc[0] if x.nunique() == 1 else None
            agg_dict = {col: single_or_none if col in col_wo_aggregation else self.method for col in input_df.columns}

            # without grouping -> aggregate complete patient
            if not self.ref:
                output_df = input_df.agg(agg_dict).to_frame().T

            # with tag_mapping -> aggregate by new tag
            else:
                # if activated, create new NewTag
                if self.tag_mapping:
                    for key, value in self.tag_mapping.items():
                        input_df.loc[input_df[self.ref] == key, 'Tag'] = value
                # aggregate
                output_df = input_df.groupby(self.ref).agg(agg_dict).reset_index(drop=True)

        # round all float columns
        output_df = output_df.round(2)
            
        # save to memory
        memory[self.output_df_name] = output_df
        logging.info(f' output: memory:{self.output_df_name} ({time()-time_start:.2f}s)')



# action class
class DataExport(PipelineAction):
    """
    Action class for exporting data
    """

    def __init__(self, pipeline,
                 input: str = 'tmp/bodycomposition',
                 file: str = 'exports/{caseid}_bc_raw.csv',
                 append: bool = False,
                 add_metadata: bool = False):
        super().__init__(pipeline)

        # io to pipeline
        self.io_inputs = [input]
        self.input_df_name = input
        self.output_df_name = file

        # excemption for append
        if not append:
            self.io_outputs = [file]
        else:
            self.io_outputs = []
            logging.info(f'  appending to {file}, this file will be ignored in reset and io-checks.')

        if add_metadata:
            self.io_inputs.append('tmp/metadata')

        # save timestamp to action
        self.timestamp = pipeline.timestamp
        self.add_metadata = add_metadata
        self.append = append


    def __call__(self, memory):
        """do."""
        super().__call__(memory)
        time_start = time()

        if self.add_metadata:

            # save variables
            n_rows = memory[self.input_df_name].shape[0]
            slicethickness = memory.get('slicethickness', None)

            # load metadata
            patients_df = pd.DataFrame({
                'case_id': [memory['id']]*n_rows,
                'case_timestamp': [self.timestamp]*n_rows,
                'pat_id': [memory['tmp/metadata']['pat_id']]*n_rows,
                'pat_prefix': [memory['tmp/metadata']['pat_prefix']]*n_rows,
                'pat_suffix': [memory['tmp/metadata']['pat_suffix']]*n_rows,
                'pat_sex': [memory['tmp/metadata']['pat_sex']]*n_rows,
                'pat_size': [memory['tmp/metadata']['pat_size']]*n_rows,
                'pat_weight': [memory['tmp/metadata']['pat_weight']]*n_rows,
                'scan_date': [memory['tmp/metadata']['scan_date']]*n_rows,
                'scan_slicethickness': [slicethickness]*n_rows
                })
            
            # concatenate
            results_df = pd.concat([patients_df, memory[self.input_df_name]], axis=1)
        
        else:
            results_df = memory[self.input_df_name]

        # export to file
        path_output = Path(memory['workspace'], self.output_df_name.format(caseid=memory['id']))
        path_output.parent.mkdir(parents=True, exist_ok=True)
        if not path_output.exists() or not self.append:
            results_df.to_csv(path_output, index=False)
        else:
            results_df.to_csv(path_output, mode='a', header=False, index=False)

        logging.info(f' output: file:{path_output} ({time()-time_start:.2f}s)')
