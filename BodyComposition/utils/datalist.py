from pathlib import Path
from typing import List, Tuple
import logging
import json
import re

class DatalistBuilder():
    """DatalistBuilder class for loading, filtering and saving data."""

    def __init__(self,
                 input_path: Path = None,
                 input_filter: str = None,
                 workspace: Path = None,
                 io_inputs: List[str] = None,
                 io_outputs: List[str] = None,):
        """Initialize DatalistBuilder class."""

        def stem2(filename: str):
            return filename.split('.')[0]

        logging.info(f'LOADING DATALIST:')

        # check if paths exist
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f'Input file or dir does not exist: {input_path}.')

        # load cases, either from json or from directory
        if input_path.suffix == '.json':
            with open(input_path, 'r') as f:
                case_paths = json.load(f)
            logging.info(f'input: *.json ({input_path}), loading')

            if 'testing' in case_paths:
                case_paths = [i['image'] for i in case_paths['testing']]
                logging.info(f' found `testing` key in json, loading all `image` paths.')
            elif 'imagesTs' in case_paths:
                case_paths = [i['image'] for i in case_paths['imagesTs']]
                logging.info(f' found `imagesTs` key in json, loading all `image` paths.')
            elif all(isinstance(case, str) for case in case_paths):
                logging.info(f' found list of paths, loading directly.')
            elif all(isinstance(case, dict) for case in case_paths):
                case_paths = [i['image'] for i in case_paths]
                logging.info(f' found list of unnamed dictionaries, loading all `image` paths.')
            else:
                raise ValueError(f'Could not parse json: {input_path}.')
            
            logging.info(f' {len(case_paths)} paths found')

            # create absolute paths, check if cases exists
            case_paths = [input_path.parent/file for file in case_paths]
            case_paths = [file for file in case_paths if file.exists()]
            logging.info(f' {len(case_paths)} files found')

        elif input_path.is_dir():
            logging.info(f'input: directory ({input_path}), loading')
            case_paths = [file for file in input_path.iterdir() if file.is_file() and not file.name.startswith('.')]
            logging.info(f' {len(case_paths)} files found')

        elif input_path.is_file():
            case_paths = [input_path]
            logging.info(f'input: single file ({input_path}), loading')

        else:
            raise ValueError(f'Invalid input file or dir: {input_path}.')
        
        # create list of cases: id, input_file, workspace
        cases = [(stem2(file.name), Path(file), Path(workspace) if workspace else file.parent.parent) for file in case_paths]

        # filter inputs; default: r'.*\.nii\.gz$'
        # on purpose not applied earlier to enable filtering irresepective of the input_path datatype
        if input_filter:
            filter_regex = re.compile(input_filter)
            cases = [(caseid, file, workspace) for caseid, file, workspace in cases if filter_regex.match(file.name)]
            logging.info(f' using filter `{input_filter}`: {len(cases)} files remaining')

        # filter regarding requirements io_inputs
        if io_inputs is not None:
            cases = [(caseid, file, workspace) for caseid, file, workspace in cases if all((workspace/io_input.format(caseid=caseid)).exists() for io_input in io_inputs)]
            logging.info(f' regarding required inputs: {len(cases)} files remaining')
        
        # save tuple as attribute
        self.cases = cases
        self.io_outputs = io_outputs
        self.io_inputs = io_inputs
        logging.info(f'identified {len(cases)} cases for processing')

    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        return self.cases[idx]
    
    def __iter__(self):
        return iter(self.cases)

    def reset_outputs(self):
        """Delete existing output files."""
        tmp_cases = set()
        for caseid, input_file, workspace in iter(self.cases):
            for io_output in self.io_outputs:
                tmp_io_output = workspace/io_output.format(caseid=caseid)
                if tmp_io_output.exists():
                    tmp_io_output.unlink()
                    tmp_cases.add(caseid)
        logging.info(f'reset outputs: removed existing files for {len(tmp_cases)} case(s): ({", ".join(tmp_cases)})')

    def skip_completed(self):
        """Remove completed cases."""
        tmp_cases = set()
        for caseid, input_file, workspace in iter(self.cases):
            if all((workspace/io_output.format(caseid=caseid)).exists() for io_output in self.io_outputs):
                tmp_cases.add(caseid)
        if tmp_cases:
            self.cases = [case for case in self.cases if case[0] not in tmp_cases]
        logging.info(f'skip complete cases: removed {len(tmp_cases)} case(s) from datalist ({", ".join(tmp_cases)})')
