#!/usr/bin/env python
import argparse
import ast
import os
from pathlib import Path
from BodyComposition.python_api import bodycomposition

# helper functions
def looks_like_dict(s):
    try:
        ast.literal_eval(s)
        return True
    except (SyntaxError, ValueError):
        return False
def looks_like_path(s):
    return os.path.isfile(s)

# main
def main():
    """
    Effectively just a parser using the bodycomposition python API.
    Usage: bin/run_batch.py -i ./data/images -f '^ct_.*\.nii\.gz$' -m BodyCompositionFast
    """

    # parse arguments
    parser = argparse.ArgumentParser(description='Run batch through pipeline.')
    parser.add_argument('--input', '-i', type=str, default='./data/images',
                        help='Path to input, either directory (e.g., `data/images`) or datalist file (*.json) or single NiFTI file.')
    parser.add_argument('--filter', '-f', type=str, default=r'.*\.nii\.gz$',
                        help='Filter to subset input files.')   
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to configuration file (*.yaml), or dictionary. Can be used to update the default configuration.')
    parser.add_argument('--method', '-m', type=str, default='BodyCompositionFast',
                        help='Name of pipeline method to be run.')
    args = parser.parse_args()

    # transform config
    if args.config is None:
        config = None
    elif looks_like_dict(args.config):
        config = ast.literal_eval(args.config)
    elif looks_like_path(args.config):
        config = Path(args.config)
    else:
        config = None

    # run pipeline
    bodycomposition(input = Path(args.input),
                    input_filter = args.filter,
                    config = config,
                    method = args.method)
            
if __name__ == "__main__":
    main() # parser is in main to be available when using pyproject.toml entrypoint
