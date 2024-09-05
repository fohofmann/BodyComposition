# libraries
import logging
import sys
import torch.cuda as cuda
from psutil import virtual_memory
from pathlib import Path
import yaml

# function to initialze logging handlers (file and console)
def init_logging(file: str = None, level_file: int = logging.INFO, level_console: int = logging.WARNING):

    # create dir if not existing
    Path(file).parent.mkdir(parents=True, exist_ok=True)

    # create handlers
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(file)
    stream_handler.setLevel(level_console)
    file_handler.setLevel(level_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # record everything
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

# class to redirect logging to a file
class LoggingWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.rstrip() != "":
            logging.log(self.level, message.rstrip())

    def flush(self):
        pass

# function for logging GPU usage
def log_gpu_usage(device=None):
    memory_info = virtual_memory()
    msg = ' usage:'
    msg += f' RAM {round(memory_info.used/(1024**3),1)}/{round(memory_info.total/(1024**3),1)}GB'
    if cuda.is_available():
        msg += f' | VRAM {round(cuda.max_memory_allocated(device)/(1024**3),1)}/{round(cuda.max_memory_reserved(device)/(1024**3),1)}GB'
    logging.info(msg)

# function for license messages
def log_license(licenses):

    # define meassges
    with open(Path('./BodyComposition/utils/licenses.yaml')) as f:
        dict_licenses = yaml.safe_load(f)

    # generate license text
    text_license = "The configured pipeline is based on frameworks, data or models from multiple authors. You should cite their work, and respect the individual licenses: \n\n"
    for license in licenses:
        if license in dict_licenses:
            license_info = dict_licenses[license]
            text_license += f"{license_info['title']} [{license_info['license']}]: {license_info['citation']}.\n\n"
        else:
            text_license += f"{license}: No license information available.\n\n"

    # log the license messages
    logging.warning(text_license)