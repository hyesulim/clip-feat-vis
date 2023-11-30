import ast
import sys
import logging
import constants
import os
import datetime

now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
run_id = f"run_{now}"

logging.basicConfig(force=True, level="INFO",
                    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()
init_stream = True

for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        init_stream = False
if init_stream:
    console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(console_handler)


def add_file_handler(properties):
    log_directory = os.path.join(properties[constants.PATH_OUTPUT], run_id)
    init_filehandler = True
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            init_filehandler = False
            break
    if init_filehandler:
        file_handler = logging.FileHandler(os.path.join(log_directory, "output.log"))
        file_handler.setFormatter(
            logging.Formatter(fmt="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(file_handler)
