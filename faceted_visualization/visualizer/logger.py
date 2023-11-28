import ast
import sys
import logging
from faceted_visualization.visualizer import constants
import os
import datetime


with open("config/run_configs.json") as f:
    properties = ast.literal_eval(f.read())

now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
run_id = f"run_{now}"

log_directory = os.path.join(properties[constants.OUTPUT_DIRECTORY], run_id)

if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logging.basicConfig(force=True, level="INFO",
                    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()
init_stream = True
init_filehandler = True
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        init_stream = False
    if isinstance(handler, logging.FileHandler):
        init_filehandler = False
if init_stream:
    console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(console_handler)
if init_filehandler:
    file_handler = logging.FileHandler(os.path.join(log_directory, "output.log"))
    file_handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"))
    logger.addHandler(file_handler)