import argparse
import os
import random
from typing import Dict

import numpy as np
import torch
import constants, image
import logging

logger = logging.getLogger()


def combine_properties(shell_arguments: argparse.Namespace, properties: dict) -> Dict:
    merged = dict()
    # Shell arguments get higher priority
    for k, v in properties.items():
        merged[k] = v
    for k, v in shell_arguments.__dict__.items():
        if v is not None:
            merged[k] = v
    return merged


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_results(image_array: torch.Tensor, output_directory: str, create_dir: bool = True):
    if create_dir and (not os.path.exists(output_directory)):
        os.makedirs(output_directory)
    output_path = os.path.join(output_directory, "output.jpg")
    logger.info("Saving image [ path = %s ]", output_path)
    pil_image = image.convert_to_PIL(image_array)
    pil_image.save(output_path)