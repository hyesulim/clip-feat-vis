from typing import Dict, Union

import torch.nn
import wandb
import logging
from PIL.Image import Image
import constants

logger = logging.getLogger()


class WandB:

    def __init__(self, enabled: Union[bool, None], api_key: Union[str, None], entity: Union[str, None],
                 run_name: Union[str, None], project: Union[str, None],
                 config: Union[Dict, None], reinit: bool = False, is_sweep: bool = False):

        self.enabled = enabled
        self.api_key = api_key
        self.entity = entity
        self.run_name = run_name
        self.project = project
        self.config = config
        self.reinit = reinit
        self.logged_in = False
        self.is_sweep = is_sweep

        if enabled and (not self.is_sweep):
            self.login()
            self.run = self.setup()

    def login(self):
        logger.info("Logging into WandB.")
        self.logged_in = wandb.login(key=self.api_key)

    def setup(self):
        run = wandb.init(
            entity=self.entity,
            name=self.run_name,
            project=self.project,
            config=self.config,
            reinit=self.reinit,
            config_exclude_keys=[
                constants.WANDB_API_KEY,
                constants.WANDB_RUN_NAME,
                constants.WANDB_PROJECT,
                constants.PATH_OUTPUT,
                constants.PATH_CONFIG_FILE,
                constants.PATH_LINEAR_PROBE,
            ]
        )
        return run

    def start_watch(self, model: torch.nn.Module):
        if (self.logged_in and (self.run is not None)) or self.is_sweep:
            wandb.watch(model, 'all')

    def log_metrics(self, metrics: Dict, step: int = None):
        if (self.logged_in and (self.run is not None)) or self.is_sweep:
            wandb.log(metrics, step)

    def log_image(self, image: Image, name: str, caption: str = None, step: int = None):
        if (self.logged_in and (self.run is not None)) or self.is_sweep:
            wandb.log({name: wandb.Image(image, caption=caption)}, step=step)

    def close(self):
        if (self.logged_in and (self.run is not None)) and (not self.is_sweep):
            self.run.finish()
            self.run = None
            self.logged_in = False


