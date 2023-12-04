import logging
from typing import Callable, Union

import torch
import torchvision.transforms.transforms

import image, wb
import pdb

logger = logging.getLogger()


def optimize(num_iterations: int,
             image_function: Callable,
             transforms: torchvision.transforms.transforms.Compose,
             model: torch.nn.Module,
             channel: int,
             objective: str,
             neuron_x: Union[None, int],
             neuron_y: Union[None, int],
             layer: str,
             linear_probe_layer: str,
             probe_weights: torch.Tensor,
             optimizer: torch.optim.Optimizer,
             model_hook: Callable,
             wandb_object: wb.WandB,
             device: str = "cpu",
             run_id: str = ""):
    logger.info("Starting optimization process [ run_id = %s ]...", run_id)
    #pdb.set_trace()
    for j in range(num_iterations):
        def closure():
            optimizer.zero_grad()
            if transforms is not None:
                model.forward(transforms(image_function()))
            else:
                model.forward(image_function())

            f_x = model_hook(layer, True)
            g_x = model_hook(linear_probe_layer, True)

            gradients = \
                torch.autograd.grad(f_x.output, g_x.output, grad_outputs=torch.ones_like(f_x.output),
                                    retain_graph=True)[0]
            if objective == 'neuron':
                nonlocal neuron_x, neuron_y
                neuron_x = f_x.output.shape[2] // 2 if neuron_x is None else neuron_x
                neuron_y = f_x.output.shape[3] // 2 if neuron_y is None else neuron_y
                loss = -(f_x.output[0, channel-1, neuron_x, neuron_y])
            else:
                loss = -(f_x.output[0, channel-1, :, :].mean())
            loss -= torch.dot(torch.mul(g_x.output, gradients).flatten().type(torch.FloatTensor).to(device),
                              probe_weights.to(device))

            loss.backward()
            if j % 100 == 0:
                logger.info("Epoch: %d/%d: Loss = %.7f :: Learning Rate = %.7f", j + 1, num_iterations, loss,
                            optimizer.param_groups[0]['lr'])
                wandb_object.log_metrics(metrics={'loss': loss, }, step=j)
                wandb_object.log_image(image=image.convert_to_PIL(image_function()), name="render")
            return loss

        optimizer.step(closure)
