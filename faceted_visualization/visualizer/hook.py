from collections import OrderedDict

import torch.nn
from typing import List, Union, Callable
import logging

logger = logging.getLogger()


class Hook:

    def __init__(self, name, module):
        self.hook_name = name
        self.hook = module.register_forward_hook(self.capture_features)
        self.tensor_hooks = dict()
        self.module = None
        self.input = None
        self.output = None

    def capture_features(self, module, input, output):
        self.module = module
        self.input = input[0]
        self.output = output

    #         print('{0!s: <24} | {1!s: <30} | {2!s: <30}'.format(self.hook_name, input[0].shape, output.shape))
    #         self.tensor_hooks['input'] = {
    #             'hook': input[0].register_hook(self.capture_input_gradient)
    #         }
    #         self.tensor_hooks['output'] = {
    #             'hook': output.register_hook(self.capture_output_gradient)
    #         }

    def capture_input_gradient(self, grad):
        self.tensor_hooks['input']['gradient'] = grad

    def capture_output_gradient(self, grad):
        self.tensor_hooks['output']['gradient'] = grad


def register_hooks(model: torch.nn.Module) -> Callable:
    logger.info("Registering hooks.")
    features = OrderedDict()

    def traverse(module: torch.nn.Module, prefix: Union[List, None] = None):
        if prefix is None:
            prefix = []
        if hasattr(module, "_modules"):
            for name, sub_module in module._modules.items():
                if sub_module is not None:
                    hook_name = "_".join(prefix + [name])
                    features[hook_name] = Hook(hook_name, sub_module)
                    traverse(sub_module, prefix=prefix + [name])
                if isinstance(sub_module, torch.nn.ReLU):
                    sub_module.inplace = False

    traverse(model)
    logger.info("Finished registering hooks.")

    def hook(layer, raw: bool = False):
        hook_obj = features[layer]
        if raw:
            return hook_obj
        else:
            return hook_obj.output

    return hook
