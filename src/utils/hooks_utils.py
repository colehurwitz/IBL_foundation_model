import os
from typing import List, Union, Optional
from dataclasses import dataclass

import torch
from torch import nn
from src.models.itransformer import CustomTransformerEncoderLayer
from torchvision.ops import MLP
import numpy as np

####################################################################################################
# iTransformer Hooks
####################################################################################################

# --------------------------------------------------------------------------------------------------
# 1. Forward hooks
# --------------------------------------------------------------------------------------------------

# 1.1. Attention weights

class AttentionWeightsHook:
    def __init__(self):
        self.attention_weights = {}
        self.module_id_to_name = {}

    def _hook(self, module, input, output):
        # Assuming that the attention weights are in the output and are the second item in a tuple
        attention_weights = output[1].detach().cpu().numpy()
        module_name = self.module_id_to_name[id(module)]
        if module_name not in self.attention_weights:
            self.attention_weights[module_name] = []
        self.attention_weights[module_name].append(attention_weights)

    def register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                module.register_forward_hook(self._hook)
                self.module_id_to_name[id(module)] = name
                print(f'Hook registered for {name}')

    def clear(self):
        self.attention_weights = {}


# --------------------------------------------------------------------------------------------------
# 2. Hook for the output of each transformer layer
# --------------------------------------------------------------------------------------------------

class TransformerLayerOutputHook:
    def __init__(self):
        self.layer_results = {}
        self.module_id_to_name = {}

    def _hook(self, module, input, output):
        layer_output = output[0].detach().cpu().numpy()
        module_name = self.module_id_to_name[id(module)]
        if module_name not in self.layer_results:
            self.layer_results[module_name] = []
        self.layer_results[module_name].append(layer_output)

    def register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, CustomTransformerEncoderLayer):
                module.register_forward_hook(self._hook)
                self.module_id_to_name[id(module)] = name
                print(f'Hook registered for {name}')

    def clear(self):
        self.layer_results = {}


# --------------------------------------------------------------------------------------------------
# 3. Hook for the input of each transformer layer
# --------------------------------------------------------------------------------------------------

class TransformerLayerInputHook:
    def __init__(self):
        self.layer_results = {}
        self.module_id_to_name = {}

    def _hook(self, module, input, output):
        layer_input = input[0].detach().cpu().numpy()
        module_name = self.module_id_to_name[id(module)]
        if module_name not in self.layer_results:
            self.layer_results[module_name] = []
        self.layer_results[module_name].append(layer_input)

    def register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, CustomTransformerEncoderLayer):
                module.register_forward_hook(self._hook)
                self.module_id_to_name[id(module)] = name
                print(f'Hook registered for {name}')

    def clear(self):
        self.layer_results = {}







