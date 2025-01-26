import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import random
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from functools import partial
from tqdm import tqdm



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# Fake dataset class. Trying to be as fake as it can be
class ImageGeneratorDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.vector_dim = (3, 128, 128)
        self.data = []
        self.create_data()

    def create_data(self):
        for i in range(self.num_samples):
            self.data.append(torch.zeros(self.vector_dim))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]




# The hook class is responsible to store the BN outputs when the test dataloader is passed
class BatchNormStatHook(object):
    """
    Hook to accumulate statistics from BatchNorm layers during inference.
    """
    def __init__(self):
        self.bn_stats = {}  # Dictionary toe layer name and accumulated statistics
        self.hooks = []
    def __call__(self, module, input, output, name):
       """
       Hook function called during the forward pass of BatchNorm layers.
       Args:
           module (nn.Module): The BatchNorm layer.
           input (torch.Tensor): Input tensor to the layer.
           output (torch.Tensor): Output tensor from the layer.
       """
       layer_name = name
       if layer_name not in self.bn_stats:
           self.bn_stats[layer_name] = {'mean': torch.zeros_like(module.running_mean), 'var': torch.zeros_like(module.running_var), 'count': 0}
        # Use the input to calculate statistics
       input = input[0] # input is a tuple, we need the first element

       b, c, h, w = input.shape
       input_flattened = input.permute(0, 2, 3, 1).reshape(-1, c)
       mean_input = input_flattened.mean(dim=0)
       var_input = input_flattened.var(dim=0, unbiased=True)
       self.bn_stats[layer_name]['mean'] += mean_input
       self.bn_stats[layer_name]['var'] += var_input
       self.bn_stats[layer_name]['count'] += 1
    def unhook(self):
      for hook in self.hooks:
          hook.remove()
      self.hooks = []


class OutputCopyHook(object):
    def __init__(self):
        self.outputs = {}  # Dictionary to store layer outputs
        self.hooks = []

    def __call__(self, module, input, output, name):
        """
        Hook function called during the forward pass of layers.
        Args:
            module (nn.Module): The layer.
            input (torch.Tensor): Input tensor to the layer.
            output (torch.Tensor): Output tensor from the layer.
            name (str): Name of the layer.
        """
        if name not in self.outputs:
            self.outputs[name] = []
        self.outputs[name].append(output.detach().clone())

    def unhook(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def compute_bn_stats(model, dataloader):
    """
    Computes mean and variance of BatchNorm layer outputs across all images in the dataloader.
    Args:
        model (nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): The dataloader for the data.
    Returns:
        dict: Dictionary containing layer names and their mean and variance statistics.
    """
    # Create a hook instance
    hook = BatchNormStatHook()

    model.eval()
    # Set BatchNorm layers to training mode
    # Iterate through the dataloader

    with torch.no_grad():
        for name, module in model.named_modules():
           if isinstance(module, torch.nn.BatchNorm2d):
               module.train()
               hook_handle = module.register_forward_hook(partial(hook, name=name))
               hook.hooks.append(hook_handle)
        for data in tqdm(dataloader):
            if type(data) == list:
                inputs = {'img': data[0].to(device)}
            else:
                inputs = {'img': data.to(device)}
            targets = {}
            meta_info = {}
            # Forward pass (hook will accumulate statistics)
            model(inputs, targets, meta_info, "test")
    # Calculate mean and variance for each layer
    for layer_name, stats in hook.bn_stats.items():
        # Average the accumulated statistics
        mean = stats['sum'] / stats['count']
        var = stats['var'] / stats['count']
        hook.bn_stats[layer_name] = {'mean': mean, 'var': var}
    # Return the accumulated statistics
    hook.unhook()
    return hook.bn_stats

# Now replace the current stats with the computed one
def replace_bn_stats(model, bn_stats, ratio=1.0, initial_bn_stats=None):
    
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in bn_stats and isinstance(module, nn.BatchNorm2d):
                print(f'Before Apply {name}---------------------------------------')
                print(f'Running Mean: {module.running_mean}')
                print(f'Running Var: {module.running_var}')
                
                # Interpolate between current and new stats based on ratio
                new_mean = (1 - ratio) * module.running_mean + ratio * bn_stats[name]['mean']
                new_var = (1 - ratio) * module.running_var + ratio * bn_stats[name]['var']
                
                module.running_mean.data.copy_(new_mean)
                module.running_var.data.copy_(new_var)
                
                print(f'After Apply {name}---------------------------------------')
                print(f'Running Mean: {module.running_mean}')
                print(f'Running Var: {module.running_var}')
