import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from adabn_utils import compute_bn_stats, replace_bn_stats, BatchNormStatHook, OutputCopyHook
from functools import partial

def perform_adabn_with_hook(model: nn.Module, target_dataloader: DataLoader) -> nn.Module:
    """
    Alternative implementation using hooks to perform AdaBN.
    
    Args:
        model: The PyTorch model to adapt
        target_dataloader: DataLoader containing samples from the target domain
        
    Returns:
        The adapted model with updated BatchNorm statistics
    """
    model.eval()
    hook = BatchNormStatHook()
    
    # Register hooks for all BatchNorm layers
    with torch.no_grad():

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.train()
                hook_handle = module.register_forward_hook(partial(hook, name=name))
                hook.hooks.append(hook_handle)
                
        # Process target domain data
        device = next(model.parameters()).device
        for data in target_dataloader:
            if isinstance(data, (tuple, list)):
                inputs = data[0].to(device)
            else:
                inputs = data.to(device)
            
            # Prepare inputs in the format expected by the model
            if inputs.dim() == 5:  # Handle video data (B,T,C,H,W)
                # raise unepected behavior
                raise ValueError("Unexpected behavior: inputs.dim() == 5")
            
            model_inputs = {'img': inputs}
            targets = {}
            meta_info = {}
            
            # Forward pass to accumulate statistics
            print("inputs.shape")
            print(inputs.shape)
            model(model_inputs, targets, meta_info, "test")
        for hook_handle in hook.hooks:
            hook_handle.remove()

        for layer_name, stats in hook.bn_stats.items():
            # Average the accumulated statistics
            mean = stats['mean'] / stats['count']
            var = stats['var'] / stats['count']
            hook.bn_stats[layer_name] = {'mean': mean, 'var': var}
        # Remove hooks
    replace_bn_stats(model, hook.bn_stats)
            
    return model

def capture_bn_outputs(model: nn.Module, dataloader) -> dict:
    """
    Captures outputs of all BatchNorm layers during inference.
    
    Args:
        model: The PyTorch model
        dataloader: DataLoader containing input samples
        
    Returns:
        Dictionary containing layer names and their corresponding outputs
    """
    model.eval()
    hook = OutputCopyHook()
    
    # Register hooks only for BatchNorm layers
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                hook_handle = module.register_forward_hook(partial(hook, name=name))
                hook.hooks.append(hook_handle)
        
        # Process data
        device = next(model.parameters()).device
        for data in dataloader:
            if isinstance(data, (tuple, list)):
                inputs = data[0].to(device)
            else:
                inputs = data.to(device)
            
            # Forward pass
            model_inputs = {'img': inputs}
            targets = {}
            meta_info = {}
            model(model_inputs, targets, meta_info, "test")
            
            # We only need one batch to capture the outputs
            break
        
        # Clean up hooks
        hook.unhook()
        
    return hook.outputs