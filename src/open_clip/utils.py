from itertools import repeat
import collections.abc

import torch
from torch import nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import FrozenBatchNorm2d

import copy

from enum import Enum
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP


def get_dist_matrix(x, y, scale=1.0, logit_bias=0.0):
    assert x.size(0) == y.size(0), "x and y should have the same batch size"
    assert x.size(-1) == y.size(-1), "x and y should have the same embedding dimension"
    return x @ y.T * scale + logit_bias

def exp_inner_prod(x, y, mean_exp=True):
    inner = (x * y).sum(dim=-1)
    return inner.mean() if mean_exp else inner

    
class NormalizeType(Enum):
    NONE = "NONE"
    L2 = "L2"
    CAPPED_L2 = "CAPPED_L2"
    def __str__(self):
        return self.value
    def get_normalize_fn(self, max_norm=1.0):
        if self == NormalizeType.NONE:
            return lambda x: x
        elif self == NormalizeType.L2:
            return lambda x: F.normalize(x, dim=-1)
        elif self == NormalizeType.CAPPED_L2:
            return lambda x: capped_l2_norm(x, max_norm)


def capped_l2_norm(x, max_norm=1.0):
    sqrt_m = torch.sqrt(torch.tensor(max_norm)) 
    norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
    mask = (norm_x < sqrt_m).float()
    x_normalized = F.normalize(x, dim=-1) * sqrt_m
    return mask * x + (1 - mask) * x_normalized

    
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

class EMA:
    def __init__(self, beta_init=0.99):
        """
        Initializes the EMA (Exponential Moving Average) class.

        Args:
            beta_init (float): The initial beta value for the moving average (higher means slower updates).
        """
        super().__init__()
        self.beta = beta_init

    def update_beta_by_value(self, beta):
        """
        Updates the beta value.

        Args:
            beta (float): New beta value to set.
        """
        self.beta = beta

    def _get_model_from_ddp(self, model):
        """
        If the model is wrapped in DistributedDataParallel, access the underlying model.

        Args:
            model (torch.nn.Module): The model to check.

        Returns:
            torch.nn.Module: The underlying model if wrapped in DDP; otherwise, the original model.
        """
        return model.module if isinstance(model, DDP) else model

    @torch.no_grad()
    def update_average(self, old, new):
        """
        Efficiently updates the moving average of a tensor using in-place operations.

        Args:
            old (torch.Tensor): The previous moving average value.
            new (torch.Tensor): The new value to update with.

        Returns:
            torch.Tensor: Updated moving average value.
        """
        old.mul_(self.beta).add_(new, alpha=1 - self.beta)
        return old

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        """
        Updates the moving average model parameters using the current model parameters.

        Args:
            ma_model (torch.nn.Module or torch.Tensor): The moving average model.
            current_model (torch.nn.Module or torch.Tensor): The current model.
        """
        # Handle DDP models
        current_model = self._get_model_from_ddp(current_model)
        ma_model = self._get_model_from_ddp(ma_model)

        if isinstance(ma_model, torch.nn.Module) and isinstance(current_model, torch.nn.Module):
            # Update model parameters
            for ma_params, current_params in zip(ma_model.parameters(), current_model.parameters()):
                self.update_average(ma_params.data, current_params.data)

            # Update BatchNorm & other buffers (e.g., running_mean, running_var)
            for ma_buffer, current_buffer in zip(ma_model.buffers(), current_model.buffers()):
                self.update_average(ma_buffer.data, current_buffer.data)

        elif isinstance(ma_model, torch.Tensor) and isinstance(current_model, torch.Tensor):
            # Handle direct tensor updates
            self.update_average(ma_model.data, current_model.data)

        else:
            raise TypeError("ma_model and current_model should both be either torch.nn.Module or torch.Tensor")        

def get_ema_model(online_encoder):
    target_encoder = copy.deepcopy(online_encoder.module if isinstance(online_encoder, torch.nn.parallel.DistributedDataParallel) else online_encoder)
    for param in target_encoder.parameters():
        param.requires_grad = False
    return target_encoder


def get_cosine_ema_beta(epoch, beta_decay_epochs, beta_max):
    if epoch < beta_decay_epochs:
        return beta_max - beta_max * 0.5 * (1 + np.cos(np.pi * epoch / beta_decay_epochs))
    else:
        return beta_max



def compute_grad_norm(parameters, norm_type=2.0):
    with torch.no_grad():
        total_norm = 0.0
        num_params = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
                num_params += 1
        total_norm = total_norm ** (1. / norm_type)
        if num_params > 0:
            mean_norm = total_norm / num_params
        else:
            mean_norm = 0.0  
        return mean_norm

def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)

# Replaces all linear layers with linear_replacement
# TODO: add int8 support for other linear layers including attn and convnets
def replace_linear(model, linear_replacement, include_modules=['c_fc', 'c_proj'], copy_weights=True):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, include_modules, copy_weights)

        if isinstance(module, torch.nn.Linear) and name in include_modules:
            old_module = model._modules[name]
            model._modules[name] = linear_replacement(
                module.in_features,
                module.out_features,
                module.bias is not None,
            )
            if copy_weights:
                model._modules[name].weight.data.copy_(old_module.weight.data)
                if model._modules[name].bias is not None:
                    model._modules[name].bias.data.copy_(old_module.bias)

    return model

def convert_int8_model_to_inference_mode(model):
    for m in model.modules():
        if hasattr(m, 'prepare_for_eval'):
            int8_original_dtype = m.weight.dtype
            m.prepare_for_eval()
            m.int8_original_dtype = int8_original_dtype
            
            
def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def step_lr_thresh(param_groups, base_lr, warmup_length, thresh_list, ratio_list, model):
    if isinstance(base_lr, list):
        assert len(param_groups) == len(base_lr)
        base_lr_list = base_lr
    else:
        base_lr_list = [base_lr for _ in range(len(param_groups))]
    def _lr_adjuster(step):
        for i, param_group in enumerate(param_groups):
            base_lr = base_lr_list[i]
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                lr = base_lr
                for thresh, ratio in zip(thresh_list, ratio_list):
                    if 1.0 / model.logit_scale.exp() <= thresh:
                        lr = base_lr * ratio
                    else:
                        break
            param_group["lr"] = lr
    return _lr_adjuster