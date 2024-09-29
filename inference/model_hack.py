import copy
from typing import Dict, Callable
from collections import OrderedDict
import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F
import einops
import math

from torch.backends.cuda import SDPBackend, sdp_kernel


def remove_all_hooks(model: torch.nn.Module) -> None:
    for _, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            if hasattr(child, "_backward_hooks"):
                child._backward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_hooks(child)


class Hacked_model(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.operator = Reference(model, **kwargs)
    
    def forward(self, model, step, x_in, c_noise, cond_in, **additional_model_inputs):
        # Register hooks
        self.operator.register_hooks(model)
        self.operator.setup(step)
        # Model forward
        out = model.apply_model(x_in, c_noise, cond_in, **additional_model_inputs)
        # Remove hooks
        self.operator.remove_hooks()

        return out
    
    def clear_storage(self):
        self.operator.clear_storage()


class Operator():
    def __init__(self, model):
        self.hook_handles = []
        self.layers = self.get_hook_layers(model)

    def get_hook_layers(self, model):
        raise NotImplementedError

    def hook(self, module, inputs, outputs):
        raise NotImplementedError

    def setup(self, step, branch, opt):
        raise NotImplementedError
    
    def clear_storage(self):
        self.storage.clear()

    def register_hooks(self, model):
        for m in model.modules():
            index = id(m)
            if index in self.layers.keys():
                handle = m.register_forward_hook(self.hook)
                self.hook_handles.append(handle)
    
    def remove_hooks(self):
        while len(self.hook_handles) > 0:
            self.hook_handles[0].remove()
            self.hook_handles.pop(0)


class Reference(Operator):
    def __init__(self, model, **kwargs):
        self.storage = nn.ParameterDict()
        self.overlap = kwargs['overlap']
        self.nframes = kwargs['nframes']
        self.refattn_amp = kwargs['refattn_amplify']
        self.refattn_hook = kwargs['refattn_hook']
        self.prev_steps = kwargs['prev_steps']
        super().__init__(model)
    
    def setup(self, step):
        self.step = step

    def get_hook_layers(self, model):
        layers = dict()
        if self.refattn_hook:
            # Hook ref attention layers in ControlNet
            layer_name = model.control_model.spatial_self_attn_type.split('.')[-1]
            i = 0
            for name, module in model.control_model.named_modules():
                if module.__class__.__name__ == layer_name and '.time_stack' not in name and '.attn1' in name:
                    layers[id(module)] = f'cnet-refcond-{i}'
                    i += 1
            # Hook ref attention layers in UNet
            layer_name = model.model.diffusion_model.spatial_self_attn_type.split('.')[-1]
            i = 0
            for name, module in model.model.diffusion_model.named_modules():
                if module.__class__.__name__ == layer_name and '.time_stack' not in name and '.attn1' in name:
                    layers[id(module)] = f'unet-refcond-{i}'
                    i += 1
        return layers
    
    @torch.no_grad()
    def hook(self, module, inputs, outputs):
        layer = self.layers[id(module)]
        if 'refcond' in layer:
            out = self.reference_attn_forward(module, inputs, outputs)
        return out


    def reference_attn_forward(self, module, inputs, outputs):
        overlap = self.overlap
        T = self.nframes
        h = module.heads

        index = id(module)
        layer = self.layers[index]
        layer_ind = int(layer.split('-')[-1])

        q = module.to_q(inputs[0])
        k = module.to_k(inputs[0])
        v = module.to_v(inputs[0])

        olap = 3

        if self.mode == 'normal':
            indices = [
                list(range(0, T)),
                list(range(0, overlap+1)) + [0]*(T-overlap-1),
            ]
        elif self.mode == 'prevref':
            if self.step < self.prev_steps:
                indices = [
                    list(range(0, 2*overlap+1)) + list(range(2*overlap+1-olap, T-olap)),
                    list(range(0, overlap+1)) + list(range(1, overlap+1)) + [0]*(T-2*overlap-1),
                ]
            else:
                '''indices = [
                    list(range(0, 2*overlap+1)) + list(range(2*overlap+1-olap, T-olap)),
                    list(range(0, overlap+1)) + [0]*overlap + [0]*(T-2*overlap-1),
                ]'''
                indices = [
                    list(range(0, T)),
                    list(range(0, overlap+1)) + [0]*(T-overlap-1),
                ]
        elif self.mode == 'normal1':
            indices = [
                list(range(0, T)),
                list(range(0, overlap+1)) + [overlap] + [0]*(T-overlap-2),
            ]
        '''elif self.mode == 'tempref':
            if self.step < self.prev_steps:
                indices = [
                    list(range(0, 2*overlap+1)) + [2*overlap]*olap + list(range(2*overlap+1, T-olap)),
                    list(range(0, overlap+1)) + list(range(1, overlap+1)) + [0]*(T-2*overlap-1),
                ]
            else:
                indices = [
                    list(range(0, 2*overlap+1)) + [2*overlap]*olap + list(range(2*overlap+1, T-olap)),
                    list(range(0, overlap+1)) + [0]*overlap + [0]*(T-2*overlap-1),
                ]'''
        
        
        k = rearrange(k, '(b t) ... -> b t ...', t=T)
        v = rearrange(v, '(b t) ... -> b t ...', t=T)
        k = torch.cat([k[:, indices[i]] for i in range(len(indices))], dim=2).clone()
        v = torch.cat([v[:, indices[i]] for i in range(len(indices))], dim=2).clone()
        k = rearrange(k, 'b t ... -> (b t) ...')
        v = rearrange(v, 'b t ... -> (b t) ...')

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        # Attention
        N = q.shape[-2]
        with sdp_kernel(**{"enable_math": True, "enable_flash": True, "enable_mem_efficient": True}):
            if layer_ind > 12 or self.mode == 'normal':
                attn_bias = None
            else:
                attn_bias = torch.zeros([T, 1, N, 2*N], device=q.device, dtype=torch.float32)
                amplify = torch.tensor(self.refattn_amp).to(attn_bias)
                amplify = rearrange(amplify, 'b t -> t 1 1 b')
                amplify = amplify.log()
                attn_bias[:, :, :, :N] = amplify[:, :, :, [0]]
                attn_bias[:, :, :, N:] = amplify[:, :, :, [1]]
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias,
            )
        del q, k, v

        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return module.to_out(out)











