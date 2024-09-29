import logging
import math
from inspect import isfunction
from typing import Any, Optional
import copy

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from packaging import version
from torch import nn
from torch.utils.checkpoint import checkpoint

logpy = logging.getLogger(__name__)

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}
    logpy.warn(
        f"No SDP backend available, likely because you are running in pytorch "
        f"versions < 2.0. In fact, you are using PyTorch {torch.__version__}. "
        f"You might want to consider upgrading."
    )

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    logpy.warn("no module 'xformers'. Processing without...")


'''This temporal attention replace the original one in SVD to disable the temporal 
attentions between the first frame (reference path) and the remaining 14 frames (video path).'''
class TemporalAttention_Masked(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        if hasattr(self, '_forward_hooks') and len(self._forward_hooks) > 0:
            # If hooked do nothing
            return x
        else:
            return self._forward(x, context, mask, additional_tokens, n_times_crossframe_attn_in_self)

    def _forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        h = self.heads

        if mask is None:
            T = x.shape[-2]
            dt = T - 14
            mask = torch.ones(T, T).to(x)
            mask[:, :dt] = 0.0
            mask[:dt, :] = 0.0
            inds = [t for t in range(dt)]
            mask[inds, inds] = 1.0
            mask = rearrange(mask, 'h w -> 1 1 h w')
            mask = mask.bool()

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        with sdp_kernel(**BACKEND_MAP[self.backend]):
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


'''The reference attention which replace the original spatial self-attention layers in SVD.'''
class ReferenceAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        if hasattr(self, '_forward_hooks') and len(self._forward_hooks) > 0:
            # If hooked do nothing
            return x
        else:
            return self._forward(x, context, mask, additional_tokens, n_times_crossframe_attn_in_self)

    def _forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        B = x.shape[0] // 14
        T = x.shape[0] // B
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        # Refconcat: Q [K, K0] [V, V0]
        k0 = rearrange(k, '(b t) ... -> b t ...', t=T)[:, [0]]
        k0 = repeat(k0, 'b t0 ... -> b (t t0) ...', t=T)
        k0 = rearrange(k0, 'b t ... -> (b t) ...')
        v0 = rearrange(v, '(b t) ... -> b t ...', t=T)[:, [0]]
        v0 = repeat(v0, 'b t0 ... -> b (t t0) ...', t=T)
        v0 = rearrange(v0, 'b t ... -> (b t) ...')
        k = torch.cat([k, k0], dim=1)
        v = torch.cat([v, v0], dim=1)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        with sdp_kernel(**BACKEND_MAP[self.backend]):
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


'''The 3D convolutional layers which disables the interactions between the 
first frame (reference path) and the remaining 14 frames (video path).'''
class Conv3d_Masked(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.weight = nn.Parameter( torch.zeros([out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]]) )
        self.bias = nn.Parameter( torch.zeros([out_channels]) )

    def forward(self, x):
        dt = x.shape[2] - 14

        zeros_pad = torch.zeros_like(x[:, :, [0]])

        xs = []
        for i in range(dt):
            xs.append( x[:, :, [i]] )
            xs.append( zeros_pad )
        xs.append( x[:, :, dt:] )
        x = torch.cat(xs, dim=2)

        x = torch.nn.functional.conv3d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            padding=self.padding,
        )

        out_ind = [2*i for i in range(dt)]

        x = torch.cat([ x[:, :, out_ind], x[:, :, 2*dt:] ], dim=2)

        return x



def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def exists(val):
    return val is not None