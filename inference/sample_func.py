# @title Sampling function
import math
import os
from glob import glob
from pathlib import Path
from typing import Optional
import copy

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire

from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import functional as TF

from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config, append_dims
import sgm.modules.diffusionmodules.discretizer as DDPM

from model_hack import Hacked_model, remove_all_hooks


@torch.no_grad()
def sample_video(model, device, inp, arg, verbose=True):

    def get_indices(n_samples, overlap):
        indices = []
        for n in range(n_samples):
            if n == 0:
                start = 1
                first_ref = 0
                second_refs = [0] * overlap
            else:
                start = end - overlap
                first_ref = 0
                second_refs = list(range(start, start+overlap))
                
            end = start + arg.num_frames - overlap - 1
            frame_ind = [first_ref] + second_refs + list(range(start, end))

            ref_ind = 0

            blend_ind = [0] + list(range(-overlap, 0))

            indices.append([frame_ind, ref_ind, blend_ind])

        return indices

    remove_all_hooks(model)
    overlap = arg.overlap
    prev_attn_steps = arg.prev_attn_steps

    n_samples = (len(inp.skts)-(arg.num_frames-overlap)) // (arg.num_frames-2*overlap-1) + 1
    blend_indices = [ list(range(0, arg.num_frames)), 
                  [0] + list(range(1, overlap+1))*2 + [overlap]*(arg.num_frames-2*overlap-1)]
    blend_steps = [0]*(overlap+1) + [25]*(overlap) + [0]*(arg.num_frames-2*overlap-1)
    indices = get_indices(n_samples=n_samples, overlap=overlap)

    # Initialization
    H, W = inp.imgs[0].shape[2:]
    shape = (arg.num_frames, 4, H // 8, W // 8)
    torch.manual_seed(arg.seed)
    x_T = torch.randn(shape, dtype=torch.float32, device="cpu").to(device)

    hacked = Hacked_model(
        model, overlap=overlap, nframes=arg.num_frames, 
        refattn_hook=True, prev_steps=prev_attn_steps,
        refattn_amplify = [
            [1.0]*(overlap+1) + [1.0]*overlap + [1.0]*3 + [1.0]*7,  # Self-attention
            [1.0]*(overlap+1) + [10.0]*overlap + [1.0]*3 + [1.0]*7,  # Ref-attention
        ]
    )

    first_cond = model.encode_first_stage(inp.imgs[0].to(device)) / model.scale_factor
    first_conds = repeat(first_cond, 'b ... -> (b t) ...', t=arg.num_frames-overlap-1)

    for i, index in enumerate(indices):
        frame_ind, ref_ind, blend_ind = index
        input_img = inp.imgs[ref_ind].to(device)
        sketches = torch.cat([inp.skts[i] for i in frame_ind]).to(device)
        if i == 0:
            hacked.operator.mode = 'normal'
            add_conds = None
            intermediates = {'xt': None, 'denoised': None, 'x0': None}
        else:
            hacked.operator.mode = arg.ref_mode
            prev_conds = x0[-overlap:] / model.scale_factor
            add_conds = {'concat': {
                'cond': torch.cat([ first_cond, prev_conds, first_conds ]),
            } }
            for k in intermediates['xt'].keys():
                intermediates['denoised'][k] = intermediates['denoised'][k][blend_ind].clone()

        x0, intermediates = sample(
            model=model, device=device, x_T=x_T, input_img=input_img,
            additional_conditions=add_conds, controls=sketches, hacked=hacked,
            blend_x0=intermediates['denoised'], blend_ind=blend_indices, blend_steps=blend_steps,
            return_intermediate=True, **vars(arg), verbose=True,
        )

        if i == 0:
            outputs = torch.cat([first_cond*model.scale_factor, x0[-14:]]).cpu()
        else:
            outputs = torch.cat([outputs[:-overlap], x0[-14:].cpu()])

        old_xT = x_T.clone()
        x_T = torch.cat([ old_xT[[0]], old_xT[-overlap:], old_xT[-overlap:], old_xT[overlap+1:-overlap], ])

    return outputs


@torch.no_grad()
def decode_video(model, device, latents, arg):
    model.en_and_decode_n_samples_a_time = arg.decoding_t

    N = latents.shape[0]
    B = arg.decoding_t
    olap = arg.decoding_olap
    f = arg.decoding_first

    end = 0

    i = 0

    with torch.autocast('cuda'):
        while end < N:
            start = i * (B - f - olap) + f
            end = min( start + B - f, N)

            indices = [0]*f + list(range(start, end))

            inputs = latents[indices]
            out = model.decode_first_stage(inputs.to(device)).cpu()
            out = torch.clamp(out, min=-1.0, max=1.0)
            if i == 0:
                outputs = out.clone()
            else:
                outputs = torch.cat([ outputs, out[f+olap:] ])
            i += 1
    
    return outputs




def sample(
    model,
    device: str,
    input_img: torch.Tensor,
    hacked = None,
    x_T: torch.Tensor = None,
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    palette: Optional[torch.Tensor] = None,
    anchor: Optional[torch.Tensor] = None,
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    output_folder: Optional[str] = "/content/outputs",
    verbose: bool = True,
    controls: torch.Tensor = None,
    blend_ind = None,
    blend_x0: torch.Tensor = None,
    scale = [1.0, 1.0],
    return_intermediate: bool = False,
    input_latent: torch.Tensor = None,
    first_control: torch.Tensor = None,
    blend_steps = None,
    gamma = 0.0,
    additional_conditions = None,
    starting_conditions = None,
    cfg_combine_forward = True,
    **kwargs,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    seed_everything(seed)

    if True:
        H, W = input_img.shape[2:]
        assert input_img.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)

        if motion_bucket_id > 255:
            print("WARNING: High motion bucket! This may lead to suboptimal performance.")
        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")
        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = input_img
        value_dict["cond_frames"] = input_img + cond_aug * torch.randn_like(input_img)
        value_dict["cond_aug"] = cond_aug
        model.sampler.verbose = verbose
        model.sampler.device = device

    with torch.no_grad():
        with torch.autocast('cuda'):
            # Prepare conditions
            c, uc, additional_model_inputs = get_conditioning(
                model,
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1, num_frames],
                T=num_frames,
                input_latent=input_latent,
                device=device,
                controls=controls, palette=palette, anchor=anchor, first_control=first_control,
                additional_conditions=additional_conditions,
            )
            # Initial noise
            if x_T is None:
                randn = torch.randn(shape, dtype=torch.float32, device="cpu").to(device)
            else:
                randn = x_T.clone()

            # Prepare for swapping conditions
            if starting_conditions is not None:
                original_c = copy.deepcopy(c)

            '''Sampling'''
            intermediate = {'xt': {}, 'denoised': {},}
            with torch.no_grad():
                x = randn.clone()
                sigmas = model.sampler.discretization(num_steps, device=device).to(torch.float32)
                x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
                num_sigmas = len(sigmas)

                for i in model.sampler.get_sigma_gen(num_sigmas):

                    # Blending
                    if blend_steps is not None and blend_ind is not None:
                        blend = (i < max(blend_steps))
                        target_ind = []
                        source_ind = []
                        for k, b in enumerate(blend_steps):
                            if i < b:
                                target_ind.append(blend_ind[0][k])
                                source_ind.append(blend_ind[1][k])
                    else:
                        blend = False

                    if return_intermediate:
                        intermediate['xt'][i] = x.clone()
                    
                    if starting_conditions is not None:
                        if i < starting_conditions['step']:
                            c = copy.deepcopy(original_c)
                            for k in starting_conditions['cond'].keys():
                                c[k] = starting_conditions['cond'][k]
                        else:
                            c = original_c

                    if True:
                        # Prepare sigma
                        s_ones = x.new_ones([x.shape[0]], dtype=torch.float32)
                        sigma = s_ones * sigmas[i]
                        next_sigma = s_ones * sigmas[i+1]
                        sigma_hat = sigma * (gamma + 1.0)
                        # Denoising
                        denoised = denoise(
                            model, hacked, i, x, c, uc, additional_model_inputs,
                            sigma_hat, scale, cfg_combine_forward,
                        )
                        # CFG guidance
                        denoised = guidance(denoised, scale, num_frames)
                        if return_intermediate:
                            intermediate['denoised'][i] = denoised.clone()

                        # x0 blending
                        if blend and blend_x0 is not None:
                            #denoised[target_ind] = blend_x0[num_steps-1][source_ind]
                            denoised[target_ind] = blend_x0[i][source_ind]

                        # Euler step
                        d = (x - denoised) / append_dims(sigma_hat, x.ndim)
                        dt = append_dims(next_sigma - sigma_hat, x.ndim)
                        x = x + dt * d

            samples_z = x.clone().to(dtype=model.first_stage_model.dtype)

    if return_intermediate:
        return samples_z, intermediate
    else:
        return samples_z, None


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def get_conditioning(model, keys, value_dict, N, T, device, input_latent, additional_conditions, dtype=None, **kwargs):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device, dtype=dtype)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device, dtype=dtype)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device, dtype=dtype),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = torch.cat([ value_dict["cond_frames"] ]*N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = torch.cat([ value_dict["cond_frames_without_noise"] ]*N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])

    c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
            )
    if input_latent is not None:
        c['concat'] = input_latent.clone() / 0.18215

    # from here, dtype is fp16
    for k in ["crossattn", "concat"]:
        uc[k] = repeat(uc[k], "b ... -> b t ...", t=T)
        uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=T)
        c[k] = repeat(c[k], "b ... -> b t ...", t=T)
        c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=T)
    for k in uc.keys():
        uc[k] = uc[k].to(dtype=torch.float32)
        c[k] = c[k].to(dtype=torch.float32)

    if 'controls' in kwargs and kwargs['controls'] is not None:
        uc['control_hint'] = kwargs['controls'].to(torch.float32)
        c['control_hint'] = kwargs['controls'].to(torch.float32)
    if 'first_control' in kwargs and kwargs['first_control'] is not None:
        c['first_control'] = kwargs['first_control'].to(torch.float32)
        uc['first_control'] = torch.zeros_like(c['first_control'])
    if 'palette' in kwargs and kwargs['palette'] is not None:
        uc['palette'] = kwargs['palette'].to(torch.float32)
        c['palette'] = kwargs['palette'].to(torch.float32)
    if 'anchor' in kwargs and kwargs['anchor'] is not None:
        uc['anchor'] = kwargs['anchor'].to(torch.float32)
        c['anchor'] = kwargs['anchor'].to(torch.float32)

    if additional_conditions is not None:
        for k in additional_conditions.keys():
            c[k] = additional_conditions[k]['cond'].to(torch.float32)
            if 'uncond' in additional_conditions[k].keys():
                uc[k] = additional_conditions[k]['uncond'].to(torch.float32)
            else:
                uc[k] = additional_conditions[k]['cond'].to(torch.float32)

    additional_model_inputs = {}
    additional_model_inputs["image_only_indicator"] = torch.zeros(1, T).to(device)
    additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

    for k in additional_model_inputs:
        if isinstance(additional_model_inputs[k], torch.Tensor):
            additional_model_inputs[k] = additional_model_inputs[k].to(dtype=torch.float32)

    return c, uc, additional_model_inputs

def denoise(
        model, hacked, step, x,
        c, uc, additional_model_inputs,
        sigma_hat, scale, cfg_combine_forward,
    ):
    # Prepare model input
    if scale[1] != 1.0 and cfg_combine_forward:
        cond_in = dict()
        if additional_model_inputs['image_only_indicator'].shape[0] == 1:
            additional_model_inputs["image_only_indicator"] = additional_model_inputs["image_only_indicator"].repeat(2, 1)
        for k in c:
            if k in ["vector", "crossattn", "concat"] + model.sampler.guider.additional_cond_keys:
                cond_in[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                cond_in[k] = c[k]
        x_in = torch.cat([x] * 2)
        s_in = torch.cat([sigma_hat] * 2)
    else:
        cond_in = c
        x_in = x
        s_in = sigma_hat

    if hacked is not None:
        model_forward = lambda inp, c_noise, cond, **add: hacked(model, step, inp, c_noise, cond, **add)
    else:
        model_forward = model.apply_model

    denoised = model.denoiser(model_forward, x_in, s_in, cond_in, **additional_model_inputs)

    if not cfg_combine_forward and scale[1] != 1.0:
        uc_denoised = model.denoiser(model_forward, x_in, s_in, uc, **additional_model_inputs)
        denoised = torch.cat([uc_denoised, denoised])
    
    if denoised.shape[0] < x_in.shape[0]:
        denoised = rearrange(denoised, '(b t) ... -> b t ...', t=additional_model_inputs["num_video_frames"]-1)
        denoised = torch.cat([denoised[:, [0]], denoised], dim=1)
        denoised = rearrange(denoised, 'b t ... -> (b t) ...')

    return denoised

def guidance(denoised, scale, num_frames):
    if scale[1] != 1.0:
        x_u, x_c = denoised.chunk(2)
        x_u = rearrange(x_u, "(b t) ... -> b t ...", t=num_frames)
        x_c = rearrange(x_c, "(b t) ... -> b t ...", t=num_frames)
        scales = torch.linspace(scale[0], scale[1], num_frames).unsqueeze(0)
        scales = repeat(scales, "1 t -> b t", b=x_u.shape[0])
        scales = append_dims(scales, x_u.ndim).to(x_u.device)
        denoised = rearrange(x_u + scales * (x_c - x_u), "b t ... -> (b t) ...")
    
    return denoised


def write_video(output_folder, fps_id, samples):
    os.makedirs(output_folder, exist_ok=True)
    base_count = len(glob(os.path.join(output_folder, "*.mp4")))
    video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"MP4V"),
        fps_id + 1,
        (samples.shape[-1], samples.shape[-2]),
    )
    
    vid = (
        (rearrange(samples, "t c h w -> t h w c") * 255)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    for frame in vid:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True