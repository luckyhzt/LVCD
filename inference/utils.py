from PIL import Image
import cv2
import numpy as np
from glob import glob
import os
import random
from tqdm import tqdm
from typing import Dict, Callable
from collections import OrderedDict
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from pathlib import Path
import yaml
from einops import rearrange, repeat
import torch.nn.functional as F
from omegaconf import OmegaConf

from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.util import default, instantiate_from_config


def write_video(frames, output_folder, fps):
    frames = torch.clamp((frames + 1.0) / 2.0, min=0.0, max=1.0)
    os.makedirs(output_folder, exist_ok=True)
    base_count = len(glob(os.path.join(output_folder, "*.mp4")))
    video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"MP4V"),
        fps + 1,
        (frames.shape[-1], frames.shape[-2]),
    )

    vid = (
        (rearrange(frames, "t c h w -> t h w c") * 255)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    for frame in vid:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()


def load_img(path, target_size=[576, 1024], image_mode='RGB'):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert(image_mode)
    
    if target_size == None:
        tform = transforms.Compose([
                transforms.ToTensor(),
            ])
    else:
        tform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
            ])
        
    image = tform(image)
    return 2.*image - 1.


def tensor_to_pil(tensor, cols=5, gap=20, gap_value=0.0):
    cols = min(cols, len(tensor))
    rows = int( np.ceil(len(tensor) / cols) )
    gap = int(gap)
    tensor = tensor.clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) * 0.5
    if len(tensor.shape) == 4:
        if (tensor.shape[0]) == 1:
            tensor_rows = tensor.squeeze(0)
        else:
            tensor_rows = []
            if cols*rows > tensor.shape[0]:
                shape = [int(cols*rows-tensor.shape[0])] + list(tensor.shape[1:])
                tensor = torch.cat([tensor, torch.zeros(shape).to(tensor)], dim=0)

            for r in range(rows):
                tensor_cols = [F.pad(tensor[i], (gap//2, gap//2), 'constant', gap_value) for i in range(r*cols, (r+1)*cols)]
                tensor_cols = torch.cat(tensor_cols, dim=-1)
                tensor_rows.append(
                    F.pad(tensor_cols, (0, 0, gap//2, gap//2), 'constant', gap_value)
                )
            tensor_rows = torch.cat(tensor_rows, dim=-2)
    else:
        tensor_rows = tensor
    img = TF.to_pil_image(tensor_rows)
    return img


def make_video(save_path, frames, fps, cols, name=None, show_index=True, verbose=True, text=None, gap=0):
    N, F, _, H, W = frames.shape
    rows = int( np.ceil(N / cols) )

    if name is None:
        base_count = len(glob(os.path.join(save_path, f"*.mp4")))
        name = f'{base_count:06d}'

    video_path = os.path.join(save_path, f'{name}.mp4')

    out_stream = cv2.VideoWriter(
        video_path,
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=fps,
        frameSize=[W*cols, H*rows])
    
    pbar = tqdm(range(F)) if verbose else range(F)

    for i in pbar:
        frame = frames[:, i]
        frame = tensor_to_pil(frame, cols=cols, gap=gap)
        frame = np.array(frame) #[:, :, [2,1,0]]
        if show_index:
            frame = cv2.putText(frame, str(i), (70, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,50,50), 2, cv2.LINE_AA)
        if isinstance(text, str):
            frame = cv2.putText(frame, text, (70, 70), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (255,255,255), 2, cv2.LINE_AA)
        elif isinstance(text, list):
            rgap = frame.shape[0] // rows
            cgap = frame.shape[1] // cols
            for i in range(len(text)):
                r = i // cols
                c = i % cols
                frame = cv2.putText(frame, text[i], (c*cgap+10, r*rgap+30), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0,0,0), 2, cv2.LINE_AA)
        # +282

        frame = frame[:, :, [2,1,0]]
        
        out_stream.write(frame)

    out_stream.release()


def pil_to_tensor(pil, device=None, size=None):
    if size is not None:
        pil = pil.resize(size)
    tensor = TF.to_tensor(pil)
    if device is not None:
        tensor = tensor.to(device)
    return tensor * 2.0 - 1.0


def save_frames(tensor, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(tensor)):
        img = tensor_to_pil(tensor[i])
        img.save(f'{save_dir}/{i}.png')


def load_model(device, config_path, svd_path, ckpt_path, use_xformer=True):
    config = OmegaConf.load(config_path)
    config.model.params.conditioner_config.params.emb_models[0].params.open_clip_embedding_config.params.init_device = device
    config.model.params.drop_first_stage_model = False
    config.model.params.init_from_unet = False
    if use_xformer:
        config.model.params.network_config.params.spatial_transformer_attn_type = 'softmax-xformers'
        config.model.params.controlnet_config.params.spatial_transformer_attn_type = 'softmax-xformers'
    else:
        config.model.params.network_config.params.spatial_transformer_attn_type = 'softmax'
        config.model.params.controlnet_config.params.spatial_transformer_attn_type = 'softmax'
    config.model.params.ckpt_path = svd_path
    config.model.params.control_model_path = ckpt_path

    with torch.device(device):
        model = instantiate_from_config(config.model).to(device).eval().requires_grad_(False)

    filter = DeepFloydDataFiltering(verbose=False, device=device)

    # change the dtype of unet
    model.model.to(dtype=torch.float32)
    model.control_model.to(dtype=torch.float32)
    model.eval()
    model = model.requires_grad_(False)

    torch.cuda.empty_cache()

    return model