import nvidia_smi
import numpy as np
from tqdm import tqdm
import argparse
import sys
from omegaconf import OmegaConf
from glob import glob
import torch
import os
from PIL import Image
from torchvision import transforms

from multiprocessing import Pool
from functools import partial

from sgm.util import default, instantiate_from_config
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
):
    config = OmegaConf.load(config)
    config.model.params.conditioner_config.params.emb_models[
        0
    ].params.open_clip_embedding_config.params.init_device = device
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    with torch.device(device):
        model = instantiate_from_config(config.model).to(device).eval().requires_grad_(False)

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter


def load_img(path, target_size=[576, 1024]):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")
    
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


def encode(img_dir, device, resolution):
    model, filter = load_model(
        config='configs/svd.yaml',
        device=device,
        num_frames=14,
        num_steps=25,
    )
    # change the dtype of unet
    model.model.to(dtype=torch.float32)
    torch.cuda.empty_cache()
    model = model.requires_grad_(False)

    save_dir = img_dir.replace(f'/img_{resolution[0]}/', f'/lat_{resolution[0]}/')
    os.makedirs(f'{save_dir}/..', exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    image_files = glob(f'{img_dir}/*.png')

    for file in tqdm(image_files):
        pt_file = file.replace(f'/img_{resolution[0]}/', f'/lat_{resolution[0]}/').replace('.png', '.pt')
        if not os.path.isfile(pt_file):
            img = load_img(file, resolution).to(device).unsqueeze(0)
            latent = model.encode_first_stage(img)
            torch.save(latent, pt_file)



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    resolution = [320, 576]
    img_dirs = [
        f'#path_to_unzipped_dataset/Animation_video/img_320/WhisperOfTheHeart',
        f'#path_to_unzipped_dataset/Animation_video/img_320/TheWindRises',
        f'#path_to_unzipped_dataset/Animation_video/img_320/TheSecretWorldOfArriey',
        f'#path_to_unzipped_dataset/Animation_video/img_320/Ponyo',
        f'#path_to_unzipped_dataset/Animation_video/img_320/MyNeighborTotoro',
        f'#path_to_unzipped_dataset/Animation_video/img_320/KikisDeliveryService',
        ]
    devices = ['cuda:0']

    total = len(img_dirs)
    num_workers = len(devices)

    for n in range(0, total, num_workers):
        args = [(img_dirs[i], devices[i-n], resolution) for i in range(n, min(n+num_workers, total))]
        pool = Pool(len(args))
        _ = pool.starmap(encode, args)



