import os
import glob
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from einops import rearrange, repeat
import pytorch_lightning as pl
from functools import partial
import json

import random
from sgm.util import instantiate_from_config



class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 num_workers=None, shuffle_test_loader=False, shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader

    def prepare_data(self):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers)


class AnimeVideoDataset_Attncond_Condaug(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 nframe_range,
                 uncond_prob=0.0,
                 sketch_type='skt',
                 train_clips='train_clips',
                 attn_aug_scale=0.0,
                 ):
        assert nframe_range[0] >= num_frames

        self.cond_aug = cond_aug
        self.attn_aug_scale = attn_aug_scale
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.uncond_prob = uncond_prob
        self.sketch_type = sketch_type

        with open(f'{data_root}/{train_clips}.json', 'r') as file:
            clips = json.load(file)

        self.all_frames = dict()
        self.candidate_frames = dict()
        n = 0
        for c in clips.keys():
            frames = clips[c]

            if len(frames) >= nframe_range[0] and len(frames) <= nframe_range[1]:
                for i in range(0, len(frames)-num_frames):
                    self.candidate_frames[n] = [frames[k] for k in range(0, i+1)]
                    self.all_frames[n] = [frames[k] for k in range(i+1, i+num_frames+1)]
                    n += 1

    def __len__(self):
        return len(self.all_frames)

    def __getitem__(self, index):
        batch = {}

        candidates = self.candidate_frames[int(index)]
        choices = max(1, int(0.25*len(candidates)))
        first_file = random.choice( candidates[0:choices] )
        first_file = f'{self.data_root}/img_{self.size[0]}/{first_file}.png'

        frm_files = [f'{self.data_root}/img_{self.size[0]}/{f}.png' for f in self.all_frames[int(index)]]
        lat_files = [f.replace('/img_', f'/lat_').replace('.png', '.pt') for f in frm_files]
        skt_files = [f.replace('/img_', f'/{self.sketch_type}_') for f in frm_files]

        # First conditional frame
        first_frame = load_img(first_file, target_size=self.size).unsqueeze(0)

        # Frame latents
        latents = []
        for f in lat_files:
            latents.append( torch.load(f, map_location='cpu') )
        latents = torch.cat(latents)

        # Control sketches
        controls = []
        for f in skt_files:
            skt = load_img(f, target_size=self.size)
            controls.append( -skt * 0.5 + 0.5 )
        controls = torch.stack(controls)

        # Cond or Uncond
        B = first_frame.shape[0]
        if self.uncond_prob > 0:
            if torch.rand([1])[0] < self.uncond_prob:
                batch['crossattn_scale'] = torch.zeros([B, 1, 1])
                batch['concat_scale'] = torch.zeros([B, 1, 1, 1])
            else:
                batch['crossattn_scale'] = torch.ones([B, 1, 1])
                batch['concat_scale'] = torch.ones([B, 1, 1, 1])

        batch['jpg'] = latents
        batch['control_hint'] = controls
        batch['cond_frames_without_noise'] = first_frame
        batch['fps_id'] = ( torch.tensor([self.fps_id]).repeat(self.num_frames) )
        batch['motion_bucket_id'] = ( torch.tensor([self.motion_bucket_id]).repeat(self.num_frames) )
        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames
        
        attn_cond = latents[[0]].clone()
        if self.cond_aug:
            mu, std = -3.0, 0.5 # mean and standard deviation
            log_sigma = np.random.normal(mu, std, 1)[0]
            sigma = np.exp(log_sigma)
            batch['cond_aug'] = repeat( torch.tensor([sigma]), '1 -> b', b=self.num_frames )
            batch['cond_frames'] = first_frame + sigma * torch.randn_like(first_frame)
            batch['attn_cond'] = attn_cond + self.attn_aug_scale * sigma * torch.randn_like(attn_cond)
        else:
            batch['cond_aug'] = repeat( torch.tensor([0.0]), '1 -> b', b=self.num_frames )
            batch['cond_frames'] = first_frame
            batch['attn_cond'] = attn_cond

        return batch


class AnimeVideoDataset_Attncond_First(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 nframe_range,
                 uncond_prob=0.0,
                 sketch_type='skt',
                 train_clips='train_clips',
                 attn_aug_scale=0.0,
                 ):
        assert nframe_range[0] >= num_frames

        self.cond_aug = cond_aug
        self.attn_aug_scale = attn_aug_scale
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.uncond_prob = uncond_prob
        self.sketch_type = sketch_type

        with open(f'{data_root}/{train_clips}.json', 'r') as file:
            clips = json.load(file)

        self.all_frames = dict()
        self.candidate_frames = dict()
        n = 0
        for c in clips.keys():
            frames = clips[c]

            if len(frames) >= nframe_range[0] and len(frames) <= nframe_range[1]:
                for i in range(0, len(frames)-num_frames+1):
                    self.candidate_frames[n] = [frames[k] for k in range(0, i+1)]
                    self.all_frames[n] = [frames[k] for k in range(i+1, i+num_frames)]
                    n += 1

    def __len__(self):
        return len(self.all_frames)

    def __getitem__(self, index):
        batch = {}

        candidates = self.candidate_frames[int(index)]
        choices = max(1, int(0.25*len(candidates)))
        first_file = random.choice( candidates[0:choices] )
        first_file = f'{self.data_root}/img_{self.size[0]}/{first_file}.png'

        frm_files = [first_file] + [f'{self.data_root}/img_{self.size[0]}/{f}.png' for f in self.all_frames[int(index)]]
        lat_files = [f.replace('/img_', f'/lat_').replace('.png', '.pt') for f in frm_files]
        skt_files = [f.replace('/img_', f'/{self.sketch_type}_') for f in frm_files]

        # First conditional frame
        first_frame = load_img(first_file, target_size=self.size).unsqueeze(0)

        # Frame latents
        latents = []
        for f in lat_files:
            latents.append( torch.load(f, map_location='cpu') )
        latents = torch.cat(latents)

        # Control sketches
        controls = []
        for f in skt_files:
            skt = load_img(f, target_size=self.size)
            controls.append( -skt * 0.5 + 0.5 )
        controls = torch.stack(controls)

        # Cond or Uncond
        B = first_frame.shape[0]
        if self.uncond_prob > 0:
            if torch.rand([1])[0] < self.uncond_prob:
                batch['crossattn_scale'] = torch.zeros([B, 1, 1])
                batch['concat_scale'] = torch.zeros([B, 1, 1, 1])
            else:
                batch['crossattn_scale'] = torch.ones([B, 1, 1])
                batch['concat_scale'] = torch.ones([B, 1, 1, 1])

        batch['jpg'] = latents
        batch['control_hint'] = controls
        batch['cond_frames_without_noise'] = first_frame
        batch['fps_id'] = ( torch.tensor([self.fps_id]).repeat(self.num_frames) )
        batch['motion_bucket_id'] = ( torch.tensor([self.motion_bucket_id]).repeat(self.num_frames) )
        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames
        
        attn_cond = latents[[1]].clone()
        if self.cond_aug:
            mu, std = -3.0, 0.5 # mean and standard deviation
            log_sigma = np.random.normal(mu, std, 1)[0]
            sigma = np.exp(log_sigma)
            batch['cond_aug'] = repeat( torch.tensor([sigma]), '1 -> b', b=self.num_frames )
            batch['cond_frames'] = first_frame + sigma * torch.randn_like(first_frame)
            batch['attn_cond'] = attn_cond + self.attn_aug_scale * sigma * torch.randn_like(attn_cond)
        else:
            batch['cond_aug'] = repeat( torch.tensor([0.0]), '1 -> b', b=self.num_frames )
            batch['cond_frames'] = first_frame
            batch['attn_cond'] = attn_cond

        return batch


class AnimeVideoDataset_Firstcond(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 nframe_range,
                 uncond_prob=0.0,
                 sketch_type='skt',
                 train_clips='train_clips',
                 attn_aug_scale=0.0,
                 ):
        assert nframe_range[0] >= num_frames

        self.cond_aug = cond_aug
        self.attn_aug_scale = attn_aug_scale
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.uncond_prob = uncond_prob
        self.sketch_type = sketch_type

        with open(f'{data_root}/{train_clips}.json', 'r') as file:
            clips = json.load(file)

        self.all_frames = dict()
        self.candidate_frames = dict()
        n = 0
        for c in clips.keys():
            frames = clips[c]

            if len(frames) >= nframe_range[0] and len(frames) <= nframe_range[1]:
                for i in range(0, len(frames)-num_frames+1):
                    self.candidate_frames[n] = [frames[k] for k in range(0, i+1)]
                    self.all_frames[n] = [frames[k] for k in range(i+1, i+num_frames)]
                    n += 1

    def __len__(self):
        return len(self.all_frames)

    def __getitem__(self, index):
        batch = {}

        candidates = self.candidate_frames[int(index)]
        choices = max(1, int(0.25*len(candidates)))
        first_file = random.choice( candidates[0:choices] )
        first_file = f'{self.data_root}/img_{self.size[0]}/{first_file}.png'

        frm_files = [first_file] + [f'{self.data_root}/img_{self.size[0]}/{f}.png' for f in self.all_frames[int(index)]]
        lat_files = [f.replace('/img_', f'/lat_').replace('.png', '.pt') for f in frm_files]
        skt_files = [f.replace('/img_', f'/{self.sketch_type}_') for f in frm_files]

        # First conditional frame
        first_frame = load_img(first_file, target_size=self.size).unsqueeze(0)

        # Frame latents
        latents = []
        for f in lat_files:
            latents.append( torch.load(f, map_location='cpu') )
        latents = torch.cat(latents)

        # Control sketches
        controls = []
        for f in skt_files:
            skt = load_img(f, target_size=self.size)
            controls.append( -skt * 0.5 + 0.5 )
        controls = torch.stack(controls)

        # Cond or Uncond
        B = first_frame.shape[0]
        if self.uncond_prob > 0:
            if torch.rand([1])[0] < self.uncond_prob:
                batch['crossattn_scale'] = torch.zeros([B, 1, 1])
                batch['concat_scale'] = torch.zeros([B, 1, 1, 1])
            else:
                batch['crossattn_scale'] = torch.ones([B, 1, 1])
                batch['concat_scale'] = torch.ones([B, 1, 1, 1])

        batch['jpg'] = latents
        batch['control_hint'] = controls
        batch['cond_frames_without_noise'] = first_frame
        batch['fps_id'] = ( torch.tensor([self.fps_id]).repeat(self.num_frames) )
        batch['motion_bucket_id'] = ( torch.tensor([self.motion_bucket_id]).repeat(self.num_frames) )
        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames
        
        attn_cond = latents[[0, 1]].clone()
        if self.cond_aug:
            mu, std = -3.0, 0.5 # mean and standard deviation
            log_sigma = np.random.normal(mu, std, 1)[0]
            sigma = np.exp(log_sigma)
            batch['cond_aug'] = repeat( torch.tensor([sigma]), '1 -> b', b=self.num_frames )
            batch['cond_frames'] = first_frame + sigma * torch.randn_like(first_frame)
            batch['attn_cond'] = attn_cond + self.attn_aug_scale * sigma * torch.randn_like(attn_cond)
        else:
            batch['cond_aug'] = repeat( torch.tensor([0.0]), '1 -> b', b=self.num_frames )
            batch['cond_frames'] = first_frame
            batch['attn_cond'] = attn_cond

        return batch



class AnimeVideoDataset_AttncondPrev(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 nframe_range,
                 uncond_prob=0.0,
                 sketch_type='skt',
                 train_clips='train_clips',
                 ):
        assert nframe_range[0] >= num_frames

        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.uncond_prob = uncond_prob
        self.sketch_type = sketch_type

        with open(f'{data_root}/{train_clips}.json', 'r') as file:
            clips = json.load(file)

        self.all_frames = dict()
        self.candidate_frames = dict()
        n = 0
        for c in clips.keys():
            frames = clips[c]

            if len(frames) >= nframe_range[0] and len(frames) <= nframe_range[1]:
                for i in range(0, len(frames)-num_frames):
                    self.candidate_frames[n] = [frames[k] for k in range(0, i+1)]
                    self.all_frames[n] = [frames[k] for k in range(i+1, i+num_frames+1)]
                    n += 1

    def __len__(self):
        return len(self.all_frames)

    def __getitem__(self, index):
        batch = {}

        candidates = self.candidate_frames[int(index)]
        choices = max(1, int(0.25*len(candidates)))
        # First frame
        first_file = random.choice( candidates[0:choices] )
        first_file = f'{self.data_root}/img_{self.size[0]}/{first_file}.png'
        # Previous frame
        prev_file = candidates[-1]
        prev_file = f'{self.data_root}/lat_{self.size[0]}/{prev_file}.pt'

        frm_files = [f'{self.data_root}/img_{self.size[0]}/{f}.png' for f in self.all_frames[int(index)]]
        lat_files = [f.replace('/img_', f'/lat_').replace('.png', '.pt') for f in frm_files]
        skt_files = [f.replace('/img_', f'/{self.sketch_type}_') for f in frm_files]

        # First conditional frame
        first_frame = load_img(first_file, target_size=self.size).unsqueeze(0)
        # Previous conditional latent
        attn_cond = torch.load(prev_file, map_location='cpu')

        # Frame latents
        latents = []
        for f in lat_files:
            latents.append( torch.load(f, map_location='cpu') )
        latents = torch.cat(latents)

        # Control sketches
        controls = []
        for f in skt_files:
            skt = load_img(f, target_size=self.size)
            controls.append( -skt * 0.5 + 0.5 )
        controls = torch.stack(controls)

        # Cond or Uncond
        B = first_frame.shape[0]
        if self.uncond_prob > 0:
            if torch.rand([1])[0] < self.uncond_prob:
                batch['crossattn_scale'] = torch.zeros([B, 1, 1])
                batch['concat_scale'] = torch.zeros([B, 1, 1, 1])
            else:
                batch['crossattn_scale'] = torch.ones([B, 1, 1])
                batch['concat_scale'] = torch.ones([B, 1, 1, 1])

        batch['jpg'] = latents
        batch['control_hint'] = controls
        batch['cond_frames_without_noise'] = first_frame
        batch['fps_id'] = ( torch.tensor([self.fps_id]).repeat(self.num_frames) )
        batch['motion_bucket_id'] = ( torch.tensor([self.motion_bucket_id]).repeat(self.num_frames) )
        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames
        
        if self.cond_aug:
            mu, std = -3.0, 0.5 # mean and standard deviation
            log_sigma = np.random.normal(mu, std, 1)[0]
            sigma = np.exp(log_sigma)
            batch['cond_aug'] = repeat( torch.tensor([sigma]), '1 -> b', b=self.num_frames )
            batch['cond_frames'] = first_frame + sigma * torch.randn_like(first_frame)
        else:
            batch['cond_aug'] = repeat( torch.tensor([0.0]), '1 -> b', b=self.num_frames )
            batch['cond_frames'] = first_frame

        batch['attn_cond'] = attn_cond

        return batch



def load_img(path, target_size):
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