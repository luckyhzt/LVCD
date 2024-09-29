# *LVCD:* Reference-based Lineart Video Colorization with Diffusion Models

## ACM Transactions on graphics & SIGGRAPH Asia 2024

[Project page](https://luckyhzt.github.io/lvcd) | [arXiv](https://arxiv.org/abs/2409.12960)

Zhitong Huang $^1$, Mohan Zhang $^2$, [Jing Liao](https://scholars.cityu.edu.hk/en/persons/jing-liao(45757c38-f737-420d-8a7f-73b58d30c1fd).html) $^{1*}$

<font size="1"> $^1$: City University of Hong Kong, Hong Kong SAR, China &nbsp;&nbsp; $^2$: WeChat, Tencent Inc., Shenzhen, China </font> \
<font size="1"> $^*$: Corresponding author </font>

## Abstract:
We propose the first video diffusion framework for reference-based lineart video colorization. Unlike previous works that rely solely on image generative models to colorize lineart frame by frame, our approach leverages a large-scale pretrained video diffusion model to generate colorized animation videos. This approach leads to more temporally consistent results and is better equipped to handle large motions. Firstly, we introduce <em>Sketch-guided ControlNet</em> which provides additional control to finetune an image-to-video diffusion model for controllable video synthesis, enabling the generation of animation videos conditioned on lineart. We then propose <em>Reference Attention</em> to facilitate the transfer of colors from the reference frame to other frames containing fast and expansive motions. Finally, we present a novel scheme for sequential sampling, incorporating the <em>Overlapped Blending Module</em> and <em>Prev-Reference Attention</em>, to extend the video diffusion model beyond its original fixed-length limitation for long video colorization. Both qualitative and quantitative results demonstrate that our method significantly outperforms state-of-the-art techniques in terms of frame and video quality, as well as temporal consistency. Moreover, our method is capable of generating high-quality, long temporal-consistent animation videos with large motions, which is not achievable in previous works.





# Installation

```shell
conda create -n lvcd python=3.10.0
conda activate lvcd
pip3 install -r requirements/pt2.txt
```

# Download pretrained models
1. Download the pretrained [SVD weights](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd.safetensors) and put it as `./checkpoints/svd.safetensors`
2. Download the finetuned weights for [Sketch-guided ControlNet](https://huggingface.co/luckyhzt/lvcd_pretrained_models/resolve/main/lvcd.ckpt) and put is as `./checkpoints/lvcd.ckpt`

# Inference
All the code for inference is placed under `./inference/`, where the jupyter notebook `sample.ipynb` demonstrates how to sample the videos. Two testing clips are also provided.