from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import os
from PIL import Image
import torch
import random
import argparse
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


def process(model, apply_canny, ddim_sampler, input_image, prompt, a_prompt, n_prompt,
            num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold,
            high_threshold):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control],
                   "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                           255).astype(
            np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results


def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion')
    parser.add_argument('--category', nargs='+',
                        type=str, default=['car'])
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print(args)

    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    apply_canny = CannyDetector()

    for cate in args.category:
        for idx in range(0, 7000):
            img = np.array(
                Image.open(
                    f'../data/{cate}/train/images/P3D-Diffusion_train_00{idx:04d}.png'))
            os.makedirs(f'../data/{cate}/train/images_diffusion', exist_ok=True)
            H, W, C = img.shape

            prompt = cate
            # a_prompt = 'becst quality, extremely detailed, sharp edges'
            a_prompt = 'best quality, extremely detailed'
            n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
            num_samples = 1
            image_resolution = 512
            ddim_steps = 30
            guess_mode = False
            strength = 1.0
            scale = 9.0
            seed = 10
            eta = 0.0
            low_thr = 150
            high_thr = 200

            results = process(
                model, apply_canny, ddim_sampler, img, prompt, a_prompt, n_prompt,
                num_samples, image_resolution, ddim_steps, guess_mode, strength, scale,
                seed, eta, low_thr, high_thr)

            img = results[1]

            img = cv2.resize(img, (W, H))

            Image.fromarray(img).save(
                f'../data/{cate}/train/images_diffusion/P3D-Diffusion_train_00{idx:04d}.png')


if __name__ == '__main__':
    main()
