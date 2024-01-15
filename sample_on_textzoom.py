"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from text_recognition.recognizer_init import *

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.image_datasets import TextZoomDataset_SR_Test


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args.gpu)
    logger.configure()

    logger.log('out_path: {}'.format(args.out_path))
    level = args.out_path.split('/')[-1].split('_')[0]
    print('Evaluating on the {} split:'.format(level))

    if level == 'easy':
        args.num_samples = 1619
    elif level == 'medium':
        args.num_samples = 1411
    elif level == 'hard':
        args.num_samples = 1343
    else:
        raise RuntimeError('Invalid level input', level)

    if args.use_gt_dimss:
        text_rec = None
    else:
        text_rec = CRNN_init(args.text_rec_model_path)
        text_rec.to(dist_util.dev())
        text_rec.eval()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()


    logger.log("loading data...")
    data = load_data_for_worker(args.batch_size, level)

    logger.log("creating samples...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        model_kwargs["low_res"] = model_kwargs["low_res"].to(dist_util.dev())
        model_kwargs["text_rec"] = text_rec


        if not args.use_ddim:
            sample = diffusion.p_sample_loop(
                model,
                (args.batch_size, 3, 32, 128),  # shape of textzoom images, (32, 128)
                clip_denoised=args.clip_denoised,
                cond_fn=None,
                model_kwargs=model_kwargs,
            )
        else:
            sample = diffusion.ddim_sample_loop(
                model,
                (args.batch_size, 3, 32, 128),
                clip_denoised=args.clip_denoised,
                cond_fn=None,
                model_kwargs=model_kwargs,
            )

        sample = ((sample+1) * 127.5).clamp(0, 255).to(th.uint8)

        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(all_samples, sample)  # gather not supported with NCCL
        for sample in all_samples:
            all_images.append(sample.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        logger.log(f"saving to {args.out_path}")
        np.savez(args.out_path, arr)

    dist.barrier()
    logger.log("sampling complete")



def load_data_for_worker(batch_size, level):
    data = TextZoomDataset_SR_Test(level)
    image_arr, gt_text_label_list = data.get_imgarr()
    image_arr = image_arr.transpose(0, 2, 3, 1)

    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    word_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            word_buffer.append(gt_text_label_list[i])

            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                res["gt_text_label"] = word_buffer

                yield res
                buffer, word_buffer = [], []




def create_argparser():
    defaults = dict(
        gpu=0,
        iter=1,
        txt_rec_model="",
        out_path="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        base_samples="",
        model_path="",
        text_rec_model_path="",
        level="",
        use_synthesizer=False,
        use_gt_dimss=True,
        kb=9,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
