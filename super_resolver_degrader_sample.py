"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    logger.log('out_path: {}'.format(args.out_path))
    logger.log('base_samples: {}'.format(args.base_samples))

    words_list_path = args.base_samples.split('/')[:-1]
    words_list_path = '/'.join(words_list_path)
    logger.log('words_list_path: {}'.format(words_list_path))

    dist_util.setup_dist(args.gpu)
    logger.configure()

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
    data = load_data_for_worker(args.base_samples, args.batch_size, words_list_path, args.iter)

    logger.log("creating samples...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        model_kwargs["low_res"] = model_kwargs["low_res"].to(dist_util.dev())

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



def load_data_for_worker(base_samples, batch_size, words_list_path, iter):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
    
    words_list = []
    words_filename = 'words_list_{}.txt'.format(iter)
    with open(os.path.join(words_list_path, words_filename), 'r') as f:
        line = f.readline()
        while line:
            line = line[:-1]
            print(line)
            words_list.append(line)
            line = f.readline()
    print('length of words list:', len(words_list))


    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    word_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            word = words_list[i]
            word_buffer.append(word)

            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
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
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
