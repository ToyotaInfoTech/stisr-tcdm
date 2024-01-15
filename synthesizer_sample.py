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
from guided_diffusion.image_datasets import RandomEnglishWords
# from guided_diffusion.image_datasets import load_data_textzoom_test



def main():
    args = create_argparser().parse_args()
    out_path = args.out_path
    logger.log('out filename: {}'.format(out_path))

    out_root = out_path.split('/')[:-1]
    out_root = '/'.join(out_root)
    print('out root', out_root)
    print('n samples', args.num_samples)

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
    data = load_data_for_worker(args.batch_size, out_root, args.num_samples, args.iter, args.max_word_length)

    logger.log("creating samples...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = next(data)

        if not args.use_ddim:
            sample = diffusion.p_sample_loop(
                model,
                (args.batch_size, 3, 32, 128),  # shape of textzoom images, (32, 128)
                clip_denoised=args.clip_denoised,
                cond_fn=None,
                model_kwargs=model_kwargs,
            )
        else:
            print('use_ddim = true')
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
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def load_data_for_worker(batch_size, out_root, n_samples, iter, max_word_length):

    data = RandomEnglishWords(max_word_length)
    words_list = data.get_imgarr(n_samples)

    words_filename = 'words_list_{}.txt'.format(iter)
    with open(os.path.join(out_root, words_filename), 'w') as f:
        for word in words_list:
            f.write(word)
            f.write('\n')

    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    word_buffer = []
    while True:
        for i in range(rank, len(words_list), num_ranks):
            word_buffer.append(words_list[i])

            if len(word_buffer) == batch_size:
                res = dict()
                res["gt_text_label"] = word_buffer

                yield res
                word_buffer = []


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
        use_synthesizer=False,
        max_word_length=13
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
