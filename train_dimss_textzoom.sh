#!/bin/bash -e

gpu=1
MODEL_FLAGS="--large_size 64 --class_cond False --diffusion_steps 1000 --learn_sigma True --noise_schedule linear --num_channels 128 --num_heads 1 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
TRAIN_FLAGS="--kb 6 --use_gt_dimss 0 --use_synthesizer 0 --use_degrader 0 --gpu ${gpu} --lr 3e-4 --batch_size 32"
mpiexec -n 1 python train_on_textzoom.py $TRAIN_FLAGS $MODEL_FLAGS
