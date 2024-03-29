#!/bin/bash -e

MODEL_FLAGS="--large_size 64 --class_cond False --diffusion_steps 1000 --learn_sigma True --noise_schedule linear --num_channels 128 --num_heads 1 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 100 --timestep_respacing 250"
model_path="./ckpt/synthesizer/ema_0.9999_150000.pt"
num_samples=1000
n_iters=5
gpu=2
max_word_length=13
for ((i=1;i<=$n_iters;i++));
do
    out_path="--out_path diff_samples/mr_samples/${num_samples}_samples_${i}.npz"
    args="--max_word_length ${max_word_length} --kb 9 --use_gt_dimss 1 --use_synthesizer 1 --gpu ${gpu} --iter ${i} --model_path ${model_path} --num_samples ${num_samples} ${MODEL_FLAGS} ${SAMPLE_FLAGS} ${out_path}"
    python synthesizer_sample.py $args
done