
#!/bin/bash -e

MODEL_FLAGS="--large_size 64 --class_cond False --diffusion_steps 1000 --learn_sigma True --noise_schedule linear --num_channels 128 --num_heads 1 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 100 --timestep_respacing 250"
model_path="./ckpt/super_resolver/ema_0.9999_100000.pt"
gpu=0
for level in easy medium hard
do
    out_path="--out_path ./diff_samples/textzoom/${level}_sr_samples_gt_dimss.npz"
    args="--kb 9 --use_gt_dimss 1 --use_synthesizer 0 --gpu ${gpu} --level ${level} --model_path ${model_path} ${MODEL_FLAGS} ${SAMPLE_FLAGS} ${out_path} ${txt_rec_model}"
    python sample_on_textzoom.py $args
    
done
