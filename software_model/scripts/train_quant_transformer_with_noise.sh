#!/bin/bash
# @Author: Hanqing Zhu(hqzhu@utexas.edu)
# @Date:   1969-12-31 18:00:00
# @Last Modified by:   Hanqing Zhu(hqzhu@utexas.edu)
# @Last Modified time: 2023-11-09 03:03:06
wbits=8
abits=8
id=8bit
lr=5e-4
weight_decay=1e-8
# batch_size=512
batch_size=128
# epochs=300
epochs=100
# port=47771
headwise=1
input_noise_std=0.03
output_noise_std=0.05

resumed_ckpt="/home/didwhddks/research/Lightening-Transformer/software_model/test/deit_tiny_8bit/8w8a_bs128_baselr5e-4_weightdecay1e-8_ft100_headwise1_noise_i_0.03_o_0.05_linear_noise/best_checkpoint.pth"

# torchrun \
# --master_port ${port} \
# --nproc_per_node=4 ../main.py \
CUDA_VISIBLE_DEVICES=0 python ../main.py \
--model deit_tiny_patch16_224_quant \
--drop-path 0 \
--batch-size ${batch_size} \
--lr ${lr} \
--min-lr 0 \
--epochs ${epochs} \
--warmup-epochs 0 \
--weight-decay ${weight_decay} \
--wbits ${wbits} \
--abits ${abits} \
--dist-eval \
--output_dir ../test/deit_tiny_${id}/${wbits}w${abits}a_bs${batch_size}_baselr${lr}_weightdecay${weight_decay}_ft${epochs}_headwise${headwise}_noise_i_${input_noise_std}_o_${output_noise_std}_linear_noise \
--resume ${resumed_ckpt} \
--data-path /home/didwhddks/research/Lightening-Transformer/software_model/imagenet \
--headwise \
--input_noise_std ${input_noise_std} \
--output_noise_std ${output_noise_std} \
--enable_linear_noise
# --finetune ../pretrained/deit_tiny_patch16_224-a1311bcf.pth \
