#!/bin/bash
# W4A4 Quantization-Aware Training for ResNet-18
# Fine-tune from full-precision pretrained checkpoint

NNODES=1
NPROC_PER_NODE=2  # 2x RTX A6000

DATA_PATH=/home/didwhddks/Lightening-Transformer/software_model/imagenet
FINETUNE=/home/didwhddks/Lightening-Transformer/software_model/pretrained/resnet/resnet18_fp32.pth
OUTPUT_DIR=/home/didwhddks/Lightening-Transformer/software_model/resumed_ckpt/resnet18_w4a4

input_noise_std=0.03
output_noise_std=0.05

mkdir -p $OUTPUT_DIR

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    ../main.py \
    --model resnet18_quant \
    --wbits 4 \
    --abits 4 \
    --finetune $FINETUNE \
    --data-path $DATA_PATH \
    --data-set IMNET \
    --output_dir $OUTPUT_DIR \
    --batch-size 128 \
    --epochs 90 \
    --weight-decay 1e-8 \
    --lr 1e-4 \
    --warmup-epochs 0 \
    --min-lr 0 \
    --num_workers 16 \
    --pin-mem \
    --input_noise_std ${input_noise_std} \
    --output_noise_std ${output_noise_std} \
    --enable_linear_noise \
    2>&1 | tee $OUTPUT_DIR/train.log
