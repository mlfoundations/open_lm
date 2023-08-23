#!/bin/bash

BATCHSIZE=1
MODEL="large2048"
EXP_NAME="benchmark-$MODEL"

torchrun --nproc-per-node 1 -m benchmark.main \
    --train-data "pipe:aws s3 cp s3://s-laion/redpajama-tars/8192-v1/{0..7}/shard-{0000000..0000300}.tar -" \
    --train-num-samples 30720 \
    --workers 6 \
    --precision amp_bfloat16 \
    --grad-checkpointing \
    --grad-clip-norm 1 \
    --log-every-n-steps 1 \
    --fsdp \
    --profile \
    --batch-size $BATCHSIZE \
    --model $MODEL \
    --name $EXP_NAME \
