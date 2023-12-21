#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2

VALSHARD="/scratch/08002/gsmyrnis/open_lm_val_data/shard_00000000.tar"

LR=0.003
SAVES=1
TOKENS=654000000
BATCHSIZE=32  #192 # 64, 128
WARM=2000 # 200, 400, 1000, 2000
MODEL="open_lm_160m" #"open_lm_1b" #"open_lm_160m"
WD=0.033
ACC=16  #24 # 4, 8
CD=3e-05 # 4, 8

TOTAL_TOKENS=`expr $TOKENS \* $SAVES`

EXP_NAME="TACC-mix-noresample-$MODEL-$BATCHSIZE-$LR-$WD-$BATCHSIZE-$TOTAL_TOKENS-$WARM-$CD"

#echo "node-list: $SLURM_JOB_NODELIST"

torchrun --nproc-per-node 3 -m open_lm.main \
    --train-num-samples $TOKENS \
    --workers 2 \
    --dataset-manifest "/scratch/08002/gsmyrnis/open_lm_tokenized/rpj/manifest.jsonl" "/scratch/08002/gsmyrnis/open_lm_tokenized/not_rpj/manifest.jsonl" \
    --train-data-mix-weights 0.725 0.275 \
    --precision amp_bfloat16 \
    --batch-size $BATCHSIZE \
    --grad-checkpointing \
    --log-every-n-steps 20 \
    --grad-clip-norm 1 \
    --lr $LR \
    --warmup $WARM \
    --model $MODEL \
    --wd $WD \
    --beta2 0.95 \
    --epochs $SAVES \
    --name $EXP_NAME \
    --logs $SCRATCH/open_lm_logs \
    --resume latest \
    --seed 124 \
    --data-key 'json' \
    --accum-freq $ACC \
    --lr-cooldown-end $CD \
    --report-to wandb \
    --wandb-project-name lm1
