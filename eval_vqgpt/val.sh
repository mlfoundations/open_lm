#!/bin/bash

# Declare an associative array with model names as keys and lists of checkpoints as values
declare -A checkpoints_for_model
checkpoints_for_model[vqgpt_small_4k_1024]="
/path/to/small/model/checkpoint1
/path/to/small/model/checkpoint2
/path/to/small/model/checkpoint3
"

checkpoints_for_model[vqgpt_medium_4k_1024]="
/path/to/big/model/checkpoint1
/path/to/big/model/checkpoint2
/path/to/big/model/checkpoint3
"

# Loop over models
for model in "${!checkpoints_for_model[@]}"; do
  # Get the list of checkpoints for the current model
  checkpoints=${checkpoints_for_model[$model]}
  
  # Loop over checkpoints for the current model
  for checkpoint in $checkpoints; do
    echo "Running model $model with checkpoint $checkpoint"
    
    torchrun --nproc-per-node 1 -m open_lm.main \
        --val-data "junk/shard-000.tar" \
        --val-num-samples 100000000 \
        --workers 6 \
        --batch-size 1 \
        --log-every-n-steps 20 \
        --precision amp_bfloat16 \
        --model "$model" \
        --resume "$checkpoint" \
        --model-norm gain_only_layer_norm \
        --qk-norm
  done
done
