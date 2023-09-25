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

# Specify the results directory
result_dir="./results"

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

    # Extract the directory path of the checkpoint
    checkpoint_dir=$(dirname "$checkpoint")
    last_part=$(basename "$checkpoint" .pt)
    third_last_part=$(basename $(dirname "$checkpoint_dir"))
    # Define the destination path for results.jsonl
    dest="${result_dir}/${model}_${third_last_part}_${last_part}_results.jsonl"


    # Copy the results.jsonl file to the result_dir with a distinct name
    cp "${checkpoint_dir}/results.jsonl" "$dest"
    
    # Print information about the copying
    echo "Copied results.jsonl from $checkpoint to $dest"
  done
done
