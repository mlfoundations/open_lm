
export PYTHONPATH=<path-to>/codes/lm-evaluation-harness:<path-to>/mbm-prismatic-dev:$PYTHONPATH

model_names=(

)

bucket_name=mbm_paper_finetune_1b
source_s3=s3://tri-ml-datasets/openlm
target_checkpoint_download_path=<path>

# Function to get the number of available GPUs
get_gpu_count() {
    nvidia-smi --list-gpus | wc -l
}

# Function to get available memory on a GPU
get_gpu_memory() {
    local gpu=$1
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu
}

# Initialize an array to store the last use time for each GPU
declare -A last_use_time
gpu_count=$(get_gpu_count)
for ((i=0; i<$gpu_count; i++)); do
    last_use_time[$i]=0
done

# Function to find the best available GPU
find_best_gpu() {
    local best_gpu=-1
    local max_memory=0
    local current_time=$(date +%s)
    
    for ((gpu=0; gpu<$gpu_count; gpu++)); do
        local available_memory=$(get_gpu_memory $gpu)
        local time_since_last_use=$((current_time - ${last_use_time[$gpu]}))
        
        # Check if GPU has more than 10GB available and hasn't been used in the last 60 seconds
        if [ $available_memory -gt 10240 ] && [ $time_since_last_use -gt 180 ]; then
            if [ $available_memory -gt $max_memory ]; then
                max_memory=$available_memory
                best_gpu=$gpu
            fi
        fi
    done
    
    echo $best_gpu
}

# Function to run task on a specific GPU
run_task() {
    local model_name=$1
    local gpu=$2

    mkdir -p $target_checkpoint_download_path/$model_name/checkpoints
    mkdir -p lm_eval_logs/$bucket_name/$model_name

    echo "Downloading model from $source_s3/$bucket_name/$model_name"
    aws s3 sync $source_s3/$bucket_name/$model_name/checkpoints/ $target_checkpoint_download_path/$model_name/checkpoints/ --exclude "*" --include "*latest-checkpoint.pt"
    
    echo "Downloading params"
    aws s3 cp $source_s3/$bucket_name/$model_name/config.json $target_checkpoint_download_path/$model_name/config.json
    
    echo "Running lm_eval for the $model_name model on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu lm_eval --model prismatic \
        --model_args pretrained="$target_checkpoint_download_path/$model_name",model_id="(openvlm-siglip)$target_checkpoint_download_path/$model_name",tokenizer=EleutherAI/gpt-neox-20b \
        --tasks agieval_sat_en,arc_easy,bigbench_conceptual_combinations_multiple_choice,bigbench_cs_algorithms_multiple_choice,boolq,copa,hellaswag,mathqa,piqa,pubmedqa \
        --device cuda \
	    --output_path lm_eval_logs/$bucket_name/$model_name \
        --batch_size 1
}

# Main execution
for model_name in "${model_names[@]}"; do
    while true; do
        gpu=$(find_best_gpu)
        if [ $gpu -ne -1 ]; then
            last_use_time[$gpu]=$(date +%s)
            run_task "$model_name" "$gpu" &
            break
        else
            sleep 10  # Wait for 10 seconds before checking again
        fi
    done
done

# Wait for all background tasks to complete
wait

# Handpicked above random:
#agieval_sat_en,arc_easy,bigbench_conceptual_combinations_multiple_choice,bigbench_cs_algorithms_multiple_choice,boolq,copa,hellaswag,mathqa,piqa,pubmedqa


# Initial tested tasks:
# agieval_lsat_ar,agieval_lsat_lr,agieval_lsat_rc,agieval_sat_en,arc_challenge,arc_easy,bigbench_cs_algorithms_multiple_choice,bigbench_conceptual_combinations_multiple_choice,bigbench_dyck_languages_multiple_choice,bigbench_elementary_math_qa_multiple_choice,bigbench_language_identification_multiple_choice,bigbench_logical_deduction_multiple_choice,bigbench_misconceptions_multiple_choice,bigbench_strange_stories_multiple_choice,bigbench_strategyqa_multiple_choice,bigbench_understanding_fables_multiple_choice,boolq,copa,hellaswag,logiqa,logiqa2,mathqa,openbookqa,piqa,pubmedqa,winogrande
