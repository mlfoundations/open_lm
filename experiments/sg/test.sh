#!/bin/bash
#SBATCH --partition=g40x
#SBATCH --job-name=moe
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/%x_%j.out
#SBATCH --comment=laion
#SBATCH --account=laion
#SBATCH --open-mode=append


module load openmpi
module load cuda/11.8

export MASTER_ADDR=`hostname`
export MASTER_PORT=12802
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG=info

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0


source /fsx/home-$USER/miniconda3/bin/activate open_lm

cd /fsx/home-$USER/open_lm_new
export PYTHONPATH="$PYTHONPATH:/fsx/home-$USER/open_lm_new"

# sbatch one_node_mix.sh 1e-3 1 1000000 64 200 aphid_neox 0.1 4 1e-5 8
LR=$1 #1e-3
SAVES=$2 # 1, 4
TOKENS=$3
BATCHSIZE=$4 # 64, 128
WARM=$5 # 200, 400, 1000, 2000
MODEL=$6 # aphid_neox, ant_neox, potato_neox,
WD=$7
ACC=$8 # 4, 8
CD=$9 # 4, 8
NUM_EXPERTS=${10}

TOTAL_TOKENS=`expr $TOKENS \* $SAVES`

EXP_NAME="mix-$MODEL-$BATCHSIZE-$LR-$WD-$BATCHSIZE-$TOTAL_TOKENS-$WARM-$CD-$NUM_EXPERTS"

echo "node-list: $SLURM_JOB_NODELIST"

torchrun --nproc-per-node 8 -m open_lm.main \
    --train-num-samples 1000000000 \
    --workers 2 \
    --dataset-manifest "s3://laion-west/rpj_tokenized_upsampled_eleutherai/manifest.jsonl" "s3://laion-west/2T_no_rpj_tokenized_upsampled_25k_shards/manifest.jsonl" \
    --train-data-mix-weights 0.725 0.275 \
    --precision amp_bfloat16 \
    --batch-size 8 \
    --log-every-n-steps 20 \
    --grad-clip-norm 1 \
    --lr 6e-4 \
    --warmup 200 \
    --model aphid_neox \
    --wd 0.01 \
    --beta2 0.95 \
    --epochs 4 \
    --report-to wandb \
    --moe-freq 2 \
    --moe-num-experts 64 \
    --wandb-project-name moe \
    --name test$RANDOM \
    --logs /fsx/home-$USER/experiments/mix_wo \
    --resume latest \
    --seed 124 \
    --data-key 'json' \
    --accum-freq 4 \
    --model-norm gain_only_layer_norm \
    --fsdp --fsdp-amp \
    --lr-cooldown-end 1e-5 \
    --no-skip-tokens \
    --accurate-total-tokens
    # --qk-norm \
    # --grad-checkpointing \
