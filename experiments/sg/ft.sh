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
module load cuda/12.1

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

# sbatch experiments/sg/one_node_mix.sh 6e-4 4 250000000 8 200 aphid_neox 0.01 4 3e-5 128
MODEL=$1 # aphid_neox, ant_neox, potato_neox,
NUM_EXPERTS=$2

LR=5e-4 #1e-3
SAVES=2 # 1, 4
TOKENS=10000000000
BATCHSIZE=8 # 64, 128
WARM=200 # 200, 400, 1000, 2000
WD=0.1
ACC=4 # 4, 8
CD=0.00001 # 4, 8


TOTAL_TOKENS=`expr $TOKENS \* $SAVES`

EXP_NAME="mix-expert-norm-$MODEL-$NUM_EXPERTS-$BATCHSIZE-$LR-$WD-$BATCHSIZE-$TOTAL_TOKENS-$WARM-$CD"

echo "node-list: $SLURM_JOB_NODELIST"

if [ $NUM_EXPERTS == "0" ]; then MOE_FREQ=0; else MOE_FREQ=2; fi

srun --account laion --cpu_bind=v --accel-bind=gn python -m open_lm.main \
    --train-num-samples $TOKENS \
    --workers 2 \
    --dataset-manifest "s3://laion-west/rpj_tokenized_upsampled_eleutherai/manifest.jsonl" "s3://laion-west/2T_no_rpj_tokenized_upsampled_25k_shards/manifest.jsonl" \
    --train-data-mix-weights 0.725 0.275 \
    --precision amp_bfloat16 \
    --batch-size $BATCHSIZE \
    --accum-freq $ACC \
    --log-every-n-steps 20 \
    --grad-clip-norm 1 \
    --lr $LR \
    --warmup $WARM \
    --model $MODEL \
    --wd $WD \
    --beta2 0.95 \
    --epochs $SAVES \
    --report-to wandb \
    --wandb-project-name moe_sweep_new_1 \
    --name $EXP_NAME \
    --fsdp --fsdp-amp \
    --logs /fsx/home-$USER/experiments/mix_wo_megablocks_newsweep_2 \
    --resume latest \
    --data-key 'json' \
    --model-norm gain_only_layer_norm \
    --lr-scheduler cosine \
    --lr-cooldown-end 0.00001 \
    --no-skip-tokens \
    --accurate-total-tokens \
    --moe-freq $MOE_FREQ \
    --moe-num-experts $NUM_EXPERTS \
    --moe-top-k 2 \
    --moe-capacity-factor 1.25 --moe-loss-weight 0.1
    # --qk-norm \
    # --grad-checkpointing \
