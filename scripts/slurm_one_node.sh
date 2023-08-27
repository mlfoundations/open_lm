#!/bin/bash
#SBATCH --partition=g40x
#SBATCH --job-name=open_lm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/%x_%j.out
#SBATCH --comment=datanet
#SBATCH --account=datanet
#SBATCH --open-mode=append
##SBATCH --exclusive
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

export PATH="/admin/home-$USER/miniconda3/condabin:$PATH"
source /admin/home-${USER}/miniconda3/etc/profile.d/conda.sh
conda activate open_lm  # install according to tng/tools/environment.yml

cd /admin/home-$USER/open_lm
export PYTHONPATH="$PYTHONPATH:/admin/home-$USER/open_lm"

LR=$1
SAVES=$2
TOKENS=$3
BATCHSIZE=$4
WARM=$5
MODEL=$6
MODEL_NORM=$7
WD=0.1
ACC=8

TOTAL_TOKENS=`expr $TOKENS \* $SAVES`

EXP_NAME="mix-$MODEL-$BATCHSIZE-$LR-$WD-$BATCHSIZE-$TOTAL_TOKENS-$WARM-$MODEL_NORM-qk_norm"

echo "node-list: $SLURM_JOB_NODELIST"

srun --comment datanet --cpu_bind=v --accel-bind=gn python -m open_lm.main \
    --train-num-samples $TOKENS \
    --workers 2 \
    --train-data "pipe:aws s3 cp s3://laion-west/rpj_tokenized_upsampled_eleutherai/shard_{00000000..00099999}.tar -" "pipe:aws s3 cp s3://laion-west/2T_no_rpj_tokenized_upsampled_25k_shards/shard_{00000000..00024999}.tar -" \
    --train-data-mix-weights 0.725 0.275 \
    --dataset-resampled \
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
    --report-to wandb \
    --wandb-project-name lm1 \
    --name $EXP_NAME \
    --logs /fsx/home-$USER/experimetns/rms_test \
    --resume latest \
    --seed 124 \
    --data-key 'json' \
    --accum-freq $ACC \
    --model-norm $MODEL_NORM \
    --qk-norm \
    --delete-previous-checkpoint
    # --load-pretrained-state \
    # --lr-cooldown-end 3e-5