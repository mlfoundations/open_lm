#!/bin/bash
#SBATCH --partition=g40
#SBATCH --job-name=sopenclip
#SBATCH --nodes 16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --comment=nextgends
#SBATCH --exclude=ip-26-0-134-66,ip-26-0-140-150,ip-26-0-131-89,ip-26-0-133-67
#SBATCH --open-mode=append
#SBATCH --exclusive

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

cd /admin/home-mitchellw/git/open_lm
export PYTHONPATH="$PYTHONPATH:/admin/home-mitchellw/git/open_lm"

BATCHSIZE=10
LR=1e-3
MODEL="m1b_neox"
WD=0.1

EXP_NAME="1p5T-bigdata-neox-$MODEL-$BATCHSIZE-$LR-$WD-nodes16-bs$BATCHSIZE-v0"

echo "node-list: $SLURM_JOB_NODELIST"

srun --comment nextgends --cpu_bind=v --accel-bind=gn python -m open_lm.main \
    --train-num-samples 25000000000 \
    --workers 2 \
    --train-data "pipe:aws s3 cp s3://s-laion/rpj_tokenized_upsampled_eleutherai/shard_{00000000..00099999}.tar -" "pipe:aws s3 cp s3://s-laion/2T_no_rpj_tokenized_upsampled_25k_shards/shard_{00000000..00024999}.tar -" \
    --train-data-mix-weights 0.725 0.275 \
    --dataset-resampled \
    --precision amp_bfloat16 \
    --batch-size $BATCHSIZE \
    --grad-checkpointing \
    --log-every-n-steps 20 \
    --grad-clip-norm 1 \
    --lr $LR \
    --warmup 2000 \
    --model $MODEL \
    --wd $WD \
    --beta2 0.95 \
    --epochs 64 \
    --report-to wandb \
    --wandb-project-name lm1 \
    --name $EXP_NAME \
    --logs /fsx/home-mitchellw/experimetns/lm \
    --resume latest \
    --fsdp \
    --fsdp-limit-all-gathers \
    --fsdp-amp \
    --data-key 'json' \
    --lr-cooldown-end 3e-5