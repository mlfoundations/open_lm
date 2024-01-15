#!/bin/bash
#SBATCH -p <partition>
#SBATCH -t 48:00:00
#SBATCH --job-name=ray_tokenize
#SBATCH --nodes <N>
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=128
#SBATCH --output=%x_%j.out

export PATH="~/miniconda3/condabin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate open_lm  # install according to tng/tools/environment.yml
cd $OPEN_LM_BASE

# The following is the recommended setup from the Ray documentation.

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus ${SLURM_CPUS_PER_TASK} --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus ${SLURM_CPUS_PER_TASK} --block &
    sleep 5
done

# Here we run the actual tokenization command
INPUT=$1
OUTPUT=$2

python -m open_lm.datapreprocess.ray.tokenize_shuffle \
    --input $INPUT \
    --output $OUTPUT \
    --ray_address $ip_head

ray stop