# Mixture of Experts Language Models

## Dependencies

```
pip install megablocks
pip install xformers==0.0.22.post4
```

## Train MoE

To train an MoE, add the `--moe-X` related arguments to the training command:

```
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
    --wandb-project-name moe \
    --name test_moe \
    --logs /fsx/home-$USER/experiments/moe \
    --resume latest \
    --seed 124 \
    --data-key 'json' \
    --accum-freq 4 \
    --model-norm gain_only_layer_norm \
    --fsdp --fsdp-amp \
    --lr-cooldown-end 1e-5 \
    --no-skip-tokens \
    --accurate-total-tokens \
    --moe-freq 2 \
    --moe-num-experts 8 \
    --moe-top-k 2 \
    --moe-capacity-factor 1.25 --moe-loss-weight 0.1
```

The above command will add an MoE FFN layer to every other Transformer block. You can use an arbitrary number of experts; you are only limited by total RAM across all GPUs. 


You can also add the `moe_expert_model_parallelism` which will distribute experts across different GPUs. However, if the #GPU is larger than #Expert, additional #GPU/#Expert tensor parallelism is applied. Currently this is not eval-friendly though, so I would not recommend using it yet.

You can evaluate the MoE in the same way as dense models:

```
torchrun --nproc-per-node 8 -m open_lm.main \
                --val-data "pipe:aws s3 cp s3://laion-west/lmdata/validation_data_tokenized/open_lm//shard_00000000.tar -" \
                --workers 6 \
                --precision amp_bfloat16 \
                --batch-size 8 \
                --log-every-n-steps 1 \
                --model open_lm_41m \
                --fsdp --fsdp-amp \
                --moe-num-experts 64 --moe-freq 2 \
                --data-key json \
                --train-num-samples 1000000000 \
                --model-norm gain_only_layer_norm \
                --name $RANDOM \
                --resume /fsx/home-suching/experiments/mix_wo/test8086/checkpoints/epoch_1.pt \
                --logs /fsx/home-$USER/experiments/eval
```


## Benchmarking

To benchmark your results, here are perplexities we obtain with our implementation across a number of compute budgets and model sizes on our A100 cluster:

### Compute budgets

| Compute type | 41M  | 87M  | 160M | 410M | 830M |
|--------------|------|------|------|------|------|
| Number of nodes | 1   | 1    | 1    | 2 | 4 |
| Number of tokens | 20.0B | 20.0B | 20.0B | 20.0B | 20.0B | 

### Perplexity
| Number of Experts | 41M | 87M | 160M | 410M | 830M  |
|--------------|------|------|------|------|------|
| 1 | 27.61 | 18.68 | 14.87 | 10.54 | 9.39  |  
| 8 | 19.85 | 14.66 | 12.26 | 9.82 | 8.84 |
| 32 | 20.55 | 15.28 |14.62 | | |


### Tokens/sec/GPU 

| Number of Experts | 41M | 87M | 160M | 410M | 830M |
|--------------|------|------|------|------|------|
| 1 | 141.2K | 106.0K | 95.5K | 30.3K | 16.0K |  
| 8 | 69.5K | 66.6K | 66.2K | 18.5K | 9.2K |

### Training Parameters

| Number of Experts | 41M | 87M | 160M | 410M | 830M |
|--------------|------|------|------|------|------|
| 8 experts | 68.9M | 165.4M | 360.6M | 1.1B | 2.4B |
| 32 experts | 164.5M | 439.9M | 1.0B | 3.5B | 7.9B |

### Inference Parameters  

| Number of Experts | 41M | 87M | 160M | 410M | 830M |  
|--------------|------|------|------|------|------|
| 2 experts | 45.0M | 96.8M | 190.7M | 509.2M | 1.1B |