# OpenLM
OpenLM is a minimal but performative language modeling (LM) repository, aimed to facilitate research on medium sized LMs. We have verified the performance of OpenLM up to 7B parameters and 256 GPUs, with larger scales planned.

# Contents
- [Release Notes](#release-notes)
- [Quickstart](#quickstart)
  - [Setup](#setup)
  - [Process training data](#process-training-data)
  - [Run training](#run-training)
  - [Evaluate Model](#evaluate-model)
- [Pretrained Models](#pretrained-models)
- [Team and Acknowledgements](#team-and-acknowledgements)

# Release Notes
- 08/18/23: Updated README.md
# Quickstart
Here we'll go over a basic example where we start from a fresh install, download and preprocess some training data, and train a model.

## Setup
We require python >=3.9, and a current installation of pyTorch, as well as several other packages. The full list of requirements is contained in `requirements.txt` and can be installed in your python enviornment via
```>>> pip install -r requirements.txt```
Some considerations:
- We like [WandB](https://wandb.ai/) and [tensorboard](https://www.tensorflow.org/tensorboard) for logging. We specify how to use these during training below.

## Process Training Data
Next you must specify a collection of tokenized data. For the purposes of this example, we will use the [Pile dataset](https://the-eye.eu/public/AI/pile/train/). If you want to download this locally, here's a bash incantation to do this (you'll likely want to do this in a detached screen, preferably overnight).
```
#!/bin/bash
mkdir raw_data    ### or wherever you want to store raw data
cd raw_data
for i in {0..29}; do  ### change 0..29 if you want fewer files
  url=$(printf "https://the-eye.eu/public/AI/pile/train/%02d.jsonl.zst" "$i")
  wget $url
done
```

Next we process our training data by running it through a BPE tokenizer and chunk it into chunks of appropriate length. By default we use the tokenizer attached with [GPT-NeoX-20B](https://github.com/EleutherAI/gpt-neox). To do this, use the script `datapreprocess/make_2048.py`:
```
>>> python datapreprocess/make_2048.py \
    --input-files raw_data/*.jsonl
    --output-dir preproc_data
    --num-workers 32
    --num-consumers 1
```
Where `input-files` passes all of its (possibly many) arguments through the python `glob` module, allowing for wildcards. Optionally, data can be stored in S3 by setting the environment variables: `S3_BASE`,  and passing the flag `--upload-to-s3` to the script. This saves sharded data to the given bucket with prefix of `S3_BASE`. E.g.
```
>>> export S3_BASE=preproc_data-v1/
>>> python datapreprocess/make2048.py --upload-to-s3 ... # same arguments as before
```

## Run Training
Tokenized data can now be passed to the main training script, `open_lm/main.py`. Distributed computatation is handled via `torchrun`, and hyperparameters are specified by a variety of keyword arguments. We highlight several of the most important ones here:
- `train-data`: location of the sharded tokenized training data. If locally generated and stored, this will point to a directory containing files like `preproc_data/2048-v1/0/XXXXXXX.tar`. Data are processed using the [webdataset](https://github.com/webdataset/webdataset) package where wildcards are supported like `preproc_data/2048-v1/0/{0000000..0000099}.tar` to select the first 100 .tar files.
- `model`: Which model to use. See the table below to see valid options and parameter sizes for each.
- `train-num-samples`: how many samples to use from the specified training dataset
- `name`: name of this particular training run for logging purposes
- `report-to`: if present, can be `wandb`, `tensorboard`, or `all` to stash logging information on WandB or Tensorboard.


Model choices are contained in the following table, where, for instance `11m` indicates an 11 million parameter model and `1b` indicates a 1 billion parameter model.
<center>

| Model Name    |
|---------------|
| `open_lm_11m` |
| `open_lm_25m` |
| `open_lm_87m` |
| `open_lm_160m`|
| `open_lm_411m`|
| `open_lm_830m`|
| `open_lm_1b`  |
| `open_lm_3b`  |
| `open_lm_7b`  |

</center>

An example training run can be called as follows:
```
>>> export CUDA_VISIBLE_DEVICES=0,1,2,3
>>> torchrun --nproc-per-node 4 -m open_lm.main   \
 --model open_lm_3b \
 --train-data /preproc_data/shard-{0000000..0000099}.tar \
 --train-num-samples 1000000000 \
 --workers 8 \
 --dataset-resampled \
 --precision amp_bfloat16 \
 --batch-size 8 \
 --grad-checkpointing \
 --log-every-n-steps 100 \
 --grad-clip-norm 1 \
 --data-key txt \
 --lr 3e-4 \
 --fsdp --fsdp-amp \
 --warmup 2000 \
 --wd 0.1 \
 --beta2 0.95 \
 --epochs 100 \
 --report-to wandb \
 --wandb-project-name open_lm_example \
 --name open_lm_ex_$RANDOM \
 --resume latest \
 --logs path/to/logging/dir/
```
Checkpoints and final model weights will be saved to the specified logs directory.

## Evaluate Model
Once trained, we can evaluate the model. This requires [LLM Foundry](https://github.com/mosaicml/llm-foundry), which can be installed via `pip install llm-foundry`. Next some configurations are required to pass to the evaluator: a skeleton of these parameters is located at [eval/in_memory_hf_eval.yaml](eval/in_memory_hf_eval.yaml). Then just run the following script, making sure to point it at the checkpoint of your trained model (and it's correspending config .json file): 
```
cd eval
python eval_openlm_ckpt.py \
--eval-yaml in_memory_hf_eval.yaml \
--model-config ../open_lm/model_configs/open_lm_3b.json  \ --checkpoint /path/to/openlm_checkpoint.pt
```


# Pretrained Models

## OpenLM 1B
OpenLM 1B is a ~1Billion parameter model trained on a 1.6T token dataset which consists of a mix of RedPajama, Pile, S2ORC, The Pile of Law, Deepmind Math, and RealNews (the full mixture of training data is described in [more detail here](https://docs.google.com/spreadsheets/d/1YW-_1vGsSPmVtEt2oeeJOecH6dYX2SuEuhOwZyGwy4k/edit?usp=sharing)). The model checkpoint can be downloaded from [HuggingFace here](https://huggingface.co/mlfoundations/open_lm_1B/tree/main). The script used to train this model (for config-copying purposes) is [located here](https://github.com/mlfoundations/open_lm/blob/main/scripts/train_example.sh). Once this checkpoint has been downloaded, you can evaluate it by following the directions in the [Evaluate Model](#evaluate-model) section above:



| **OpenLM-1B** | **250B Tokens** | **500B tokens** | **750B tokens** | **1T Tokens** | **1.25T Tokens** | **1.5T Tokens** | **1.6T Tokens** |
|----------------|-----------------|-----------------|-----------------|---------------|------------------|-----------------|-----------------|
|                |                 |                 |                 |               |                  |                 |                 |
| arc_challenge  |            0.27 |            0.28 |            0.29 |          0.28 |             0.29 |            0.31 |            0.31 |
| arc_easy       |            0.49 |            0.50 |            0.51 |          0.53 |             0.54 |            0.56 |            0.56 |
| boolq          |            0.60 |            0.61 |            0.62 |          0.62 |             0.65 |            0.64 |            0.65 |
| copa           |            0.71 |            0.70 |            0.70 |          0.78 |             0.71 |            0.73 |            0.70 |
| hellaswag      |            0.50 |            0.54 |            0.54 |          0.57 |             0.59 |            0.61 |            0.61 |
| lambada_openai |            0.56 |            0.57 |            0.61 |          0.61 |             0.65 |            0.65 |            0.66 |
| piqa           |            0.70 |            0.70 |            0.71 |          0.72 |             0.73 |            0.74 |            0.74 |
| triviaqa       |                 |                 |                 |               |                  |                 |                 |
| winogrande     |            0.55 |            0.57 |            0.58 |          0.59 |             0.61 |            0.60 |            0.60 |
| MMLU           |            0.24 |            0.24 |            0.24 |          0.23 |             0.26 |            0.24 |            0.25 |
| Jeopardy       |            0.01 |            0.02 |            0.01 |          0.01 |             0.04 |            0.09 |            0.10 |
| Winograd       |            0.75 |            0.77 |            0.77 |          0.79 |             0.81 |            0.80 |            0.79 |
|                |                 |                 |                 |               |                  |                 |                 |
| **Average**    |        **0.49** |        **0.50** |        **0.51** |      **0.52** |         **0.53** |        **0.54** |        **0.54** |


| **1B Baselines** | **OPT-1.3B** | **Pythia-1B** | **Neox-1.3B** | **OPT-IML-1.3B** |
|------------------|-------------:|--------------:|--------------:|-----------------:|
| arc_challenge    |         0.27 |          0.26 |          0.26 |             0.30 |
| arc_easy         |         0.49 |          0.51 |          0.47 |             0.58 |
| boolq            |         0.58 |          0.61 |          0.62 |             0.72 |
| copa             |         0.75 |          0.68 |          0.72 |             0.73 |
| hellaswag        |         0.54 |          0.49 |          0.48 |             0.54 |
| lambada_openai   |         0.59 |          0.58 |          0.57 |             0.57 |
| piqa             |         0.72 |          0.70 |          0.72 |             0.73 |
| triviaqa         |              |               |               |                  |
| winogrande       |         0.59 |          0.53 |          0.55 |             0.59 |
| MMLU             |         0.25 |          0.26 |          0.26 |             0.30 |
| Jeopardy         |         0.01 |          0.00 |          0.00 |             0.12 |
| Winograd         |         0.74 |          0.71 |          0.75 |             0.73 |
| **Average**      |     **0.50** |      **0.48** |      **0.49** |         **0.54** |


## OpenLM 7B
OpenLM 7B is not yet done training.


| **OpenLM-7B**  | **275B Tokens** | **500B tokens** | **675B tokens** | **775B tokens** | **1T Tokens** | **1.25T Tokens** | **1.5T Tokens** | **1.6T Tokens** | **LLAMA-7B** | **MPT-7B** |
|-----------------|-----------------|-----------------|-----------------|-----------------|---------------|------------------|-----------------|-----------------|--------------|------------|
| arc_challenge   |            0.35 |            0.35 |            0.36 |            0.37 |          0.39 |             0.39 |                 |                 |         0.41 |       0.39 |
| arc_easy        |            0.60 |            0.61 |            0.62 |            0.62 |          0.63 |             0.66 |                 |                 |         0.65 |       0.67 |
| boolq           |            0.67 |            0.66 |            0.69 |            0.69 |          0.70 |             0.70 |                 |                 |         0.77 |       0.75 |
| copa            |            0.75 |            0.79 |            0.75 |            0.80 |          0.80 |             0.78 |                 |                 |         0.78 |       0.81 |
| hellaswag       |            0.64 |            0.67 |            0.68 |            0.68 |          0.69 |             0.70 |                 |                 |         0.75 |       0.76 |
| lambada_openai  |            0.67 |            0.68 |            0.69 |            0.70 |          0.70 |             0.70 |                 |                 |         0.74 |       0.70 |
| piqa            |            0.75 |            0.76 |            0.76 |            0.76 |          0.77 |             0.77 |                 |                 |         0.79 |       0.80 |
| triviaqa        |                 |                 |                 |                 |               |                  |                 |                 |              |            |
| winogrande      |            0.62 |            0.65 |            0.65 |            0.65 |          0.67 |             0.67 |                 |                 |         0.68 |       0.68 |
| MMLU-0 shot     |            0.25 |            0.25 |            0.27 |            0.27 |          0.28 |             0.30 |                 |                 |         0.30 |       0.30 |
| Jeopardy        |            0.15 |            0.18 |            0.23 |            0.22 |          0.16 |             0.21 |                 |                 |         0.33 |       0.31 |
| Winograd        |            0.82 |            0.81 |            0.84 |            0.84 |          0.85 |             0.86 |                 |                 |         0.81 |       0.88 |
|                 |                 |                 |                 |                 |               |                  |                 |                 |              |            |
| **Average**     |        **0.57** |        **0.58** |        **0.60** |        **0.60** |      **0.60** |         **0.61** |                 |                 |     **0.64** |   **0.64** |
| **MMLU-5 shot** |                 |                 |                 |                 |               |         **0.34** |                 |                 |     **0.34** |            |



# Team and acknowledgements

Team (so-far, * = equal contrib): Suchin Gururangan*, Mitchell Wortsman*, Samir Yitzhak Gadre, Achal Dave, Maciej Kilian, Weijia Shi, Georgios Smyrnis, Gabriel Ilharco, Matt Jordan, Reinhard Heckel, Alex Dimakis, Ali Farhadi, Ludwig Schmidt.

Code is based heavily on [open-clip](https://github.com/mlfoundations/open_clip) developed by a team including Ross Wightman, Romain Beaumont, Mehdi Cherti, Jenia Jitsev, and [open-flamingo](https://github.com/mlfoundations/open_flamingo), developed by a team including Anas Awadalla and Irena Gao. Additional inspiration is from [lit-llama](https://github.com/Lightning-AI/lit-llama).
We are greatful to stability.ai for resource support.
