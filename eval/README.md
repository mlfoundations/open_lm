## Eval with HELM

### Installing environment

```bash
git clone --depth 1 git@github.com:ruixin31/helm.git -b open_lm-dev
git clone --depth 1 git@github.com:ruixin31/open_lm.git -b wrapped_kv_cached
conda create -n openlm-helm python=3.10
conda activate openlm-helm
pip install -e helm[all]
pip install -e open_lm
pip install datasets~=2.5.2
pip install transformers~=4.33.1
# Note: when xformers == 0.0.22.post7, we observed the error *** RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead. It looks like torch's issue though
# This is working: output.contiguous().view(batchsize, q_len, -1)
# Error line: https://github.com/mlfoundations/open_lm/blob/66efccff9d13b877b802cef334662a82f2ec5c06/open_lm/model.py#L162
# On-going discussion on slack 
pip install xformers==0.0.21
pip install torchvision==0.15.2

```

### Running eval + looking at results
```bash
cd open_lm

python eval/eval_openlm_ckpt_helm.py --model <model_type> --checkpoint <path_to_checkpoint> --experiment <experiment_name>
helm-summarize --suite <experiment_name>
helm-server
```

For example, we can use 

```bash
python eval/eval_openlm_ckpt_helm.py --model open_lm_11m --checkpoint /mmfs1/gscratch/sewoong/rx31/projects/open_lm/path/to/logging/dir/open_lm_ex_21884/checkpoints/epoch_100.pt --experiment default

python eval/eval_openlm_ckpt_helm.py --model open_lm_1b --positional-embedding-type head_rotary --checkpoint /mmfs1/home/rx31/projects/open_lm/open_lm_1b.pt --experiment default
```
To evaluate a 1b or 11m model. Don't forget to add flags such as `--qk-norm`

There are ways to better use multi-gpu, such as there's a built-in slurm runner. To use that, you need to change 
`https://github.com/ruixin31/helm/blob/a6a0cc8667d7e46e4b7a5a609de9d62652a221d4/src/helm/benchmark/slurm_runner.py#L257-L284`, as well as uncomment
`https://github.com/ruixin31/open_lm/blob/27c460e465f91735283409cddfb9dcf4babd2552/eval/eval_openlm_ckpt_helm.py#L140-L142`

### Misc

- HELM is going through a major refactor, and they expect to finish it in the next week. The existing codebase will change and remove some of the hacks I used to make it work. It should get cleaner after that. 

- After installization, you might see the following error. I just ignored it

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
blanc 0.3.3 requires torch<2.0,>=1.0, but you have torch 2.1.0 which is incompatible.
crfm-helm 0.3.0 requires transformers~=4.33.1, but you have transformers 4.34.1 which is incompatible.
Successfully installed accelerate-0.20.3 datasets-2.14.7 open-lm-0.0.16 tokenizers-0.14.1 torch-2.1.0 torchvision-0.16.0 transformers-4.34.1
```

