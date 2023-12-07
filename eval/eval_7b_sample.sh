git clone https://github.com/mosaicml/llm-foundry.git
cp -r llm-foundry/scripts/eval/local_data/* local_data/
python eval_openlm_ckpt.py \
        --eval-yaml in_memory_hf_eval.yaml \
        --model open_lm_7b  \
        --checkpoint $1 \
        --positional-embedding-type head_rotary \
        --qk-norm


