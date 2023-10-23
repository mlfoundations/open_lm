from dataclasses import fields
import argparse
from typing import Dict, Union
import json
import os
from copy import deepcopy
from open_lm.params import add_model_args


def do_wrap(
    checkpoint: Union[str, os.PathLike],
    params,
    out_dir: Union[str, os.PathLike],
    soft_link: bool,
):
    from open_lm.utils.transformers.hf_config import OpenLMConfig
    from transformers import GPTNeoXTokenizerFast, LlamaTokenizerFast

    os.makedirs(out_dir, exist_ok=True)

    config = OpenLMConfig(params)
    output_config_dict = config.to_diff_dict()
    output_config_dict["auto_map"] = {
        "AutoConfig": "imports.OpenLMConfig",
        "AutoModelForCausalLM": "imports.OpenLMforCausalLM",
    }
    json.dump(
        output_config_dict, open(os.path.join(out_dir, "config.json"), "w"), indent=2
    )

    # Imports file
    imports_file = "from open_lm.utils.transformers.hf_model import OpenLMforCausalLM\nfrom open_lm.utils.transformers.hf_config import OpenLMConfig"
    with open(os.path.join(out_dir, "imports.py"), "w") as f:
        f.write(imports_file)

    # Checkpoint file
    checkpoint_file = os.path.join(out_dir, "checkpoint.pt")
    if soft_link:
        os.symlink(os.path.realpath(checkpoint), checkpoint_file)
    else:
        import shutil

        shutil.copyfile(checkpoint, checkpoint_file)
    

    # Tokenizer files
    if "gpt-neox-20b" in args.tokenizer:
        tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    elif "llama" in args.tokenizer:
        tokenizer = LlamaTokenizerFast.from_pretrained(args.tokenizer)
    tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--out-dir")
    parser.add_argument("--soft-link", default=True, action="store_true")

    parser.add_argument(
        "--model", type=str, default="m1b_neox", help="Name of the model to use."
    )
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    add_model_args(parser)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    out_dir = args.out_dir
    
    if not os.path.exists(checkpoint):
        raise ValueError(f"Checkpoint {checkpoint} does not exist.")

    if os.path.exists(out_dir):
        raise ValueError(f"Output directory {out_dir} already exists.")

    from open_lm.model import create_params

    do_wrap(args.checkpoint, create_params(args), args.out_dir, args.soft_link)
