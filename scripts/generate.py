"""Script to generate text from a trained model using HuggingFace wrappers."""

import argparse
import json
import builtins as __builtin__

import torch

from composer.utils import dist, get_device
from open_lm.utils.transformers.hf_model import OpenLMforCausalLM
from open_lm.utils.transformers.hf_config import OpenLMConfig
from open_lm.utils.llm_foundry_wrapper import SimpleComposerOpenLMCausalLM
from transformers import GPTNeoXTokenizerFast


builtin_print = __builtin__.print


@torch.inference_mode()
def run_model(open_lm: OpenLMforCausalLM, tokenizer, cfg, args):
    dist.initialize_dist(get_device(None), timeout=600)
    input = tokenizer(args.input_text)
    input = {k: torch.tensor(v).unsqueeze(0).cuda() for k, v in input.items()}
    composer_model = SimpleComposerOpenLMCausalLM(open_lm, tokenizer)
    output = composer_model.generate(
        input["input_ids"],
        pad_token_id=50282,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_gen_len,
    )
    output = tokenizer.decode(output[0].cpu().numpy())
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-config")
    parser.add_argument("--input-text", required=True)
    parser.add_argument("--rotary-old", action="store_true")
    parser.add_argument("--qk-norm", action="store_true")
    parser.add_argument('--max-gen-len', default=200, type=int)
    parser.add_argument('--temperature', default=0.8, type=float)
    parser.add_argument('--top-p', default=0.95, type=float)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint)
    with open(args.model_config, "r") as f:
        model_cfg = json.load(f)

    model_cfg["rotary_old"] = args.rotary_old
    model_cfg["qk_norm"] = args.qk_norm

    open_lm = OpenLMforCausalLM(OpenLMConfig(**model_cfg))
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

    state_dict = checkpoint["state_dict"]
    state_dict = {x.replace("module.", ""): y for x, y in state_dict.items()}
    open_lm.model.load_state_dict(state_dict)
    open_lm.eval().cuda()
    run_model(open_lm, tokenizer, model_cfg, args)


if __name__ == "__main__":
    main()
