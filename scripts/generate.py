"""Script to generate text from a trained model using HuggingFace wrappers."""

import argparse
import json
import builtins as __builtin__

import torch

from composer.utils import dist, get_device
from open_lm.utils.transformers.hf_model import OpenLMforCausalLM
from open_lm.utils.transformers.hf_config import OpenLMConfig
from open_lm.utils.llm_foundry_wrapper import SimpleComposerOpenLMCausalLM
from open_lm.model import create_params
from open_lm.params import add_model_args
from transformers import GPTNeoXTokenizerFast


builtin_print = __builtin__.print


@torch.inference_mode()
def run_model(open_lm: OpenLMforCausalLM, tokenizer, args):
    dist.initialize_dist(get_device(None), timeout=600)
    input = tokenizer(args.input_text)
    input = {k: torch.tensor(v).unsqueeze(0).cuda() for k, v in input.items()}
    composer_model = SimpleComposerOpenLMCausalLM(open_lm, tokenizer)
    composer_model = composer_model.cuda()
    output = composer_model.generate(
        input["input_ids"],
        pad_token_id=50282,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_gen_len,
    )
    output = tokenizer.decode(output[0].cpu().numpy())
    print("-" * 50)
    print("\t\t Model output:")
    print("-" * 50)
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--model", type=str, default="open_lm_1b", help="Name of the model to use")

    parser.add_argument("--input-text", required=True)
    parser.add_argument("--max-gen-len", default=200, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)

    add_model_args(parser)
    args = parser.parse_args()
    print("Loading checkpoint from disk...")
    checkpoint = torch.load(args.checkpoint)
    dist.initialize_dist(get_device(None), timeout=600)
    print("Loading model into the right classes...")
    open_lm = OpenLMforCausalLM(OpenLMConfig(create_params(args)))
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

    state_dict = checkpoint["state_dict"]
    state_dict = {x.replace("module.", ""): y for x, y in state_dict.items()}
    open_lm.model.load_state_dict(state_dict)
    open_lm.model.eval()

    run_model(open_lm, tokenizer, args)


if __name__ == "__main__":
    main()
