"""Script to generate text from a trained model, without HuggingFace wrappers.

This script is useful for simple generation, and to debug any issues with HuggingFace integration.
The output of this script should match that of generate.py when `--temperature 0` is passed.
"""

# Thanks to Gabriel for this code.
import argparse
import os
import glob
import yaml
from dataclasses import dataclass
from typing import List
from yaml import Loader

import torch
from transformers import GPTNeoXTokenizerFast

from open_lm.model import Transformer, create_model


@dataclass
class GenerationArgs:
    max_gen_len: int = 200
    temperature: float = 0.8
    top_p: float = 0.95


class Generator:
    def __init__(self, model: Transformer):
        self.model = model
        self.tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
        self.pad_token_id = 50282
        self.seq_len = 2048

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        gen_args: GenerationArgs = GenerationArgs(),
    ) -> List[str]:
        bsz = len(prompts)

        prompt_tokens = [self.tokenizer.encode(x) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(self.seq_len, gen_args.max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.pad_token_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.pad_token_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            last_logits = self.model(tokens[:, prev_pos:cur_pos].clone())[0][:, -1, :]
            if gen_args.temperature > 0:
                probs = torch.softmax(last_logits / gen_args.temperature, dim=-1)
                next_token = sample_top_p(probs, gen_args.top_p)
            else:
                next_token = torch.argmax(last_logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            # TODO: enable caching again for inference
            # prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            t = t[: len(prompt_tokens[i]) + gen_args.max_gen_len]
            decoded_i = self.tokenizer.decode(t)

            decoded = []
            for t in decoded_i:
                decoded.append(t)

        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class ModelArgs:
    def __init__(self, path: str):
        with open(path, "r") as f:
            params = yaml.load(f, Loader=Loader)
        for k, v in params.items():
            setattr(self, k, v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    # TODO: Make this take as input --model-config, similar to generate.py
    parser.add_argument("--params", default="")
    parser.add_argument("--wandb-dir", default="")
    parser.add_argument("--input-text", required=True)
    parser.add_argument("--max-gen-len", default=200, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)

    args = parser.parse_args()

    if args.wandb_dir != "":
        if args.params == "":
            args.params = os.path.join(args.wandb_dir, "params.txt")
        if args.checkpoint == "":
            chkpt_dir = os.path.join(args.wandb_dir, "checkpoints", "epoch_*.pt")
            list_of_files = glob.glob(chkpt_dir)
            latest_file = max(list_of_files, key=os.path.getctime)
            args.checkpoint = latest_file
    else:
        assert args.params != "", "Must provide params file or a wandb directory."
        assert args.checkpoint != "", "Must provide checkpoint file or a wandb directory."

    checkpoint = torch.load(args.checkpoint)
    open_lm = create_model(ModelArgs(args.params)).half()

    state_dict = checkpoint["state_dict"]
    state_dict = {x.replace("module.", ""): y for x, y in state_dict.items()}
    open_lm.load_state_dict(state_dict)
    open_lm.eval().cuda()
    generator = Generator(open_lm)
    input_text = [
        args.input_text,
    ]
    output = generator.generate(
        input_text,
        GenerationArgs(args.max_gen_len, args.temperature, args.top_p),
    )
    print("".join(output))


if __name__ == "__main__":
    main()
