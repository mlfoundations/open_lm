import argparse
import builtins as __builtin__
import time
from typing import List

import torch
from composer.loggers import InMemoryLogger, LoggerDestination
from composer.trainer import Trainer
from composer.utils import dist, get_device, reproducibility

try:
    from llmfoundry.utils.builders import build_icl_evaluators, build_logger
except ImportError:
    import logging

    logging.warning("llmfoundry not installed. Please install llmfoundry `pip install llm-foundry` to run this script.")

from omegaconf import OmegaConf as om
from transformers import GPTNeoXTokenizerFast, LlamaTokenizerFast

from open_lm.model import create_params
from open_lm.params import add_model_args
from open_lm.utils.llm_foundry_wrapper import SimpleComposerOpenLMCausalLM
from open_lm.utils.transformers.hf_config import OpenLMConfig
from open_lm.utils.transformers.hf_model import OpenLMforCausalLM

builtin_print = __builtin__.print


def setup_for_distributed(is_master):
    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


@torch.no_grad()
def evaluate(model, tokenizer, cfg):
    cfg.dist_timeout = cfg.get("dist_timeout", 600.0)

    reproducibility.seed_all(cfg.seed)
    dist.initialize_dist(get_device(None), timeout=cfg.dist_timeout)
    setup_for_distributed(dist.get_global_rank() == 0)

    composer_model = SimpleComposerOpenLMCausalLM(model, tokenizer)

    evaluators, logger_keys = build_icl_evaluators(
        cfg.icl_tasks, tokenizer, cfg.max_seq_len, cfg.device_eval_batch_size
    )

    in_memory_logger = InMemoryLogger()  # track metrics in the in_memory_logger
    loggers: List[LoggerDestination] = [
        build_logger(name, logger_cfg) for name, logger_cfg in (cfg.get("loggers") or {}).items()
    ]
    loggers.append(in_memory_logger)

    fsdp_config = cfg.get("fsdp_config", None)
    fsdp_config = om.to_container(fsdp_config, resolve=True) if fsdp_config is not None else None

    load_path = cfg.get("load_path", None)

    trainer = Trainer(
        model=composer_model,
        loggers=loggers,
        precision=cfg.precision,
        fsdp_config=fsdp_config,  # type: ignore
        load_path=load_path,
        load_weights_only=True,
        progress_bar=False,
        log_to_console=True,
        dist_timeout=cfg.dist_timeout,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    a = time.time()
    trainer.eval(eval_dataloader=evaluators)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    b = time.time()

    print(f"Ran eval in: {b-a} seconds")

    for key in logger_keys:
        if key in in_memory_logger.data:
            result = in_memory_logger.data[key][0][1].item()
            print(f"{key}: {result}")


def main():
    """
    Usage:
    python eval_openlm_ckpt.py --checkpoint <path_to_openlm_checkpoint>  --model <name_of_model_config> --eval-yaml <path_to_eval_yaml> --tokenizer <tokenizer_name_or_path>
    example:
    cd eval
    python eval_openlm_ckpt.py --checkpoint ../checkpoints/llama2_7b.pt --model llama2_7b.json --eval-yaml in_memory_hf_eval.yaml --tokenizer <path_to_tokenizer>
    multi-gpu example:
    cd eval
    torchrun --nproc_per_node 3 python eval_openlm_ckpt.py --checkpoint ../checkpoints/llama2_7b.pt --model llama2_7b.json --eval-yaml in_memory_hf_eval.yaml --tokenizer <path_to_tokenizer>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--model", type=str, default="m1b_neox", help="Name of the model to use.")
    parser.add_argument("--eval-yaml")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    add_model_args(parser)
    args = parser.parse_args()

    with open(args.eval_yaml) as f:
        eval_cfg = om.load(f)

    print("Loading checkpoint from disk")
    checkpoint = torch.load(args.checkpoint)

    print("Loading model into the right classes")
    open_lm = OpenLMforCausalLM(OpenLMConfig(create_params(args)))
    if "gpt-neox-20b" in args.tokenizer:
        tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    elif "llama" in args.tokenizer:
        tokenizer = LlamaTokenizerFast.from_pretrained(args.tokenizer)

    state_dict = checkpoint["state_dict"]
    state_dict = {x.replace("module.", ""): y for x, y in state_dict.items()}
    open_lm.model.load_state_dict(state_dict)
    open_lm.model.eval()

    evaluate(open_lm, tokenizer, eval_cfg)


if __name__ == "__main__":
    main()
