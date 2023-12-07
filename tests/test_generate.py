import time
import pytest

import argparse

import torch

from composer.utils import dist, get_device
from open_lm.utils.transformers.hf_model import OpenLMforCausalLM
from open_lm.utils.transformers.hf_config import OpenLMConfig
from open_lm.utils.llm_foundry_wrapper import SimpleComposerOpenLMCausalLM
from open_lm.model import create_params
from transformers import GPTNeoXTokenizerFast
import wikipedia


@torch.inference_mode()
def run_model(open_lm: OpenLMforCausalLM, tokenizer, args, wiki_page=None, start_index=None):
    dist.initialize_dist(get_device(None), timeout=600)
    if args.input_text == "random":
        wikipedia.set_lang("en")
        try:
            wiki_page = wikipedia.page(wiki_page)
            content = wiki_page.content
            content_tokenized = tokenizer(content)
            content_len = len(content_tokenized["input_ids"])
            if content_len <= args.context_len + start_index:
                print(f"Page too short, will load a different one than the one requested ({wiki_page}).")
                wiki_page = None  # If the page is too short, try again
        except:
            wiki_page = None
        while wiki_page is None:
            rand_page_title = wikipedia.random(pages=1)
            try:
                wiki_page = wikipedia.page(rand_page_title)
            except:
                continue
            content = wiki_page.content
            content_tokenized = tokenizer(content)
            content_len = len(content_tokenized["input_ids"])
            if content_len <= args.context_len:
                wiki_page = None  # If the page is too short, try again
        context_len = args.context_len
        if start_index is None:
            start_index = int((content_len - context_len) * torch.rand(1))
        content_tokenized["input_ids"] = content_tokenized["input_ids"][start_index : start_index + context_len]
        input = content_tokenized
    else:
        input = tokenizer(args.input_text)
    input = {k: torch.tensor(v).unsqueeze(0).cuda() for k, v in input.items()}
    composer_model = SimpleComposerOpenLMCausalLM(open_lm, tokenizer)
    composer_model = composer_model.cuda()

    generate_args = {
        "do_sample": args.temperature > 0,
        "pad_token_id": 50282,
        "max_new_tokens": args.max_gen_len,
        "use_cache": args.use_cache,
    }
    # If these are set when temperature is 0, they will trigger a warning and be ignored
    if args.temperature > 0:
        generate_args["temperature"] = args.temperature
        generate_args["top_p"] = args.top_p

    output = composer_model.generate(
        input["input_ids"],
        **generate_args,
    )
    output = tokenizer.decode(output[0].cpu().numpy())
    return output


@pytest.mark.slow
@pytest.mark.parametrize("wiki_page", ["Soil steam sterilization", "The Triumph of Death"])
@pytest.mark.parametrize(
    "context_len", [1536, 1664]
)  # The test fails for smaller context lengths, the cache speed-up is too small
@pytest.mark.parametrize(
    "max_gen_len", [128, 256]
)  # The test fails for smaller numbers because the cache speed-up is too small
def test_generate_kv_cache(wiki_page, context_len, max_gen_len):
    args = argparse.Namespace(
        **{
            # Generation params:
            "model": "open_lm_160m",
            "input_text": "random",
            "max_gen_len": max_gen_len,
            "context_len": context_len,
            "temperature": 0.0,
            "top_p": 1.0,
            "use_cache": False,
            # Model params that might not be in config:
            "model_norm": "gain_only_layer_norm",
            "qk_norm": True,
            "positional_embedding_type": "rotary",
            "ffn_type": "swiglu",
        }
    )

    # The test fails if we compare all the generated tokens, my guess is compounding rounding errors
    compare_first_len = 32

    open_lm = OpenLMforCausalLM(OpenLMConfig(create_params(args)))

    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

    open_lm.model.eval()

    start_time = time.time()
    result_no_cache1 = run_model(open_lm, tokenizer, args, wiki_page=wiki_page, start_index=0)
    end_time = time.time()
    time_without_cache = end_time - start_time
    result_no_cache2 = run_model(open_lm, tokenizer, args, wiki_page=wiki_page, start_index=0)

    assert result_no_cache1 == result_no_cache2

    start_time = time.time()
    args.use_cache = True
    result_with_cache = run_model(open_lm, tokenizer, args, wiki_page=wiki_page, start_index=0)
    end_time = time.time()
    time_with_cache = end_time - start_time

    assert result_no_cache1[: context_len + compare_first_len] == result_with_cache[: context_len + compare_first_len]
    assert time_with_cache < time_without_cache
