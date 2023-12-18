import time
import pytest

from transformers import GPTNeoXTokenizerFast

from open_lm.utils.transformers.hf_model import OpenLMforCausalLM
from open_lm.utils.transformers.hf_config import OpenLMConfig
from open_lm.model import create_params
from tests.shared import MockTrainArgs
from .utils import run_model


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("wiki_page", ["Soil steam sterilization", "The Triumph of Death"])
@pytest.mark.parametrize("context_len", [256])
@pytest.mark.parametrize("max_gen_len", [1024, 1792])
def test_generate_kv_cache(wiki_page, context_len, max_gen_len):
    """Test that the model generates faster with cache than without."""
    args = MockTrainArgs(
        model="open_lm_160m",
        **{
            # Generation params:
            "input_text": "random",
            "max_gen_len": max_gen_len,
            "context_len": context_len,
            "temperature": 0.0,
            "top_p": 1.0,
            "use_cache": False,
            # Model params that might not be in config:
            "model_norm": "gain_only_layer_norm",
            "qk_norm": False,
            "positional_embedding_type": "rotary",
            "ffn_type": "swiglu",
            "moe_num_experts": None,
            "moe_freq": 0,
            "moe_weight_parallelism": False,
            "moe_expert_model_parallelism": False,
            "moe_capacity_factor": 1.25,
            "moe_loss_weight": 0.1,
            "moe_top_k": 2,
            "num_beams": 1,
        }
    )

    open_lm = OpenLMforCausalLM(OpenLMConfig(create_params(args)))

    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

    open_lm.model.eval()

    start_time = time.time()
    args.use_cache = False
    run_model(open_lm, tokenizer, args, wiki_page=wiki_page, start_index=0)
    end_time = time.time()
    time_without_cache = end_time - start_time

    start_time = time.time()
    args.use_cache = True
    run_model(open_lm, tokenizer, args, wiki_page=wiki_page, start_index=0)
    end_time = time.time()
    time_with_cache = end_time - start_time

    assert time_with_cache < time_without_cache
