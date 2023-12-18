import pytest

from open_lm.utils.transformers.hf_model import OpenLMforCausalLM
from open_lm.utils.transformers.hf_config import OpenLMConfig
from open_lm.model import create_params
from tests.shared import MockTrainArgs
from tests.utils import run_model, CharacterTokenizer


# Download the checkpoint from HuggingFace Hub if it doesn't exist and set the args
@pytest.mark.gpu
@pytest.mark.slow
@pytest.fixture(scope="module")
def args():
    args = MockTrainArgs(
        model="open_lm_test_tiny",
        **{
            # Generation params:
            "input_text": "random",
            "max_gen_len": None,
            "context_len": None,
            "temperature": 0.0,
            "top_p": 1.0,
            "use_cache": False,
            "num_beams": 1,
            # Model params that might not be in config:
            "model_norm": "default_layer_norm",
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
        }
    )
    return args


@pytest.fixture(scope="module")
def tiny_open_lm(args):
    tiny_open_lm = OpenLMforCausalLM(OpenLMConfig(create_params(args)))
    tiny_open_lm.model.eval()
    return tiny_open_lm


# Create a mock tokenizer with a tiny vocab
@pytest.fixture(scope="module")
def tiny_tokenizer():
    # The tiny model has a vocab size of 16, there are 7 special tokens, so we add 9 more
    tokenizer = CharacterTokenizer(["a", "b", "c", "d", "e", "f", "g", "h", "i"])
    return tokenizer


@pytest.mark.parametrize("wiki_page", ["Soil steam sterilization", "The Triumph of Death"])
@pytest.mark.parametrize("context_len", [4, 8])
@pytest.mark.parametrize("max_gen_len", [4, 8])
@pytest.mark.parametrize("num_beams", [1, 4])
def test_tiny_generate_kv_cache(tiny_open_lm, tiny_tokenizer, args, wiki_page, context_len, max_gen_len, num_beams):
    """
    This test checks that the results of the generation are the same with and without cache.
    """
    args.max_gen_len = max_gen_len
    args.context_len = context_len
    args.num_beams = num_beams

    if max_gen_len + context_len > tiny_open_lm.model.seq_len:
        pytest.skip("The model cannot generate sequences that long")

    args.use_cache = False
    result_no_cache1 = run_model(tiny_open_lm, tiny_tokenizer, args, wiki_page=wiki_page, start_index=0)
    result_no_cache2 = run_model(tiny_open_lm, tiny_tokenizer, args, wiki_page=wiki_page, start_index=0)

    # Check that the results are the same without cache (would fail if the sampling was not deterministic)
    assert result_no_cache1 == result_no_cache2

    args.use_cache = True
    result_with_cache = run_model(tiny_open_lm, tiny_tokenizer, args, wiki_page=wiki_page, start_index=0)

    # Check that the results are the same as without cache
    assert result_no_cache1 == result_with_cache
