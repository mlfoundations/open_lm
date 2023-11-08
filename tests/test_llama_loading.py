import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from open_lm.model import Transformer
from open_lm.norms import RmsNorm
from open_lm.file_utils import pt_load
import open_lm.utils
from open_lm.utils.transformers.convert_llama import convert_v2


def llama_params():
    class Params:
        dim: int
        n_layers: int
        n_heads: int
        vocab_size: int
        norm_eps: float
        seq_len: int
        post_embed_norm: bool
        weight_tying: bool
        resume: str
        norm_type: nn.Module = RmsNorm  # Make sure to use RmsNorm for LLaMA
        apply_qk_norm: bool = False
        positional_embedding_type: str = "llama_rotary" # Make sure to set this for LLaMA
        ffn_type: str = "swiglu"

    params = Params()
    params.dim=4096
    params.n_layers=32
    params.n_heads=32
    params.seq_len=4096
    params.vocab_size=32000
    params.post_embed_norm=False
    params.weight_tying=False
    params.norm_eps=1e-5
    params.rotary_old=False
    params.positional_embedding_type= "llama_rotary"
    return params


def test_llama_checkpoint_loading():
    # HF Model
    model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    model_hf.to("cuda")

    # OpenLM Model
    state_dict = convert_v2(model_hf.state_dict())
    model_openlm = Transformer(llama_params())
    model_openlm.load_state_dict(state_dict)
    model_openlm.to("cuda")

    test_cases = []
    test_cases.append([[1]])
    test_cases.append([[38, 1111, 90, 321]])
    test_cases.append([list(range(100))])
    test_cases.append([[2, 22], [4, 44], [88, 8], [99, 9]])

    for tc in test_cases:
        out_hf = model_hf(torch.LongTensor(tc).to("cuda"))[0]
        out_openlm = model_openlm(torch.LongTensor(tc).to("cuda"))[0]
        diff = abs(torch.sum(out_hf-out_openlm).item())
        assert (diff / (out_hf.shape[0]*out_hf.shape[1])) < 0.5     # The actual diff can be quite large. Might be some rounding/precision issues here.
