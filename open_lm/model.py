import math
import json
import re
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F

import xformers.ops as xops

from huggingface_hub import PyTorchModelHubMixin

from open_lm.norms import get_norm_class
from open_lm.positional_embedding.head_rotary import HeadRotaryWithCast
from open_lm.positional_embedding.rotary import RotaryWithCast
from open_lm.positional_embedding.llama_rotary import LLaMARotaryWithCast
# from open_lm.moe.mixture_of_experts import MoE
from megablocks.layers.moe import MoE
from megablocks.layers.arguments import Arguments as MoEArgs


try:  # optional import
    from mamba_ssm import MambaLMHeadModel
except ImportError:
    MambaLMHeadModel = None

# from openclip
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs(model_config_paths=None):
    global _MODEL_CONFIGS

    config_iter = None
    if model_config_paths is not None:
        config_iter = [
            Path(model_config_paths),
        ]
    else:
        config_iter = _MODEL_CONFIG_PATHS

    config_ext = (".json",)
    config_files = []
    for config_path in config_iter:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(Path(config_path))
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


# args and default params follow llama (except with LayerNorm instead of RmsNorm)
@dataclass
class Params:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1
    norm_eps: float = 1e-5
    seq_len: int = 2048
    post_embed_norm: bool = False
    weight_tying: bool = False
    norm_type: nn.Module = nn.LayerNorm
    apply_qk_norm: bool = False
    moe_loss_weight: float = 0.1
    moe_capacity_factor: float = 1.25
    moe_expert_model_parallelism: bool = False
    moe_weight_parallelism: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_freq : int = 0
    positional_embedding_type: str = "rotary"
    ffn_type: str = "swiglu"


def get_rectangular_mask(shape, q_seq_len, k_seq_len, device, dtype):
    # xformers requires the mask to be built with a shape that is a multiple of 8
    # probably because of the way it is implemented in CUDA
    next_multiple_8 = (k_seq_len + 7) // 8 * 8  #
    mask = torch.ones((q_seq_len, next_multiple_8), device=device, dtype=bool)
    mask[:, -q_seq_len:] = torch.tril(mask[:, -q_seq_len:], diagonal=0)
    return torch.zeros((*shape, q_seq_len, next_multiple_8), device=device, dtype=dtype).masked_fill(
        ~mask, float("-inf")
    )[:, :, :, :k_seq_len]


def xformers_attn(queries, keys, values, is_causal):
    # xformers assumes q, k, v are [batch, seq_len, heads, embed_dim]
    # We assume that queries match the last part of the key / value sequences
    # see (https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.fmha.attn_bias.LowerTriangularFromBottomRightMask)
    # we would like to replace the mask generation with: mask = xops.fmha.attn_bias.LowerTriangularFromBottomRightMask()
    # sadly we cannot us this because it needs xformers>=0.0.23 and this is not compatible with torch<2.1.1 while llm-foundry requires torch<2.1.1

    mask = None
    # If queries have shape [batch, 1, heads, dim] it means there is only one query in the sequence.
    # In this case, there is no notion of causal masking, so we can just set the mask to None.
    # This is actually needed to get the desired behavior with seq_len=1.
    if is_causal and queries.shape[1] == keys.shape[1]:
        mask = xops.LowerTriangularMask()
    elif is_causal and queries.shape[1] > 1:
        # Build causal mask that assumes queries are in the end of the sequence.
        batch, q_seq_len, heads, _ = queries.shape
        k_seq_len = keys.shape[1]
        mask = get_rectangular_mask((batch, heads), q_seq_len, k_seq_len, queries.device, queries.dtype)
    return xops.memory_efficient_attention(queries, keys, values, attn_bias=mask)


def torch_attn(queries, keys, values, is_causal):
    # Need to call contiguous in torch >=2.1, otherwise later calls to .view() fail.
    # Possibly related: https://github.com/pytorch/pytorch/issues/110213 - behavior of scaled_dot_product_attention
    # changed between 2.0 and 2.1
    if is_causal and keys.shape[1] > queries.shape[1] > 1:
        q_seq_len = queries.shape[1]
        k_seq_len = keys.shape[1]
        # Same as above, we would like to use:
        # mask = xops.fmha.attn_bias.LowerTriangularFromBottomRightMask().materialize((1, 1, q_seq_len, k_seq_len), queries.dtype, queries.device)
        mask = get_rectangular_mask((1, 1), q_seq_len, k_seq_len, queries.device, queries.dtype)
        return (
            F.scaled_dot_product_attention(
                queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), attn_mask=mask
            )
            .transpose(1, 2)
            .contiguous()
        )
    elif queries.shape[1] == 1:
        return (
            F.scaled_dot_product_attention(queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2))
            .transpose(1, 2)
            .contiguous()
        )
    else:
        return (
            F.scaled_dot_product_attention(
                queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), is_causal=is_causal
            )
            .transpose(1, 2)
            .contiguous()
        )


def get_pos_embed(args: Params):
    head_dim = args.dim // args.n_heads
    if args.positional_embedding_type == "rotary":
        return RotaryWithCast(head_dim, args.seq_len)
    elif args.positional_embedding_type == "llama_rotary":
        return LLaMARotaryWithCast(head_dim, args.n_heads, args.seq_len)
    elif args.positional_embedding_type == "head_rotary":
        return HeadRotaryWithCast(head_dim, args.seq_len)
    else:
        raise RuntimeError(f"Unknown positional embedding type {args.positional_embedding_type}")


class CustomAttn(nn.Module):
    def __init__(self, layer_id, args: Params):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.in_proj = nn.Linear(args.dim, 3 * args.n_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.pos_embed = get_pos_embed(args)
        self.attn_fn = xformers_attn if torch.cuda.is_available() else torch_attn
        self.apply_qk_norm = args.apply_qk_norm

        # initialize norm layers for queries and keys if needed
        self.q_norm = (
            args.norm_type(
                args.n_heads * self.head_dim,
                eps=args.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            args.norm_type(
                args.n_heads * self.head_dim,
                eps=args.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )

        self.layer_id = layer_id
        self.dim = args.dim
        self.reset_parameters()

    def reset_parameters(self):
        # initialize weights by trunc_normal(1/sqrt(fan_in))
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.in_proj.weight, std=std, a=-3 * std, b=3 * std)
        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = std / math.sqrt(2 * (self.layer_id + 1))
        torch.nn.init.trunc_normal_(self.out_proj.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor, is_causal=True, past_key_value=None, use_cache=False):
        batchsize, q_len, _ = x.shape
        queries, keys, vals = self.in_proj(x).chunk(3, dim=-1)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = queries.view(batchsize, q_len, self.n_heads, self.head_dim)
        keys = keys.view(batchsize, q_len, self.n_heads, self.head_dim)
        vals = vals.view(batchsize, q_len, self.n_heads, self.head_dim)

        past_length = 0 if past_key_value is None else past_key_value[0].shape[1]
        queries, keys, vals = self.pos_embed(queries, keys, vals, offset=past_length)

        if past_key_value is not None and use_cache:
            keys = torch.cat([past_key_value[0], keys], dim=1)
            vals = torch.cat([past_key_value[1], vals], dim=1)

        if use_cache:
            past_key_value = [keys, vals]

        output = self.attn_fn(
            queries,
            keys,
            vals,
            is_causal=is_causal,
        )

        output = output.view(batchsize, q_len, -1)

        return self.out_proj(output), past_key_value

class Block(nn.Module):
    def __init__(self, layer_id, args: Params):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        
        self.head_dim = args.dim // args.n_heads
        self.attention = CustomAttn(layer_id, args)
        self._ffn_type = args.ffn_type
        if args.ffn_type == "swiglu":
            # this follows llama / lit llama -- go to multiple of 256
            self.hidden_dim = 256 * ((int(2 * 4 * args.dim / 3) + 256 - 1) // 256)
            self.feed_forward = xops.SwiGLU(args.dim, self.hidden_dim, args.dim, bias=False)
        elif args.ffn_type == "gelu":
            # Follows mosaic mpt7b, but without a bias.
            self.hidden_dim = args.dim * 4
            self._ff_w1 = nn.Linear(args.dim, self.hidden_dim, bias=False)
            self._ff_w2 = nn.Linear(self.hidden_dim, args.dim, bias=False)
            self.feed_forward = nn.Sequential(self._ff_w1, nn.GELU(approximate="none"), self._ff_w2)
        elif args.ffn_type == "moe":
            moe_args = MoEArgs(hidden_size=args.dim,
                               ffn_hidden_size=args.dim * 4,
                               moe_num_experts=args.moe_num_experts,
                               moe_weight_parallelism=args.moe_weight_parallelism,
                               moe_expert_model_parallelism=args.moe_expert_model_parallelism,
                               moe_top_k=args.moe_top_k,
                               moe_capacity_factor=args.moe_capacity_factor,
                               moe_loss_weight=args.moe_loss_weight,
                               device=torch.cuda.current_device(),
                               bf16=False,
                               fp16=False)
            self.feed_forward = MoE(moe_args)



        self.layer_id = layer_id
        self.attention_norm = args.norm_type(
            args.dim,
            eps=args.norm_eps,
        )
        self.ffn_norm = args.norm_type(
            args.dim,
            eps=args.norm_eps,
        )
        self.attention.seq_len = args.seq_len
        self.reset_parameters()

    def reset_parameters(self):
        if self._ffn_type == "swiglu":
            # initialize weights trunc_normal(1/sqrt(fan_in))
            std = 1.0 / math.sqrt(self.dim)
            torch.nn.init.trunc_normal_(self.feed_forward.w12.weight, std=std, a=-3 * std, b=3 * std)
            # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
            std = 1.0 / math.sqrt(self.hidden_dim)
            std = std / math.sqrt(2 * (self.layer_id + 1))
            torch.nn.init.trunc_normal_(self.feed_forward.w3.weight, std=std, a=-3 * std, b=3 * std)
        elif self._ffn_type == "gelu":
            std = 1.0 / math.sqrt(self.dim)
            torch.nn.init.trunc_normal_(self._ff_w1.weight, std=std, a=-3 * std, b=3 * std)

            std = 1.0 / math.sqrt(self.hidden_dim)
            std = std / math.sqrt(2 * (self._layer_id + 1))
            torch.nn.init.trunc_normal_(self._ff_w2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x, past_key_value=None, use_cache=False):
        h, past_key_value = self.attention(
            self.attention_norm(x),
            is_causal=True,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        h = x + h
        if self.ffn_type == "moe":
            ffn_out, _ = self.feed_forward(self.ffn_norm(h))
        else:
            ffn_out = self.feed_forward(self.ffn_norm(h))
        out = h + ffn_out
        return out, past_key_value
    

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x), is_causal=True)
        if self.ffn_type == "moe":
            ffn_out, _ = self.feed_forward(self.ffn_norm(h))
        else:
            ffn_out = self.feed_forward(self.ffn_norm(h))
        out = h + ffn_out
        return out


class Transformer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, params):
        super().__init__()
        # for convenience we often share param names with llama
        self.params = params
        self.dim = params.dim
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.moe_num_experts = params.moe_num_experts
        self.seq_len = params.seq_len
        self.post_embed_norm = (
            params.norm_type(
                params.dim,
                eps=params.norm_eps,
            )
            if params.post_embed_norm
            else nn.Identity()
        )
        self.weight_tying = params.weight_tying

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        ffn_type_ = params.ffn_type
        for layer_id in range(params.n_layers):
            if params.moe_freq > 0 and layer_id % params.moe_freq == 0:
                params.ffn_type = "moe"
            else:
                params.ffn_type = ffn_type_
            self.layers.append(Block(layer_id, params))

        # get class for normalization layers
        self.norm = params.norm_type(
            params.dim,
            eps=params.norm_eps,
        )
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        if self.weight_tying:
            self.tok_embeddings.weight = self.output.weight
        self.grad_checkpointing = False
        self.reset_parameters()

    def reset_parameters(self):
        # initialize weight 1/sqrt(dim)
        # this is 1/fan_in for output, as is default, and Maciej Kilian tried another option
        # for the embed layer (from RWKV paper) but this was better.
        std = 1.0 / math.sqrt(self.params.dim)
        torch.nn.init.trunc_normal_(self.output.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.tok_embeddings.weight, std=std, a=-3 * std, b=3 * std)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(self, input, past_key_values=None, use_cache=False):
        x = self.tok_embeddings(input)
        x = self.post_embed_norm(x)

        if past_key_values is None:
            past_key_values = [None] * self.n_layers
        elif isinstance(past_key_values, tuple):
            past_key_values = list(past_key_values)
        for i, layer in enumerate(self.layers):
            if self.grad_checkpointing:
                x, past_key_values[i] = checkpoint(layer, x, past_key_values[i], use_cache)
            else:
                x, past_key_values[i] = layer(x, past_key_values[i], use_cache=use_cache)
        if past_key_values[0] is None:
            past_key_values = None
        x = self.norm(x)
        output = self.output(x)
        # follow llama in casting this to float.
        return output.float(), x, past_key_values

    def get_input_embeddings(self):
        return self.tok_embeddings

    def get_output_embeddings(self):
        return self.output


def create_params(args):
    cfg = None

    if args.model.endswith(".json"):
        _rescan_model_configs(model_config_paths=args.model)
        args.model = Path(args.model).stem

    if args.model in _MODEL_CONFIGS:
        cfg = deepcopy(_MODEL_CONFIGS[args.model])
    else:
        raise ValueError("Pass a pre-defined open_lm model name or a json config")

    # Note: here all the parameters should come from the config file
    # but for retro-compatibility, we add new model parameters to the args (with a default value that matches the old version)
    # These args are managed separately by the argparser
    # If a parameter is in the model config, regardless of the args, we use the config parameters
    # If a parameter is not in the model config, we use the args parameter

    if "mamba" in args.model:
        return {
            "d_model": cfg["d_model"],
            "n_layer": cfg["n_layer"],
            "vocab_size": cfg["vocab_size"],
            "seq_len": cfg["seq_len"],
        }
    else:
        return Params(
            dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            seq_len=cfg["seq_len"],
            vocab_size=cfg["vocab_size"],
            post_embed_norm=cfg["post_embed_norm"],
            weight_tying=cfg["weight_tying"],
            norm_type=get_norm_class(cfg.get("model_norm", args.model_norm)),
            apply_qk_norm=cfg.get("qk_norm", args.qk_norm),
            positional_embedding_type=cfg.get("positional_embedding_type", args.positional_embedding_type),
            ffn_type=cfg.get("ffn_type", args.ffn_type),
            moe_num_experts=cfg.get("moe_num_experts", args.moe_num_experts),
            moe_loss_weight=cfg.get("moe_loss_weight", args.moe_loss_weight),
            moe_expert_model_parallelism=cfg.get("moe_expert_model_parallelism", args.moe_expert_model_parallelism),
            moe_weight_parallelism=cfg.get("moe_weight_parallelism", args.moe_weight_parallelism),
            moe_capacity_factor=cfg.get("moe_capacity_factor", args.moe_capacity_factor),
            moe_freq=cfg.get("moe_freq", args.moe_freq),
            moe_top_k=cfg.get("moe_top_k", args.moe_top_k),
        )


class Mamba(nn.Module):
    # Experimental architecture, please "pip install mamba-ssm"
    # https://arxiv.org/abs/2312.00752
    def __init__(self, params):
        if MambaLMHeadModel is None:
            raise ImportError(
                "MambaLMHeadModel is not available. Please install the 'mamba_ssm' package by running 'pip install mamba-ssm'."
            )

        super().__init__()
        self.seq_len = params.pop("seq_len")
        self.vocab_size = params["vocab_size"]

        self.model = MambaLMHeadModel(**params)

    def reset_parameters(self):
        return

    def forward(self, x):
        out = self.model(x).logits
        return out, None


def create_model(args):
    if "mamba" in args.model:
        model = Mamba(create_params(args))
        return model
    else:
        model = Transformer(create_params(args))
        return model
