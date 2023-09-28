import math
import json
import re
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import xformers.ops as xops

from open_lm.norms import get_norm_class
from open_lm.positional_embedding.head_rotary import HeadRotaryWithCast
from open_lm.positional_embedding.rotary import RotaryWithCast

# from openclip
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {
        k: v
        for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }


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
    rotary_old: bool = False


def xformers_attn(queries, keys, values, is_causal):
    # xformers assumes q, k, v are [batch, seq_len, heads, embed_dim]
    mask = None
    if is_causal:
        mask = xops.LowerTriangularMask()
    return xops.memory_efficient_attention(queries, keys, values, attn_bias=mask)


class CustomAttn(nn.Module):
    def __init__(self, layer_id, args: Params):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.in_proj = nn.Linear(args.dim, 3 * args.n_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.pos_embed = HeadRotaryWithCast(self.head_dim, args.seq_len) if args.rotary_old else RotaryWithCast(self.head_dim, args.seq_len)
        self.attn_fn = xformers_attn
        self.apply_qk_norm = args.apply_qk_norm

        # initialize weights by trunc_normal(1/sqrt(fan_in))
        std = 1.0 / math.sqrt(args.dim)
        torch.nn.init.trunc_normal_(self.in_proj.weight, std=std, a=-3 * std, b=3 * std)
        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = std / math.sqrt(2 * (layer_id + 1))
        torch.nn.init.trunc_normal_(
            self.out_proj.weight, std=std, a=-3 * std, b=3 * std
        )

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

    def forward(self, x: torch.Tensor, is_causal=True):
        batchsize, seqlen, _ = x.shape
        queries, keys, vals = self.in_proj(x).chunk(3, dim=-1)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = queries.view(batchsize, seqlen, self.n_heads, self.head_dim)
        keys = keys.view(batchsize, seqlen, self.n_heads, self.head_dim)
        vals = vals.view(batchsize, seqlen, self.n_heads, self.head_dim)

        queries, keys, vals = self.pos_embed(queries, keys, vals)

        output = self.attn_fn(queries, keys, vals, is_causal=is_causal)

        output = output.view(batchsize, seqlen, -1)

        return self.out_proj(output)


class Block(nn.Module):
    def __init__(self, layer_id, args: Params):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = CustomAttn(layer_id, args)

        # this follows llama / lit llama -- go to multiple of 256
        hidden_dim = 256 * ((int(2 * 4 * args.dim / 3) + 256 - 1) // 256)

        self.feed_forward = xops.SwiGLU(args.dim, hidden_dim, args.dim, bias=False)
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

        # initialize weights trunc_normal(1/sqrt(fan_in))
        std = 1.0 / math.sqrt(args.dim)
        torch.nn.init.trunc_normal_(
            self.feed_forward.w12.weight, std=std, a=-3 * std, b=3 * std
        )
        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = 1.0 / math.sqrt(hidden_dim)
        std = std / math.sqrt(2 * (layer_id + 1))
        torch.nn.init.trunc_normal_(
            self.feed_forward.w3.weight, std=std, a=-3 * std, b=3 * std
        )

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x), is_causal=True)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        # for convenience we often share param names with llama
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
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
        for layer_id in range(params.n_layers):
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

        # initialize weight 1/sqrt(dim)
        # this is 1/fan_in for output, as is default, and Maciej Kilian tried another option
        # for the embed layer (from RWKV paper) but this was better.
        std = 1.0 / math.sqrt(params.dim)
        torch.nn.init.trunc_normal_(self.output.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(
            self.tok_embeddings.weight, std=std, a=-3 * std, b=3 * std
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(self, input):
        x = self.tok_embeddings(input)
        x = self.post_embed_norm(x)

        for layer in self.layers:
            if self.grad_checkpointing:
                x = checkpoint(layer, x)
            else:
                x = layer(x)

        x = self.norm(x)
        output = self.output(x)
        # follow llama in casting this to float.
        return output.float(), x

    def get_input_embeddings(self):
        return self.tok_embeddings

    def get_output_embeddings(self):
        return self.output


def create_params(args):
    cfg = deepcopy(_MODEL_CONFIGS[args.model])
    return Params(
        dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        seq_len=cfg["seq_len"],
        vocab_size=cfg["vocab_size"],
        post_embed_norm=cfg["post_embed_norm"],
        weight_tying=cfg["weight_tying"],
        norm_type=get_norm_class(args.model_norm),
        apply_qk_norm=args.qk_norm,
        rotary_old=args.rotary_old
    )


def create_model(args):
    return Transformer(create_params(args))
