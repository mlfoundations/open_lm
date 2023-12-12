import math
import json
import re
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass
import functools

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import xformers.ops as xops

from huggingface_hub import PyTorchModelHubMixin

from open_lm.norms import get_norm_class
from open_lm.positional_embedding.head_rotary import HeadRotaryWithCast
from open_lm.positional_embedding.rotary import RotaryWithCast
from open_lm.positional_embedding.llama_rotary import LLaMARotaryWithCast
# from open_lm.moe.mixture_of_experts import MoE
from open_lm.moe.megablocks_moe import MoE, get_load_balancing_loss
from megablocks.layers.arguments import Arguments as MoEArgs


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
    moe_dropout: float = 0.0
    moe_gate: str = "top_2"
    moe_num_experts: int = 8
    moe_freq : int = 0
    positional_embedding_type: str = "rotary"
    ffn_type: str = "swiglu"


def xformers_attn(queries, keys, values, is_causal):
    # xformers assumes q, k, v are [batch, seq_len, heads, embed_dim]
    mask = None
    if is_causal:
        mask = xops.LowerTriangularMask()
    return xops.memory_efficient_attention(queries, keys, values, attn_bias=mask)


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
        
        self.attn_fn = xformers_attn
        self.apply_qk_norm = args.apply_qk_norm

        # initialize weights by trunc_normal(1/sqrt(fan_in))
        std = 1.0 / math.sqrt(args.dim)
        torch.nn.init.trunc_normal_(self.in_proj.weight, std=std, a=-3 * std, b=3 * std)
        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = std / math.sqrt(2 * (layer_id + 1))
        torch.nn.init.trunc_normal_(self.out_proj.weight, std=std, a=-3 * std, b=3 * std)

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

class SwitchMLP(nn.Module):
    """
    Routes input to one of N MLP "experts"
    """
    def __init__(self, args):
        super().__init__()
        self.router = torch.nn.Linear(args.dim, args.moe_num_experts)
        self.experts = torch.nn.ModuleList()
        hidden_dim = 256 * ((int(2 * 4 * args.dim / 3) + 256 - 1) // 256)
        for i in range(args.moe_num_experts):
            self.experts.append(xops.SwiGLU(args.dim, hidden_dim, args.dim, bias=False))

    def forward(self, hidden_states):
        # hidden_states: [s, b, h]
        s = hidden_states.size(0)
        b = hidden_states.size(1)
        h = hidden_states.size(2)
        route = self.router(hidden_states)
        route = torch.nn.functional.softmax(route, dim=2)
        max_prob, max_ind = torch.max(route, dim=2)
        max_prob = torch.unsqueeze(max_prob, 2) # [s b 1]

        # TODO (rprenger) TODO this could be made easier to read
        # Converting [s, b, h] to [s*b, h].
        # Each vector could be routed differently
        hidden_states = hidden_states.view(-1, hidden_states.size(2)) # [s*b h]
        max_prob = max_prob.view(-1, max_prob.size(2)) # [s*b 1]
        max_ind = max_ind.view(-1) # [s*b]

        output_total = torch.empty_like(hidden_states).bfloat16()
        #TODO (rprenger) This does each expert in serial, but it could be parallelized

        for expert_num, expert in enumerate(self.experts):
            local_indices = (max_ind == expert_num).nonzero()
            hidden = hidden_states[local_indices,:]
            output = expert(hidden)
            output_total[local_indices,:] = output

        output_total = output_total*max_prob
        output_total = output_total.view(s, b, h)

        return output_total
    
class Block(nn.Module):
    def __init__(self, layer_id, args: Params):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = CustomAttn(layer_id, args)
        self.moe_num_experts = args.moe_num_experts
        self.moe_dropout = args.moe_dropout
        self.moe_gate = args.moe_gate
        self.ffn_type = args.ffn_type
        if args.ffn_type == "swiglu":
            # this follows llama / lit llama -- go to multiple of 256
            hidden_dim = 256 * ((int(2 * 4 * args.dim / 3) + 256 - 1) // 256)
            self.feed_forward = xops.SwiGLU(args.dim, hidden_dim, args.dim, bias=False)
        elif args.ffn_type == "gelu":
            # Follows mosaic mpt7b, but without a bias.
            hidden_dim = args.dim * 4
            self._ff_w1 = nn.Linear(args.dim, hidden_dim, bias=False)
            self._ff_w2 = nn.Linear(hidden_dim, args.dim, bias=False)
            self.feed_forward = nn.Sequential(self._ff_w1, nn.GELU(approximate="none"), self._ff_w2)
        elif args.ffn_type == "switch_moe":
            self.feed_forward = SwitchMLP(args)
        elif args.ffn_type == "megablocks_moe":
            moe_args = MoEArgs(hidden_size=args.dim,
                               ffn_hidden_size=args.dim * 4,
                               moe_num_experts=self.moe_num_experts,
                               num_layers=6,
                               moe_weight_parallelism=True,
                               moe_top_k = 1,
                               moe_capacity_factor=2,
                               moe_loss_weight=0.1,
                               fp16=False)
            self.feed_forward = MoE(moe_args)
        elif args.ffn_type == "xformers_moe":
            self.feed_forward = MoE(dim_model=args.dim,
                                    dropout=self.moe_dropout,
                                    ffn_type="swiglu",
                                    number_of_experts=self.moe_num_experts,
                                    gate=self.moe_gate)
            def div_by_num_experts(num_experts, tensor):
                return tensor / num_experts
            expert_normalization_term = math.sqrt(self.moe_num_experts)
            for p in self.feed_forward.moe.experts.parameters():
                p.expert = True
                # Scale grads by world_size/pg_size so that grads match the equivalent replicated
                # world size expected within Trainer
                p.register_hook(functools.partial(div_by_num_experts, expert_normalization_term))


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

        if args.ffn_type == "swiglu":
            # initialize weights trunc_normal(1/sqrt(fan_in))
            std = 1.0 / math.sqrt(args.dim)
            torch.nn.init.trunc_normal_(self.feed_forward.w12.weight, std=std, a=-3 * std, b=3 * std)
            # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
            std = 1.0 / math.sqrt(hidden_dim)
            std = std / math.sqrt(2 * (layer_id + 1))
            torch.nn.init.trunc_normal_(self.feed_forward.w3.weight, std=std, a=-3 * std, b=3 * std)
        elif args.ffn_type == "gelu":
            std = 1.0 / math.sqrt(args.dim)
            torch.nn.init.trunc_normal_(self._ff_w1.weight, std=std, a=-3 * std, b=3 * std)

            std = 1.0 / math.sqrt(hidden_dim)
            std = std / math.sqrt(2 * (layer_id + 1))
            torch.nn.init.trunc_normal_(self._ff_w2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x), is_causal=True)
        if self.ffn_type == "megablocks_moe":
            ffn_out, _ = self.feed_forward(self.ffn_norm(h))
        else:
            ffn_out = self.feed_forward(self.ffn_norm(h))
        # if torch.distributed.get_rank() == 0:
        #     print(get_load_balancing_loss()[0])
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
                params.ffn_type = "megablocks_moe"
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

        # initialize weight 1/sqrt(dim)
        # this is 1/fan_in for output, as is default, and Maciej Kilian tried another option
        # for the embed layer (from RWKV paper) but this was better.
        std = 1.0 / math.sqrt(params.dim)
        torch.nn.init.trunc_normal_(self.output.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.tok_embeddings.weight, std=std, a=-3 * std, b=3 * std)

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
        moe_dropout=cfg.get("moe_dropout", args.moe_dropout),
        moe_num_experts=cfg.get("moe_num_experts", args.moe_num_experts),
        moe_gate=cfg.get("moe_gate", args.moe_gate),
        moe_freq=cfg.get("moe_freq", args.moe_freq),
    )


def create_model(args):
    return Transformer(create_params(args))
