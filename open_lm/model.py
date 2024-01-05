import math

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import xformers.ops as xops

from huggingface_hub import PyTorchModelHubMixin

from open_lm.attention import get_attn_func, xformers_attn, torch_attn
from open_lm.norms import get_norm_class
from open_lm.positional_embedding.head_rotary import HeadRotaryWithCast
from open_lm.positional_embedding.rotary import RotaryWithCast
from open_lm.positional_embedding.llama_rotary import LLaMARotaryWithCast
from open_lm.params import Params, create_params


# from open_lm.moe.mixture_of_experts import MoE
try:
    from megablocks.layers.moe import MoE
    from megablocks.layers.arguments import Arguments as MoEArgs
except ImportError:
    import logging

    logging.warning(f"Megablocks not installed. To train MoE, install with pip install megablocks.")

try:  # optional import
    from mamba_ssm import MambaLMHeadModel
except ImportError:
    MambaLMHeadModel = None


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
        self.attn_fn = get_attn_func(
            args.attn_params.name,
            args.attn_params.activation,
            args.attn_params.seq_scalar,
            args.attn_params.seq_scalar_alpha,
        )
        self.apply_qk_norm = args.apply_qk_norm

        # initialize norm layers for queries and keys if needed
        NormClass = get_norm_class(args.norm_name)
        self.q_norm = (
            NormClass(
                args.n_heads * self.head_dim,
                eps=args.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            NormClass(
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
            moe_args = MoEArgs(
                hidden_size=args.dim,
                ffn_hidden_size=args.dim * 4,
                moe_num_experts=args.moe_num_experts,
                moe_weight_parallelism=args.moe_weight_parallelism,
                moe_expert_model_parallelism=args.moe_expert_model_parallelism,
                moe_top_k=args.moe_top_k,
                moe_capacity_factor=args.moe_capacity_factor,
                moe_loss_weight=args.moe_loss_weight,
                device=torch.cuda.current_device(),
                bf16=False,
                fp16=False,
            )
            self.feed_forward = MoE(moe_args)

        self.layer_id = layer_id
        NormClass = get_norm_class(args.norm_name)
        self.attention_norm = NormClass(
            args.dim,
            eps=args.norm_eps,
        )
        self.ffn_norm = NormClass(
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
        if self._ffn_type == "moe":
            ffn_out, _ = self.feed_forward(self.ffn_norm(h))
        else:
            ffn_out = self.feed_forward(self.ffn_norm(h))
        out = h + ffn_out
        return out, past_key_value


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
        NormClass = get_norm_class(params.norm_name)
        self.post_embed_norm = (
            NormClass(
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
        self.norm = NormClass(
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
        return out, None, None


def create_model(args):
    if "mamba" in args.model:
        model = Mamba(create_params(args))
        return model
    else:
        model = Transformer(create_params(args))
        return model
