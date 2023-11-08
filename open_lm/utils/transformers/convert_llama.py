"""
This script converts the weights from LLAMA to OpenLM compatible weights.
Usage: `python convert_llama_to_openlm.py <llama_weight_path> <openlm_weight_path>`
"""

import os
import argparse
import torch
import sys
from transformers import AutoModelForCausalLM


def transform_qk(qk_weight):
    qk_split_128 = torch.stack(qk_weight.split(128))
    qk_split_2_64 = qk_split_128.view(qk_split_128.shape[0], 2, 64, qk_split_128.shape[-1])
    return qk_split_2_64.transpose(1,2).reshape(qk_split_128.shape[0]*qk_split_128.shape[1], qk_split_128.shape[2])


def convert(llama_state_dict: dict) -> dict:
    openlm_state_dict = {}

    n_layer = len(
        set([key.split(".")[1] for key in llama_state_dict if "layers." in key])
    )
    print(f"n_layer: {n_layer}")

    for key in ["tok_embeddings.weight", "norm.weight", "output.weight"]:
        value = llama_state_dict[key]
        assert key not in openlm_state_dict
        openlm_state_dict[key] = value

    for i in range(n_layer):
        src_key_1, src_key_2, src_key_3 = (
            f"layers.{i}.attention.wq.weight",
            f"layers.{i}.attention.wk.weight",
            f"layers.{i}.attention.wv.weight",
        )
        tgt_key = f"layers.{i}.attention.in_proj.weight"
        assert tgt_key not in openlm_state_dict
        openlm_state_dict[tgt_key] = torch.cat(
            [
                llama_state_dict[src_key_1],
                llama_state_dict[src_key_2],
                llama_state_dict[src_key_3],
            ],
            dim=0,
        )

        src_key = f"layers.{i}.attention.wo.weight"
        tgt_key = f"layers.{i}.attention.out_proj.weight"
        assert tgt_key not in openlm_state_dict
        openlm_state_dict[tgt_key] = llama_state_dict[src_key]

        src_key_1, src_key_2 = (
            f"layers.{i}.feed_forward.w1.weight",
            f"layers.{i}.feed_forward.w3.weight",
        )
        tgt_key = f"layers.{i}.feed_forward.w12.weight"
        assert tgt_key not in openlm_state_dict
        openlm_state_dict[tgt_key] = torch.cat(
            [llama_state_dict[src_key_1], llama_state_dict[src_key_2]], dim=0
        )

        src_key = f"layers.{i}.feed_forward.w2.weight"
        tgt_key = f"layers.{i}.feed_forward.w3.weight"
        assert tgt_key not in openlm_state_dict
        openlm_state_dict[tgt_key] = llama_state_dict[src_key]

        tgt_key = f"layers.{i}.attention_norm.weight"
        assert tgt_key not in openlm_state_dict
        openlm_state_dict[tgt_key] = llama_state_dict[tgt_key]

        tgt_key = f"layers.{i}.ffn_norm.weight"
        assert tgt_key not in openlm_state_dict
        openlm_state_dict[tgt_key] = llama_state_dict[tgt_key]

    return openlm_state_dict


def convert_v2(llama_state_dict: dict) -> dict:
    """
    This function assumes the LLaMA model is loaded using HF.
    """
    openlm_state_dict = {}

    n_layer = len(
        set([key.split(".")[2] for key in llama_state_dict if "layers." in key])
    )
    print(f"n_layer: {n_layer}")

    openlm_state_dict["tok_embeddings.weight"] = llama_state_dict["model.embed_tokens.weight"]
    openlm_state_dict["norm.weight"] = llama_state_dict["model.norm.weight"]
    openlm_state_dict["output.weight"] = llama_state_dict["lm_head.weight"]

    for i in range(n_layer):
        src_key_1, src_key_2, src_key_3 = (
            f"model.layers.{i}.self_attn.q_proj.weight",
            f"model.layers.{i}.self_attn.k_proj.weight",
            f"model.layers.{i}.self_attn.v_proj.weight",
        )
        tgt_key = f"layers.{i}.attention.in_proj.weight"
        assert tgt_key not in openlm_state_dict
        openlm_state_dict[tgt_key] = torch.cat(
            [
                transform_qk(llama_state_dict[src_key_1]),
                transform_qk(llama_state_dict[src_key_2]),
                llama_state_dict[src_key_3],
            ],
            dim=0,
        )

        src_key = f"model.layers.{i}.self_attn.o_proj.weight"
        tgt_key = f"layers.{i}.attention.out_proj.weight"
        assert tgt_key not in openlm_state_dict
        openlm_state_dict[tgt_key] = llama_state_dict[src_key]

        src_key_1, src_key_2 = (
            f"model.layers.{i}.mlp.gate_proj.weight",
            f"model.layers.{i}.mlp.up_proj.weight",
        )
        tgt_key = f"layers.{i}.feed_forward.w12.weight"
        assert tgt_key not in openlm_state_dict
        openlm_state_dict[tgt_key] = torch.cat(
            [llama_state_dict[src_key_1], llama_state_dict[src_key_2]], dim=0
        )

        src_key = f"model.layers.{i}.mlp.down_proj.weight"
        tgt_key = f"layers.{i}.feed_forward.w3.weight"
        assert tgt_key not in openlm_state_dict
        openlm_state_dict[tgt_key] = llama_state_dict[src_key]

        src_key = f"model.layers.{i}.input_layernorm.weight"
        tgt_key = f"layers.{i}.attention_norm.weight"
        assert tgt_key not in openlm_state_dict
        openlm_state_dict[tgt_key] = llama_state_dict[src_key]

        src_key = f"model.layers.{i}.post_attention_layernorm.weight"
        tgt_key = f"layers.{i}.ffn_norm.weight"
        assert tgt_key not in openlm_state_dict
        openlm_state_dict[tgt_key] = llama_state_dict[src_key]

    return openlm_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf', help='HF pretrained model')
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    ## Use this if the --model is a .pt or .pth file
    # llama_state_dict = torch.load(args.model)
    # openlm_state_dict = {"state_dict": convert(llama_state_dict)}

    ## Use this if the --model is a HugginFace model
    llama_state_dict = AutoModelForCausalLM.from_pretrained(args.model).state_dict()
    openlm_state_dict = {"state_dict": convert_v2(llama_state_dict)}
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    torch.save(openlm_state_dict, f"{args.output_dir}/model_openlm.pt")
