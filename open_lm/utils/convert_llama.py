"""
This script converts the weights from LLAMA to OpenLM compatible weights.
Usage: `python convert_llama_to_openlm.py <llama_weight_path> <openlm_weight_path>`
"""

import torch
import sys


def convert(llama_state_dict: dict) -> dict:
    openlm_state_dict = {}

    n_layer = len(set([key.split(".")[1] for key in llama_state_dict if "layers." in key]))
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
        openlm_state_dict[tgt_key] = torch.cat([llama_state_dict[src_key_1], llama_state_dict[src_key_2]], dim=0)

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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: `python convert_llama_to_openlm.py <llama_weight_path> <openlm_weight_path>`")
        sys.exit(1)
    llama_state_dict = torch.load(sys.argv[1])
    openlm_state_dict = {"state_dict": convert(llama_state_dict)}
    torch.save(openlm_state_dict, sys.argv[2])
