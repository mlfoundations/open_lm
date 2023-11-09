import torch
from transformers import GPTNeoXTokenizerFast
from tests.shared import MockTrainArgs
from open_lm.model import create_model


def test_resize_token_embeddings():
    args = MockTrainArgs("open_lm_11m")

    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    num_tokens_old = len(tokenizer)
    for i in range(300):
        tokenizer.add_tokens(f"<<{str(i)}>>")

    model = create_model(args).to(args.device)
    emb_in_old = model.get_input_embeddings().weight
    emb_out_old = model.get_output_embeddings().weight

    model.resize_token_embeddings(len(tokenizer))

    new_shape_in = model.get_input_embeddings().weight.shape
    new_shape_out = model.get_output_embeddings().weight.shape

    # Check shapes
    assert num_tokens_old + 300 == new_shape_in[0]
    assert num_tokens_old + 300 == new_shape_out[0]

    # Check values
    assert torch.sum(model.get_input_embeddings().weight[: emb_in_old.shape[0], :] - emb_in_old) < 0.00001
    assert torch.sum(model.get_output_embeddings().weight[: emb_out_old.shape[0], :] - emb_out_old) < 0.00001
