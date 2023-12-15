import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class HfWrapper(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            torch_dtype=torch.bfloat16,
        )

        self.params = self.model.config
        self.vocab_size = self.model.config.vocab_size
        self.seq_len = args.hf_seq_len

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        if enable:
            self.model.gradient_checkpointing_enable()
        else:
            self.model.gradient_checkpointing_disable()

    def forward(self, input):
        return self.model(input_ids=input)[0], None


def create_wrapped_hf_model(hf_model_name):
    return HfWrapper(hf_model_name)
