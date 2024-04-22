from argparse import Namespace
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from open_lm.utils.transformers.hf_config import OpenLMConfig
from open_lm.model import Transformer, create_params
import torch
import torch.nn as nn
from typing import Union, Tuple, Optional, List
import os


class OpenLMModel(PreTrainedModel):
    config_class = OpenLMConfig

    def __init__(self, config):
        # This has to be done before init as it sets makes sure hf config is correct
        if hasattr(config, "params"):
            params = config.params
        else:
            params = create_params(Namespace(**config.params_args_dict))
        config.set_params(params)
        super().__init__(config)

        self.supports_gradient_checkpointing = True
        self.model = Transformer(params)

    @property
    def gradient_checkpointing(self):
        return self.model.grad_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self.model.grad_checkpointing = value

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        return self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)


class OpenLMforCausalLM(OpenLMModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.lm_head = None
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        raise NotImplementedError

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, OpenLlamaForCausalLM
        >>> model = OpenLlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""
        assert position_ids is None, "Position IDs are not supported"
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        logits, _, past_key_values = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        output = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values, loss=loss)
        return output

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[1]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_cache = ()
        for layer_past in past_key_values:
            reordered_cache += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_cache

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        if (
            os.path.isdir(pretrained_model_name_or_path)
            and kwargs.get("config", None) is not None
            and getattr(kwargs["config"], "checkpoint_file", None) is not None
        ):
            # Setting torch default dtype
            torch_dtype = getattr(kwargs["config"], "torch_dtype", None)
            if isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype)
            if torch_dtype is not None:
                torch.set_default_dtype(torch_dtype)

            print("Loading checkpoint from directory")
            checkpoint_path = kwargs["config"].checkpoint_file
            checkpoint = torch.load(checkpoint_path)

            state_dict = checkpoint["state_dict"]
            state_dict = {x.replace("module.", ""): y for x, y in state_dict.items()}
            state_dict = {f"model.{x}": y for x, y in state_dict.items()}

            return super().from_pretrained(None, state_dict=state_dict, **kwargs)
        else:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


if __name__ == "__main__":
    openlm_config = OpenLMConfig.from_pretrained("utils/transformers/open_lm_config")
    print(OpenLMModel(openlm_config))
