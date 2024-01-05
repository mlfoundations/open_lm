from dataclasses import fields
from typing import List

from transformers import PretrainedConfig

from open_lm.params import Params


class OpenLMConfig(PretrainedConfig):
    model_type = "openlm"

    def __init__(self, params: Params, **kwargs):
        # Used by huggingface transformers
        super().__init__(**kwargs)
        self.tie_word_embeddings = params.weight_tying
        for field in fields(Params):
            setattr(self, field.name, getattr(params, field.name))
