from dataclasses import fields
from typing import List, Optional

from transformers import PretrainedConfig

from open_lm.model import Params


class OpenLMConfig(PretrainedConfig):
    model_type = "openlm"

    def __init__(self, params: Optional[Params] = None, **kwargs):
        # Used by huggingface transformers
        super().__init__(**kwargs)
        if params is not None:
            self.tie_word_embeddings = params.weight_tying
            for field in fields(Params):
                setattr(self, field.name, getattr(params, field.name))
