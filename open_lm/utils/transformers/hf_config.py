from dataclasses import fields
from typing import List, Optional, Dict
from argparse import Namespace

from transformers import PretrainedConfig

from open_lm.model import Params, create_params


class OpenLMConfig(PretrainedConfig):
    model_type = "openlm"

    def __init__(
        self,
        params: Optional[Params] = None,
        params_args: Optional[Namespace] = None,
        params_args_dict: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize the HFConfig class. Any of the three arguments can be used to initialize the class.
        Note that the instance can get serialized when passing in either params_args or params_args_dict.

        Args:
            params (Optional[Params]): The parameters object.
            params_args (Optional[Namespace]): The namespace object containing the parameters arguments.
            params_args_dict (Optional[Dict]): The dictionary containing the parameters arguments.
            **kwargs: Additional keyword arguments.
        """
        # Used by huggingface transformers
        super().__init__(**kwargs)

        if params_args is not None:
            params_args_dict = vars(params_args)

        self.params_args_dict = params_args_dict

        if params is not None:
            self.params = params

    def set_params(self, params: Params):
        self.tie_word_embeddings = params.weight_tying
        for field in fields(Params):
            setattr(self, field.name, getattr(params, field.name))
